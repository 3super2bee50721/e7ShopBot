import tkinter as tk
from tkinter import messagebox
import threading
import time
import random
import logging
import cv2
import numpy as np
from PIL import ImageGrab
import pyautogui
from datetime import datetime


# ---------- 全局状态变量 ----------
is_running = False
stop_flag = False
refresh_count = 0
bookmark_groups = 0   # 圣约书签购买组数
medal_groups = 0      # 神秘奖牌购买组数

# ========== 全局配置 ==========

TEMPLATE_DIR = "./templates/"
TEMPLATES = {
    "bookmark": cv2.imread(TEMPLATE_DIR + "saint_bookmark.png", cv2.IMREAD_GRAYSCALE),
    "medal": cv2.imread(TEMPLATE_DIR + "mystic_medal.png", cv2.IMREAD_GRAYSCALE),
    "bookmark_text": cv2.imread(TEMPLATE_DIR + "bookmark_text.png", cv2.IMREAD_GRAYSCALE),
    "medal_text": cv2.imread(TEMPLATE_DIR + "medal_text.png", cv2.IMREAD_GRAYSCALE),
    "buy_btn": cv2.imread(TEMPLATE_DIR + "buy_btn.png", cv2.IMREAD_GRAYSCALE),
    "confirm_buy": cv2.imread(TEMPLATE_DIR + "confirm_buy.png", cv2.IMREAD_GRAYSCALE),
    "refresh_btn": cv2.imread(TEMPLATE_DIR + "refresh_btn.png", cv2.IMREAD_GRAYSCALE),
    "confirm_refresh": cv2.imread(TEMPLATE_DIR + "confirm_refresh.png", cv2.IMREAD_GRAYSCALE),
    "available": cv2.imread(TEMPLATE_DIR + "available.png", cv2.IMREAD_GRAYSCALE),      # 新增
    "unavailable": cv2.imread(TEMPLATE_DIR + "unavailable.png", cv2.IMREAD_GRAYSCALE),  # 新增
}

CONFIDENCE = 0.85
SCROLL_AMOUNT = 300

# 偏移参数（需根据实际微调）
BUY_OFFSET_X = 600
BUY_OFFSET_Y = 40
SEARCH_RADIUS = 30

# 价格配置
PRICE_BOOKMARK = 184000
PRICE_MEDAL = 280000
REFRESH_COST = 3


# 随机延迟范围（秒）
DELAY_RANGE = {
    "click": (0.2, 0.6),
    "scroll": (0.1, 0.3),
    "between_loop": (1.0, 2.0),
    "wait_template": (0.1, 0.3),
    "purchase_interval": (0.3, 0.8),
}

# ---------- 日志配置 ----------
LOG_FILE = "purchase_log.txt"
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


# ---------- 辅助函数 ----------

def check_buy_status(item_pos, confidence=0.75):
    """
    检测道具的购买状态
    返回: 'available' (可购买), 'unavailable' (已购买/0/1), 或 None (未检测到)
    """
    x, y, w, h = item_pos
    # 搜索区域：从左上角向右360~610像素，高度与道具相同（可微调）
    region_x = x + 360
    region_y = y - 5
    region_w = 250
    region_h = h + 10
    screen = capture_screen(None)
    h_screen, w_screen = screen.shape
    region_x = max(0, region_x)
    region_y = max(0, region_y)
    region_w = min(region_w, w_screen - region_x)
    region_h = min(region_h, h_screen - region_y)
    roi = screen[region_y:region_y+region_h, region_x:region_x+region_w]
    if roi.shape[0] == 0 or roi.shape[1] == 0:
        return None

    available_template = TEMPLATES.get("available")
    unavailable_template = TEMPLATES.get("unavailable")
    if available_template is None or unavailable_template is None:
        logging.warning("状态模板未加载")
        return None

    # 检测 "1/1"
    res1 = cv2.matchTemplate(roi, available_template, cv2.TM_CCOEFF_NORMED)
    _, max1, _, _ = cv2.minMaxLoc(res1)
    # 检测 "0/1"
    res0 = cv2.matchTemplate(roi, unavailable_template, cv2.TM_CCOEFF_NORMED)
    _, max0, _, _ = cv2.minMaxLoc(res0)

    logging.debug(f"1/1匹配值: {max1:.3f}, 0/1匹配值: {max0:.3f}")

    if max1 >= confidence and max1 > max0:
        return 'available'
    elif max0 >= confidence and max0 > max1:
        return 'unavailable'
    else:
        return None

def check_text_in_region(item_pos, text_key, confidence=0.8):
    """
    在道具左上角向右300像素的矩形区域内验证文字
    item_pos: (x,y,w,h)
    """
    x, y, w, h = item_pos
    # 文字区域：从左上角开始，宽度300，高度=h
    region_rect = (x, y, 300, h)
    screen = capture_screen(None)
    template = TEMPLATES.get(text_key)
    if template is None:
        return False
    x0, y0, w0, h0 = region_rect
    roi = screen[y0:y0+h0, x0:x0+w0]
    if roi.shape[0] < template.shape[0] or roi.shape[1] < template.shape[1]:
        return False
    result = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    return max_val >= confidence


def random_delay(category):
    if category in DELAY_RANGE:
        min_t, max_t = DELAY_RANGE[category]
        time.sleep(random.uniform(min_t, max_t))

def capture_screen(region=None):
    screenshot = ImageGrab.grab(bbox=region) if region else ImageGrab.grab()
    return cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2GRAY)

def find_template_in_region(image_gray, template, region_rect):
    if template is None:
        return None
    x, y, w, h = region_rect
    roi = image_gray[y:y+h, x:x+w]
    if roi.shape[0] < template.shape[0] or roi.shape[1] < template.shape[1]:
        return None
    result = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    if max_val >= CONFIDENCE:
        center_x = x + max_loc[0] + template.shape[1] // 2
        center_y = y + max_loc[1] + template.shape[0] // 2
        return (center_x, center_y)
    return None

def find_all_item_positions(item_key, region=None):
    screen = capture_screen(region)
    template = TEMPLATES.get(item_key)
    if template is None:
        return []
    result = cv2.matchTemplate(screen, template, cv2.TM_CCOEFF_NORMED)
    locations = np.where(result >= CONFIDENCE)
    h, w = template.shape
    points = []
    for pt in zip(*locations[::-1]):
        points.append((pt[0], pt[1], w, h))
    return points


def click_buy_for_item(item_pos, item_type, region=None):
    x, y, w, h = item_pos

    # ---- 1. 文字验证 ----
    text_key = "bookmark_text" if item_type == "bookmark" else "medal_text"
    if not check_text_in_region(item_pos, text_key):
        logging.warning(f"道具 ({x},{y}) 文字验证失败，跳过")
        return False

    # ---- 2. 购买状态检测 ----
    status = check_buy_status(item_pos)
    if status == 'unavailable':
        logging.info(f"道具 ({x},{y}) 状态为 0/1，已购买，跳过")
        return False
    elif status != 'available':
        # 如果未检测到明确状态，可尝试二次检测（等待0.5秒重试）
        time.sleep(0.5)
        status = check_buy_status(item_pos)
        if status != 'available':
            logging.warning(f"道具 ({x},{y}) 状态不确定，跳过")
            return False

    # ---- 3. 点击购买（坐标：左下角向右600像素） ----
    click_x = x + 600 + random.randint(-5, 5)
    click_y = y + h + random.randint(-5, 5)
    screen_w, screen_h = pyautogui.size()
    click_x = max(0, min(screen_w, click_x))
    click_y = max(0, min(screen_h, click_y))

    time.sleep(3)   # 购买前等待

    pyautogui.moveTo(click_x, click_y, duration=random.uniform(0.05, 0.15))
    pyautogui.click()
    random_delay("click")

    if wait_and_click_template("confirm_buy", timeout=3.5, region=region):
        return True
    else:
        logging.warning("购买确认弹窗未出现")
        return False

def wait_and_click_template(template_name, timeout=3.5, region=None):
    start = time.time()
    while time.time() - start < timeout:
        screen = capture_screen(region)
        template = TEMPLATES.get(template_name)
        if template is not None:
            result = cv2.matchTemplate(screen, template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            if max_val >= CONFIDENCE:
                h, w = template.shape
                center = (max_loc[0] + w//2, max_loc[1] + h//2)
                pyautogui.moveTo(center[0] + random.randint(-3, 3), center[1] + random.randint(-3, 3), duration=random.uniform(0.05, 0.15))
                pyautogui.click()
                random_delay("click")
                return True
        random_delay("wait_template")
    return False

def click_template_direct(template_name, region=None):
    screen = capture_screen(region)
    template = TEMPLATES.get(template_name)
    if template is None:
        return False
    result = cv2.matchTemplate(screen, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    if max_val >= CONFIDENCE:
        h, w = template.shape
        center = (max_loc[0] + w//2, max_loc[1] + h//2)
        pyautogui.moveTo(center[0] + random.randint(-3, 3), center[1] + random.randint(-3, 3), duration=random.uniform(0.05, 0.15))
        pyautogui.click()
        random_delay("click")
        return True
    return False

def scan_and_purchase(region=None):
    global bookmark_groups, medal_groups   # 修正变量名
    all_items = []
    for key in ["bookmark", "medal"]:
        positions = find_all_item_positions(key, region)
        for (x, y, w, h) in positions:
            all_items.append((key, x, y, w, h))
    if not all_items:
        logging.debug("当前区域未检测到任何目标道具")
        return False
    all_items.sort(key=lambda t: t[2])
    purchased_any = False
    for item_key, x, y, w, h in all_items:
        logging.info(f"尝试购买 {item_key} 在 ({x},{y})")
        success = click_buy_for_item((x, y, w, h), item_key, region)
        if success:
            purchased_any = True
            if item_key == "bookmark":
                bookmark_groups += 1
                logging.info(f"✅ 成功购买 圣约书签 (累计{bookmark_groups}组)")
            else:
                medal_groups += 1
                logging.info(f"✅ 成功购买 神秘奖牌 (累计{medal_groups}组)")
            random_delay("purchase_interval")
        else:
            logging.warning(f"购买 {item_key} 失败")
    return purchased_any

# ---------- 自动化主循环（在后台线程运行） ----------
def automation_loop(max_refresh):
    global is_running, stop_flag, refresh_count
    logging.info("自动化线程启动")
    time.sleep(3)   # 等待3秒，让您切换窗口

    # ----- 点击固定坐标 (1455, 700) 激活窗口 -----
    pyautogui.click(1455, 700)
    time.sleep(0.5)

    region = None  # 可自定义窗口区域

    while is_running and not stop_flag and refresh_count < max_refresh:
        try:
            # ----- 先扫描当前商店，尝试购买 -----
            logging.info("扫描当前可见区域...")
            purchased = scan_and_purchase(region)
            # 向下滚动一次
            pyautogui.scroll(-SCROLL_AMOUNT)
            time.sleep(0.5)
            logging.info("扫描滚动后的区域...")
            purchased2 = scan_and_purchase(region)
            # 如果购买到了任何道具，则本轮结束（不刷新）
            if purchased or purchased2:
                logging.info("本轮已购买道具，跳过刷新")
                random_delay("between_loop")
                continue

            # ----- 没有可购买道具，执行刷新 -----
            logging.info("未检测到可购买道具，执行刷新...")
            if not click_template_direct("refresh_btn", region):
                logging.warning("未找到刷新按钮")
                time.sleep(2)
                continue
            # 刷新确认前延迟0.5秒
            time.sleep(0.5)
            if not wait_and_click_template("confirm_refresh", timeout=12.0, region=region):
                logging.warning("刷新确认弹窗未出现")
                time.sleep(2)
                continue

            refresh_count += 1
            logging.info(f"刷新成功 (第{refresh_count}次)")

            # 刷新后等待加载（增加额外0.75秒）
            time.sleep(random.uniform(2.0, 3.0) + 0.75)

            # 点击固定坐标 (1455,700)
            pyautogui.click(1455, 700)
            time.sleep(0.5)

            # 扫描刷新后的商店（同样两次）
            logging.info("扫描当前可见区域...")
            scan_and_purchase(region)
            pyautogui.scroll(-SCROLL_AMOUNT)
            time.sleep(0.5)
            logging.info("扫描滚动后的区域...")
            scan_and_purchase(region)

            random_delay("between_loop")

        except Exception as e:
            logging.error(f"自动化循环异常: {e}")
            time.sleep(2)

    if refresh_count >= max_refresh:
        logging.info("达到最大刷新次数，自动停止")
    elif stop_flag:
        logging.info("用户手动停止")
    is_running = False

# ---------- GUI 应用程序 ----------
class AutoShopApp:
    def __init__(self, root):
        self.root = root
        root.title("秘密商店自动化助手")
        root.geometry("500x250")
        root.resizable(False, False)

        # 变量
        self.max_refresh_var = tk.IntVar(value=100)
        self.refresh_count_var = tk.StringVar(value="0")
        self.bookmark_var = tk.StringVar(value="0")
        self.medal_var = tk.StringVar(value="0")
        self.bookmark_prob_var = tk.StringVar(value="0.00%")
        self.medal_prob_var = tk.StringVar(value="0.00%")
        self.status_var = tk.StringVar(value="就绪")

        # 第一行：最大刷新次数 + 按钮
        frame_top = tk.Frame(root)
        frame_top.pack(pady=10)

        tk.Label(frame_top, text="最大刷新次数:").pack(side=tk.LEFT, padx=5)
        self.max_entry = tk.Entry(frame_top, textvariable=self.max_refresh_var, width=8)
        self.max_entry.pack(side=tk.LEFT, padx=5)

        self.start_btn = tk.Button(frame_top, text="开始", command=self.start_auto, width=8)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        self.stop_btn = tk.Button(frame_top, text="停止", command=self.stop_auto, width=8, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)

        # 第二行：统计数据
        frame_stats = tk.Frame(root)
        frame_stats.pack(pady=5)

        tk.Label(frame_stats, text="已刷新:").pack(side=tk.LEFT, padx=5)
        tk.Label(frame_stats, textvariable=self.refresh_count_var, width=6, relief=tk.SUNKEN).pack(side=tk.LEFT)
        tk.Label(frame_stats, text="圣约书签:").pack(side=tk.LEFT, padx=5)
        tk.Label(frame_stats, textvariable=self.bookmark_var, width=6, relief=tk.SUNKEN).pack(side=tk.LEFT)
        tk.Label(frame_stats, text="神秘奖牌:").pack(side=tk.LEFT, padx=5)
        tk.Label(frame_stats, textvariable=self.medal_var, width=6, relief=tk.SUNKEN).pack(side=tk.LEFT)

        # 第三行：概率
        frame_prob = tk.Frame(root)
        frame_prob.pack(pady=5)

        tk.Label(frame_prob, text="书签概率:").pack(side=tk.LEFT, padx=5)
        tk.Label(frame_prob, textvariable=self.bookmark_prob_var, width=8, relief=tk.SUNKEN).pack(side=tk.LEFT)
        tk.Label(frame_prob, text="奖牌概率:").pack(side=tk.LEFT, padx=5)
        tk.Label(frame_prob, textvariable=self.medal_prob_var, width=8, relief=tk.SUNKEN).pack(side=tk.LEFT)

        # 第四行：状态日志
        frame_status = tk.Frame(root)
        frame_status.pack(pady=10)

        tk.Label(frame_status, text="状态:").pack(side=tk.LEFT, padx=5)
        self.status_label = tk.Label(frame_status, textvariable=self.status_var, relief=tk.SUNKEN, width=40, anchor=tk.W)
        self.status_label.pack(side=tk.LEFT, padx=5)

        # 检查模板是否存在
        missing = [name for name, img in TEMPLATES.items() if img is None]
        if missing:
            msg = f"以下模板图片未加载：{missing}\n请检查 ./templates/ 目录"
            messagebox.showwarning("模板缺失", msg)
            self.status_var.set("模板缺失，请检查")
        else:
            self.status_var.set("就绪")

        # 线程控制
        self.thread = None

        # 启动定时刷新（每500ms更新一次）
        self.schedule_update()

    # ---------- 定时刷新 ----------
    def schedule_update(self):
        """每500毫秒自动刷新UI，由主线程定时调用"""
        self.update_ui()
        self.root.after(500, self.schedule_update)

    def update_ui(self):
        """更新界面统计数据（在主线程中执行）"""
        self.refresh_count_var.set(str(refresh_count))
        self.bookmark_var.set(str(bookmark_groups))
        self.medal_var.set(str(medal_groups))

        if refresh_count > 0:
            bk_prob = (bookmark_groups / refresh_count) / 6 * 100
            med_prob = (medal_groups / refresh_count) / 6 * 100
            self.bookmark_prob_var.set(f"{bk_prob:.2f}%")
            self.medal_prob_var.set(f"{med_prob:.2f}%")
        else:
            self.bookmark_prob_var.set("0.00%")
            self.medal_prob_var.set("0.00%")

        # 更新按钮状态
        if is_running:
            self.start_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)
            self.max_entry.config(state=tk.DISABLED)
            self.status_var.set("运行中...")
        else:
            self.start_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
            self.max_entry.config(state=tk.NORMAL)
            if refresh_count > 0:
                self.status_var.set(f"已停止，共刷新 {refresh_count} 次")
            else:
                self.status_var.set("就绪")

    # ---------- 控制方法 ----------
    def start_auto(self):
        global is_running, stop_flag, refresh_count
        if is_running:
            return

        if any(img is None for img in TEMPLATES.values()):
            messagebox.showerror("错误", "模板图片缺失，无法启动")
            return

        try:
            max_ref = self.max_refresh_var.get()
            if max_ref <= 0:
                messagebox.showwarning("警告", "最大刷新次数必须大于0")
                return
        except:
            messagebox.showerror("错误", "请输入有效的数字")
            return

        stop_flag = False
        is_running = True
        # UI状态会在定时刷新中自动更新，无需手动调用

        try:
            # 启动自动化线程，不传递任何回调
            self.thread = threading.Thread(target=automation_loop, args=(max_ref,), daemon=True)
            self.thread.start()
        except Exception as e:
            import traceback
            traceback.print_exc()
            messagebox.showerror("启动失败", f"错误：{e}")
            is_running = False
            self.update_ui()  # 立即刷新UI以恢复按钮状态

    def stop_auto(self):
        global stop_flag, is_running
        if is_running:
            stop_flag = True
            self.status_var.set("正在停止...")
            if self.thread and self.thread.is_alive():
                self.thread.join(timeout=2.0)
            is_running = False
            # UI状态会在定时刷新中更新

# ---------- 主程序入口 ----------
if __name__ == "__main__":
    root = tk.Tk()
    app = AutoShopApp(root)
    root.mainloop()