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
import pygetwindow as gw
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

import pygetwindow as gw

def get_game_window():
    """
    获取当前活动窗口作为游戏窗口。
    返回 (left, top, width, height) 元组，若获取失败则返回全屏区域。
    """
    try:
        active_win = gw.getActiveWindow()
        if active_win is not None:
            left, top, w, h = active_win.left, active_win.top, active_win.width, active_win.height
            logging.info(f"活动窗口区域: left={left}, top={top}, width={w}, height={h}")
            return (left, top, w, h)
        else:
            logging.warning("未获取到活动窗口，将使用全屏")
    except Exception as e:
        logging.error(f"获取活动窗口失败: {e}")

    screen_w, screen_h = pyautogui.size()
    return (0, 0, screen_w, screen_h)


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

def check_text_in_region(item_pos, text_key, region=None, confidence=0.8):
    """
    在道具左上角向右 30% 的窗口宽度、高度与道具相同的矩形区域内验证文字
    item_pos: (x,y,w,h) 相对于 region 的坐标
    region: (left, top, width, height) 或 None（全屏）
    """
    x, y, w, h = item_pos

    # 获取窗口宽度
    if region is not None:
        win_width = region[2]
    else:
        win_width = pyautogui.size().width

    # 文字区域宽度为窗口宽度的 30%
    text_width = int(win_width * 0.30)
    region_rect = (x, y, text_width, h)

    screen = capture_screen(region)  # 截取 region 区域（若 None 则全屏）
    template = TEMPLATES.get(text_key)
    if template is None:
        return False

    x0, y0, w0, h0 = region_rect
    img_h, img_w = screen.shape
    # 确保不超出截图区域
    x0 = max(0, min(x0, img_w - 1))
    y0 = max(0, min(y0, img_h - 1))
    w0 = min(w0, img_w - x0)
    h0 = min(h0, img_h - y0)
    if w0 <= 0 or h0 <= 0:
        return False

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
    """
    截取屏幕指定区域。region 格式为 (left, top, width, height)，
    内部自动转换为 bbox (left, top, left+width, top+height)。
    """
    if region is not None:
        left, top, width, height = region
        # 确保宽高为正
        if width <= 0 or height <= 0:
            logging.error(f"无效的截图区域: {region}")
            region = None
        else:
            bbox = (left, top, left + width, top + height)
    else:
        bbox = None

    screenshot = ImageGrab.grab(bbox=bbox)
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

    # ---- 1. 文字验证（可选，可保留或注释） ----
    # text_key = "bookmark_text" if item_type == "bookmark" else "medal_text"
    # if not check_text_in_region(item_pos, text_key):
    #     logging.warning(f"道具 ({x},{y}) 文字验证失败，跳过")
    #     return False

    # ---- 2. 计算搜索区域（相对于 region 或全屏） ----
    if region is not None:
        reg_left, reg_top, reg_width, reg_height = region
    else:
        reg_left, reg_top, reg_width, reg_height = 0, 0, pyautogui.size().width, pyautogui.size().height

    start_ratio = 0.28
    end_ratio = 0.46
    search_x_offset = int(reg_width * start_ratio)
    search_w = int(reg_width * (end_ratio - start_ratio))
    search_x = x + search_x_offset
    search_y = y
    search_h = h

    if region is not None:
        max_x = reg_left + reg_width
        max_y = reg_top + reg_height
        search_x = max(reg_left, min(search_x, max_x - 1))
        search_y = max(reg_top, min(search_y, max_y - 1))
        search_w = min(search_w, max_x - search_x)
        search_h = min(search_h, max_y - search_y)
    else:
        screen_w, screen_h = pyautogui.size()
        search_x = max(0, min(search_x, screen_w - 1))
        search_y = max(0, min(search_y, screen_h - 1))
        search_w = min(search_w, screen_w - search_x)
        search_h = min(search_h, screen_h - search_y)

    if search_w <= 0 or search_h <= 0:
        logging.warning(f"道具 ({x},{y}) 搜索区域无效，跳过")
        return False

    screen_roi = capture_screen(region)
    roi = screen_roi[search_y:search_y+search_h, search_x:search_x+search_w]
    if roi.shape[0] == 0 or roi.shape[1] == 0:
        logging.warning(f"道具 ({x},{y}) ROI 为空")
        return False

    available_template = TEMPLATES.get("available")
    unavailable_template = TEMPLATES.get("unavailable")
    if available_template is None or unavailable_template is None:
        logging.warning("状态模板未加载")
        return False

    res_avail = cv2.matchTemplate(roi, available_template, cv2.TM_CCOEFF_NORMED)
    _, max_avail, _, max_loc_avail = cv2.minMaxLoc(res_avail)
    res_unavail = cv2.matchTemplate(roi, unavailable_template, cv2.TM_CCOEFF_NORMED)
    _, max_unavail, _, _ = cv2.minMaxLoc(res_unavail)

    logging.debug(f"1/1匹配值: {max_avail:.3f}, 0/1匹配值: {max_unavail:.3f}")

    confidence_status = 0.75

    if max_avail >= confidence_status and max_avail > max_unavail:
        h_avail, w_avail = available_template.shape
        click_x_rel = search_x + max_loc_avail[0] + w_avail // 2
        click_y_rel = search_y + max_loc_avail[1] + h_avail // 2

        if region is not None:
            click_abs_x = reg_left + click_x_rel
            click_abs_y = reg_top + click_y_rel
        else:
            click_abs_x = click_x_rel
            click_abs_y = click_y_rel

        logging.info(f"点击 '1/1' 按钮: ({click_abs_x}, {click_abs_y})")

        time.sleep(3)
        pyautogui.moveTo(click_abs_x + random.randint(-3, 3),
                         click_abs_y + random.randint(-3, 3),
                         duration=random.uniform(0.05, 0.15))
        pyautogui.click()
        random_delay("click")

        # ---------- 超时处理：停止程序 ----------
        if not wait_and_click_template("confirm_buy", timeout=3.5, region=region):
            logging.error("购买确认弹窗超时（3.5秒），停止程序")
            global stop_flag
            stop_flag = True
            return False
        return True

    elif max_unavail >= confidence_status and max_unavail > max_avail:
        logging.info(f"道具 ({x},{y}) 状态为 0/1，已购买，跳过")
        return False
    else:
        logging.warning(f"道具 ({x},{y}) 未检测到明确状态，跳过")
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
                center_x = max_loc[0] + w//2
                center_y = max_loc[1] + h//2
                if region is not None:
                    left, top, _, _ = region
                    center_x += left
                    center_y += top
                pyautogui.moveTo(center_x + random.randint(-3, 3), center_y + random.randint(-3, 3), duration=random.uniform(0.05, 0.15))
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
        center_x = max_loc[0] + w//2
        center_y = max_loc[1] + h//2
        if region is not None:
            left, top, _, _ = region
            center_x += left
            center_y += top
        pyautogui.moveTo(center_x + random.randint(-3, 3), center_y + random.randint(-3, 3), duration=random.uniform(0.05, 0.15))
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
    time.sleep(1)

    # --- 获取窗口区域 ---
    try:
        active_win = gw.getActiveWindow()
        if active_win is not None:
            left, top, w, h = active_win.left, active_win.top, active_win.width, active_win.height
            logging.info(f"原始窗口区域: left={left}, top={top}, width={w}, height={h}")
            offset_x = 8
            offset_y = 30
            adjusted_left = left + offset_x
            adjusted_top = top + offset_y
            adjusted_width = w - 2 * offset_x
            adjusted_height = h - offset_y - offset_x
            window_region = (adjusted_left, adjusted_top, adjusted_width, adjusted_height)
            logging.info(f"调整后窗口区域: left={adjusted_left}, top={adjusted_top}, width={adjusted_width}, height={adjusted_height}")
        else:
            logging.warning("未获取到活动窗口，使用全屏")
            screen_w, screen_h = pyautogui.size()
            window_region = (0, 0, screen_w, screen_h)
    except Exception as e:
        logging.error(f"获取窗口失败: {e}")
        screen_w, screen_h = pyautogui.size()
        window_region = (0, 0, screen_w, screen_h)

    left, top, win_w, win_h = window_region
    logging.info(f"游戏窗口区域: left={left}, top={top}, width={win_w}, height={win_h}")

    time.sleep(1)
    click_x = left + int(win_w * 0.75)
    click_y = top + int(win_h * 0.5)
    pyautogui.click(click_x, click_y)
    time.sleep(0.5)

    region = window_region
    scroll_pixels = int(win_h * 0.25)

    # ========== 程序启动初始扫描（相当于刚刷新后的完整扫描） ==========
    logging.info("程序启动初始扫描（执行完整顶部+底部扫描）...")
    logging.info("扫描当前可见区域（顶部）...")
    scan_with_retry(scan_and_purchase, region, max_attempts=5, no_purchase_required=2, delay=0.3)

    logging.info("连续下滑两次...")
    for _ in range(2):
        pyautogui.scroll(-scroll_pixels)
        time.sleep(random.uniform(0.3, 0.4))

    logging.info("扫描当前可见区域（底部）...")
    scan_with_retry(scan_and_purchase, region, max_attempts=5, no_purchase_required=2, delay=0.3)
    logging.info("初始扫描完成，进入主循环。")

    # ---------- 主循环 ----------
    while is_running and not stop_flag and refresh_count < max_refresh:
        try:
            # ========== 刷新前检测（只扫描当前可见区域，此时页面在底部） ==========
            logging.info("扫描当前可见区域（刷新前检测）...")
            cleared = scan_with_retry(scan_and_purchase, region, max_attempts=5, no_purchase_required=2, delay=0.3)
            if cleared:
                logging.info("当前区域已清空（无更多可购买道具）")
            else:
                logging.warning("扫描达到尝试上限，但视为清空，继续执行")

            # ========== 执行刷新 ==========
            logging.info("未检测到可购买道具（或已全部购买），执行刷新...")
            refresh_x = left + int(win_w * 0.16)
            refresh_y = top + win_h - int(win_h * 0.07)
            refresh_x = max(left, min(left + win_w, refresh_x))
            refresh_y = max(top, min(top + win_h, refresh_y))
            logging.info(f"点击刷新按钮: ({refresh_x}, {refresh_y})")
            pyautogui.click(refresh_x, refresh_y)
            time.sleep(0.5)

            # ---------- 超时处理：停止程序 ----------
            if not wait_and_click_template("confirm_refresh", timeout=12.0, region=region):
                logging.error("刷新确认弹窗超时（12秒），停止程序")
                stop_flag = True
                break

            refresh_count += 1
            logging.info(f"刷新成功 (第{refresh_count}次)")

            # 刷新后等待加载
            time.sleep(random.uniform(2.5, 3.0) + 1.0)
            # ---------- 超时处理：停止程序 ----------
            if not wait_for_loading(region):
                logging.error("刷新后加载超时（15秒），停止程序")
                stop_flag = True
                break

            # 激活窗口
            pyautogui.click(left + int(win_w * 0.75), top + int(win_h * 0.5))
            time.sleep(0.5)

            # ========== 刷新后扫描（顶部 + 底部） ==========
            logging.info("扫描当前可见区域（顶部）...")
            scan_with_retry(scan_and_purchase, region, max_attempts=5, no_purchase_required=2, delay=0.3)

            logging.info("连续下滑两次...")
            for _ in range(2):
                pyautogui.scroll(-scroll_pixels)
                time.sleep(random.uniform(0.3, 0.4))

            logging.info("扫描当前可见区域（底部）...")
            scan_with_retry(scan_and_purchase, region, max_attempts=5, no_purchase_required=2, delay=0.3)

            random_delay("between_loop")

        except Exception as e:
            logging.error(f"自动化循环异常: {e}")
            time.sleep(2)

    if refresh_count >= max_refresh:
        logging.info("达到最大刷新次数，自动停止")
    elif stop_flag:
        logging.info("用户手动停止或发生错误")
    is_running = False


def scan_with_retry(scan_func, region, max_attempts=5, no_purchase_required=2, delay=0.5):
    """
    执行扫描购买，直到连续 no_purchase_required 次没有购买到任何道具，
    或达到最大尝试次数 max_attempts。
    返回 True 表示区域已清空（连续无道具），False 表示达到尝试上限但仍有道具（保守）。
    """
    no_purchase_count = 0
    attempts = 0
    while attempts < max_attempts and no_purchase_count < no_purchase_required:
        purchased = scan_func(region)
        attempts += 1
        if purchased:
            no_purchase_count = 0
            logging.info(f"第 {attempts} 次扫描购买到道具，继续扫描...")
            time.sleep(delay)  # 购买后等待，让界面更新
        else:
            no_purchase_count += 1
            if no_purchase_count < no_purchase_required:
                logging.info(f"第 {attempts} 次扫描未发现道具，{delay}秒后重扫...")
                time.sleep(delay)
    if no_purchase_count >= no_purchase_required:
        logging.info(f"连续 {no_purchase_required} 次扫描无道具，区域已清空")
        return True
    else:
        logging.info(f"达到最大尝试次数 {max_attempts}，停止扫描")
        return False

def is_template_present(template_name, region=None, confidence=CONFIDENCE):
    """检测指定模板是否在当前截图区域内出现（不执行点击）"""
    screen = capture_screen(region)
    template = TEMPLATES.get(template_name)
    if template is None:
        return False
    result = cv2.matchTemplate(screen, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(result)
    return max_val >= confidence

def wait_for_loading(region, timeout=15, check_interval=0.5):
    """
    等待商店加载完成（通过检测刷新按钮是否可识别）
    timeout: 最大等待时间（秒）
    check_interval: 检测间隔
    返回 True 表示加载完成，False 表示超时
    """
    start = time.time()
    while time.time() - start < timeout:
        if is_template_present("refresh_btn", region):
            logging.info("加载完成：检测到刷新按钮")
            return True
        time.sleep(check_interval)
    logging.warning("加载超时，继续执行")
    return False

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
            bk_prob = (bookmark_groups / refresh_count) * 100
            med_prob = (medal_groups / refresh_count) * 100
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

        # 提示用户将游戏窗口置于最前
        messagebox.showinfo("提示", "点击“确定”后点一下游戏窗口。")

        stop_flag = False
        is_running = True

        try:
            self.thread = threading.Thread(target=automation_loop, args=(max_ref,), daemon=True)
            self.thread.start()
        except Exception as e:
            import traceback
            traceback.print_exc()
            messagebox.showerror("启动失败", f"错误：{e}")
            is_running = False
            self.update_ui()

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
