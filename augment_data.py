import numpy as np
import random

# --- 1. 雜訊擴增 ---
def add_noise_safe(data, sigma=0.008):
    noise = np.random.normal(0, sigma, data.shape)
    mask = (np.linalg.norm(data, axis=1, keepdims=True) > 1e-6).astype(float)
    return data + (noise * mask), sigma

# --- 2. 隨機局部旋轉 ---
def rotate_landmarks_safe(data, max_angle=30):
    if np.linalg.norm(data) < 1e-6: return data, 0.0
    angle_deg = random.uniform(-max_angle, max_angle)
    angle_rad = np.radians(angle_deg)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    new_data = data.copy()
    for start_idx in [0, 33]:
        hand_points = data[:, start_idx : start_idx + 33]
        if np.linalg.norm(hand_points) < 1e-6: continue
        cx, cy = data[:, start_idx], data[:, start_idx + 1]
        for i in range(11):
            x_idx = start_idx + i * 3
            y_idx = start_idx + i * 3 + 1
            rel_x = data[:, x_idx] - cx
            rel_y = data[:, y_idx] - cy
            new_data[:, x_idx] = cx + (rel_x * cos_a - rel_y * sin_a)
            new_data[:, y_idx] = cy + (rel_x * sin_a + rel_y * cos_a)
    return new_data, angle_deg

# --- 3. 隨機空間平移 ---
def random_shift_safe(data, max_shift=0.05):
    if np.linalg.norm(data) < 1e-6: return data, (0,0)
    sx, sy = random.uniform(-max_shift, max_shift), random.uniform(-max_shift, max_shift)
    new_data = data.copy()
    num_landmarks = data.shape[1] // 3
    # 僅針對座標點 (x,y,z) 的 x, y 進行平移
    for i in range(num_landmarks):
        x_idx, y_idx = i*3, i*3+1
        mask = (np.abs(data[:, x_idx]) + np.abs(data[:, y_idx]) > 1e-6)
        new_data[mask, x_idx] += sx
        new_data[mask, y_idx] += sy
    return new_data, (sx, sy)

# --- 4. 局部縮放手指 ---
def scale_hand_parts(data, scale_range=(0.9, 1.1)):
    factor = random.uniform(*scale_range)
    new_data = data.copy()
    for start_idx in [0, 33]: 
        if np.linalg.norm(data[:, start_idx:start_idx+33]) < 1e-6: continue
        cx, cy = data[:, start_idx], data[:, start_idx+1]
        for i in range(11):
            xi, yi = start_idx + i*3, start_idx + i*3+1
            new_data[:, xi] = cx + (data[:, xi] - cx) * factor
            new_data[:, yi] = cy + (data[:, yi] - cy) * factor
    return new_data, factor

# --- 5. 關鍵點丟失 (模擬遮擋) ---
def finger_dropout(data, prob=0.1):
    new_data = data.copy()
    fingertips = [2, 4, 6, 8, 10]
    fingertips += [pt + 11 for pt in fingertips]
    for pt in fingertips:
        if random.random() < prob: new_data[:, pt*3 : pt*3+3] = 0
    return new_data, prob

# --- 6. 水平翻轉 (模擬左撇子 - 自動維度調整版) ---
def horizontal_flip_safe(data):
    """
    自動偵測維度並執行「安全翻轉」：
    - 座標點：執行 1.0 - x
    - 幾何特徵：不執行 X 翻轉（因為是距離/角度）
    - 結構對調：交換左/右特徵區塊
    """
    new_data = data.copy()
    dim = data.shape[1]
    
    # --- 步驟 A: X 座標鏡像 (僅針對座標類型的索引) ---
    if dim in [66, 81]:
        # 全員皆為 (x, y, z) 座標
        new_data[:, 0::3] = 1.0 - new_data[:, 0::3]
    elif dim == 72:
        # 手部座標 (0-32) 與 (36-68)
        new_data[:, 0:33:3] = 1.0 - new_data[:, 0:33:3]
        new_data[:, 36:69:3] = 1.0 - new_data[:, 36:69:3]
    elif dim == 74:
        # 手部座標 (0-32) 與 (37-69)
        new_data[:, 0:33:3] = 1.0 - new_data[:, 0:33:3]
        new_data[:, 37:70:3] = 1.0 - new_data[:, 37:70:3]

    # --- 步驟 B: 左右特徵塊交換 ---
    if dim == 66:
        # [左手33] [右手33]
        l, r = new_data[:, 0:33].copy(), new_data[:, 33:66].copy()
        new_data[:, 0:33], new_data[:, 33:66] = r, l
    elif dim == 72:
        # [左手33][左幾何3] [右手33][右幾何3]
        l_all, r_all = new_data[:, 0:36].copy(), new_data[:, 36:72].copy()
        new_data[:, 0:36], new_data[:, 36:72] = r_all, l_all
    elif dim == 74:
        # [左手33][左幾何4] [右手33][右幾幾4]
        l_all, r_all = new_data[:, 0:37].copy(), new_data[:, 37:74].copy()
        new_data[:, 0:37], new_data[:, 37:74] = r_all, l_all
    elif dim == 81:
        # [左手33] [右手33] [鼻3] [左肩3] [右肩3] [左肘3] [右肘3]
        l_h, r_h = new_data[:, 0:33].copy(), new_data[:, 33:66].copy()
        new_data[:, 0:33], new_data[:, 33:66] = r_h, l_h
        # 肩膀交換 (69-71 vs 72-74)
        l_s, r_s = new_data[:, 69:72].copy(), new_data[:, 72:75].copy()
        new_data[:, 69:72], new_data[:, 72:75] = r_s, l_s
        # 手肘交換 (75-77 vs 78-80)
        l_e, r_e = new_data[:, 75:78].copy(), new_data[:, 78:81].copy()
        new_data[:, 75:78], new_data[:, 78:81] = r_e, l_e
        
    return new_data