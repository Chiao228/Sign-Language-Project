import numpy as np
import os
import random
import cv2
from scipy.interpolate import interp1d

# --- 1. 配置路徑與參數 ---
INPUT_DIR = "tsl_features_66"           
OUTPUT_DIR = "tsl_augmented_features_66"    
REPORT_DIR = "augmentation_reports_66"      
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 由於我們只留了 11 個點，所以就不再對所有的線段做檢查繪製了。此 CONN 可廢棄或簡化。
CONN = []

# --- 2. 核心擴增技術 ---

def add_noise_safe(data, sigma=0.008):
    noise = np.random.normal(0, sigma, data.shape)
    mask = (np.linalg.norm(data, axis=1, keepdims=True) > 1e-6).astype(float)
    return data + (noise * mask), sigma

# --- 隨機局部旋轉 ---
def rotate_landmarks_safe(data, max_angle=30):
    """
    執行局部旋轉：以每隻手的掌心為中心旋轉，避免全域旋轉造成的劇烈位移。
    """
    if np.linalg.norm(data) < 1e-6: return data, 0.0
    
    angle_deg = random.uniform(-max_angle, max_angle)
    angle_rad = np.radians(angle_deg)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    
    new_data = data.copy()
    
    # 針對左手 (0-32) 與 右手 (33-65) 分別處理
    for start_idx in [0, 33]:
        # 檢查該手是否存在
        hand_points = data[:, start_idx : start_idx + 33]
        if np.linalg.norm(hand_points) < 1e-6: continue
        
        # 1. 找到旋轉中心（通常是手腕，即該區塊的前 3 碼 x,y,z）
        cx, cy = data[:, start_idx], data[:, start_idx + 1]
        
        # 2. 針對該手的 11 個點進行旋轉
        for i in range(11):
            x_idx = start_idx + i * 3
            y_idx = start_idx + i * 3 + 1
            
            # 減去中心 (移到原點)
            rel_x = data[:, x_idx] - cx
            rel_y = data[:, y_idx] - cy
            
            # 旋轉
            new_data[:, x_idx] = cx + (rel_x * cos_a - rel_y * sin_a)
            new_data[:, y_idx] = cy + (rel_x * sin_a + rel_y * cos_a)
            
    return new_data, angle_deg

# --- 隨機空間平移 ---
def random_shift_safe(data, max_shift=0.05):
    if np.linalg.norm(data) < 1e-6: return data, (0,0)
    sx, sy = random.uniform(-max_shift, max_shift), random.uniform(-max_shift, max_shift)
    new_data = data.copy()
    num_landmarks = data.shape[1] // 3
    for i in range(num_landmarks):
        x_idx, y_idx = i*3, i*3+1
        mask = (np.abs(data[:, x_idx]) + np.abs(data[:, y_idx]) > 1e-6)
        new_data[mask, x_idx] += sx
        new_data[mask, y_idx] += sy
    return new_data, (sx, sy)

# --- 局部縮放手指 ---
def scale_hand_parts(data, scale_range=(0.9, 1.1)):
    factor = random.uniform(*scale_range)
    new_data = data.copy()
    for start_idx in [0, 33]: # 雙手起始索引
        if np.linalg.norm(data[:, start_idx:start_idx+33]) < 1e-6: continue
        cx, cy = data[:, start_idx], data[:, start_idx+1] # 以手部原點為基準
        for i in range(11):
            xi, yi = start_idx + i*3, start_idx + i*3+1
            new_data[:, xi] = cx + (data[:, xi] - cx) * factor
            new_data[:, yi] = cy + (data[:, yi] - cy) * factor
    return new_data, factor

# --- 關鍵點丟失 (模擬遮擋) ---
def finger_dropout(data, prob=0.1):
    new_data = data.copy()
    # 新版 11 點/手 配置下：指尖分別是 2, 4, 6, 8, 10
    fingertips = [2, 4, 6, 8, 10]
    # 加上右手偏移
    fingertips += [pt + 11 for pt in fingertips]
    for pt in fingertips:
        if random.random() < prob: new_data[:, pt*3 : pt*3+3] = 0
    return new_data, prob

def time_warp(data, factor_range=(0.8, 1.2)):
    if np.linalg.norm(data) < 1e-6 or data.shape[0] < 3: return data, 1.0
    factor = random.uniform(*factor_range)
    target_frames = int(data.shape[0] * factor)
    x = np.linspace(0, 1, data.shape[0])
    new_x = np.linspace(0, 1, target_frames)
    f = interp1d(x, data, axis=0, kind='linear', fill_value="extrapolate")
    return f(new_x), factor

# --- 水平翻轉 (模擬左撇子) ---
def horizontal_flip_safe(data):
    """
    支援 66 維特徵分佈 (x, y, z)：
    - 0-32: 左手 (11點 * 3)
    - 33-65: 右手 (11點 * 3)
    """
    new_data = data.copy()
    
    # 1. 將所有 X 座標取反 (假設座標已正規化在 0~1 之間，則用 1-x)
    new_data[:, 0::3] = 1.0 - new_data[:, 0::3]
    
    # 2. 交換左右手的特徵區塊
    left_hand = new_data[:, 0:33].copy()
    right_hand = new_data[:, 33:66].copy()
    
    new_data[:, 0:33] = right_hand
    new_data[:, 33:66] = left_hand
    
    return new_data

# --- 3. 視覺化報告工具 (略，保持原樣) ---
# ... (save_comparison_screenshot 內容同前)

# --- 4. 主執行程序 ---
def run_full_augmentation():
    print("🚀 開始執行進階資料擴增作業...")
    all_words = [d for d in os.listdir(INPUT_DIR) if os.path.isdir(os.path.join(INPUT_DIR, d))]
    total_count = 0
    for word in sorted(all_words):
        word_path, out_path = os.path.join(INPUT_DIR, word), os.path.join(OUTPUT_DIR, word)
        os.makedirs(out_path, exist_ok=True)
        report_done = 0
        for npy_file in os.listdir(word_path):
            if not npy_file.endswith('.npy'): continue
            data = np.load(os.path.join(word_path, npy_file))
            np.save(os.path.join(out_path, f"orig_{npy_file}"), data)
            for i in range(10):
                # 依序執行所有擴增技術
                aug, r = rotate_landmarks_safe(data)
                aug, s = random_shift_safe(aug)
                aug, sc = scale_hand_parts(aug)
                aug, d = finger_dropout(aug)
                aug, n = add_noise_safe(aug)
                aug, t = time_warp(aug)
                np.save(os.path.join(out_path, f"aug_{i}_{npy_file}"), aug)
            total_count += 1
    print(f"✅ 擴增完成！")

if __name__ == "__main__":
    run_full_augmentation()