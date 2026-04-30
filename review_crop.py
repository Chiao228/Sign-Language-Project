#!/usr/bin/env python3
import os
import numpy as np
import subprocess
from pathlib import Path

# ==========================================
# 1. 自動搜尋播放器
# ==========================================
COMMON_PLAYER_PATHS = [
    (r"C:\Program Files\VideoLAN\VLC\vlc.exe", "vlc"),
    (r"C:\Program Files (x86)\VideoLAN\VLC\vlc.exe", "vlc"),
    (r"C:\Program Files\DAUM\PotPlayer\PotPlayerMini64.exe", "pot"),
    (r"C:\Program Files\DAUM\PotPlayer\PotPlayerMini.exe", "pot"),
    (r"C:\Program Files (x86)\DAUM\PotPlayer\PotPlayerMini64.exe", "pot")
]

def find_player():
    for path, p_type in COMMON_PLAYER_PATHS:
        if os.path.exists(path):
            return path, p_type
    return None, None

PLAYER_PATH, PLAYER_TYPE = find_player()
SOURCE_DIR = Path("tsl_features_66")
VIDEO_DIR = Path("tsl_tracking_66")

def get_crop_info(file_path):
    """ 計算與 dynamic_energy_crop_138.py 完全一致的裁切起點 """
    features = np.load(file_path).astype(np.float32)
    N = len(features)
    
    # 1. 重心偵測
    lh_pts = features[:, 0:33].reshape(-1, 11, 3)
    rh_pts = features[:, 33:66].reshape(-1, 11, 3)
    lh_center = np.mean(lh_pts, axis=1) 
    rh_center = np.mean(rh_pts, axis=1)
    centers = np.hstack([lh_center, rh_center]) 
    
    # 2. 滾動標準差
    energy_list = []
    win_size = 5
    for i in range(len(centers)):
        start_w = max(0, i - win_size // 2)
        end_w = min(len(centers), i + win_size // 2 + 1)
        window = centers[start_w:end_w]
        energy_list.append(np.mean(np.std(window, axis=0)))
    energy = np.array(energy_list)
    
    # 3. 平滑處理 (11幀)
    kernel_size = min(11, len(energy))
    kernel = np.ones(kernel_size) / kernel_size
    smoothed = np.convolve(energy, kernel, mode='same')
    
    # 4. 門檻 40% (目前最新版)
    max_e = np.max(smoothed)
    dyn_thresh = max(0.01, max_e * 0.40) 
    valid_indices = np.where(smoothed > dyn_thresh)[0]
    
    if len(valid_indices) < 3:
        return 0, N, 0.0, max_e
    
    start_idx = max(0, valid_indices[0] - 2)
    end_idx = min(N, valid_indices[-1] + 3)
    return start_idx, end_idx, start_idx / 30.0, max_e

def process_and_play(target_npy):
    start_f, end_f, start_sec, max_e = get_crop_info(target_npy)
    print(f"\n📊 分析影片: {target_npy.name}")
    print(f"   範圍: 第 {start_f} ~ {end_f} 幀 ({start_sec:.2f} 秒) | MaxE: {max_e:.4f}")

    # 尋找原始影片
    rel_path = target_npy.relative_to(SOURCE_DIR)
    possible_extensions = [".mp4", ".mkv", ".avi", ".mov"]
    video_path = None
    for ext in possible_extensions:
        test_path = VIDEO_DIR / rel_path.with_suffix(ext)
        if test_path.exists():
            video_path = test_path
            break
    
    if video_path and PLAYER_PATH:
        try:
            if PLAYER_TYPE == "vlc":
                subprocess.Popen([PLAYER_PATH, str(video_path), f"--start-time={start_sec}"])
            elif PLAYER_TYPE == "pot":
                import datetime
                time_str = str(datetime.timedelta(seconds=start_sec))
                subprocess.Popen([PLAYER_PATH, str(video_path), f"/seek={time_str}"])
            else:
                os.startfile(video_path)
            return True
        except Exception as e:
            print(f"❌ 無法開啟播放器: {e}")
    elif video_path:
        os.startfile(video_path)
        return True
    else:
        print(f"❌ 找不到原始影片檔案: {rel_path.with_suffix('.mp4')}")
        return False

def main():
    print("="*50)
    print("🔍 手語裁切自動檢查工具 (連續審查版)")
    print("="*50)
    
    if PLAYER_PATH:
        print(f"✅ 已偵測到播放器: {os.path.basename(PLAYER_PATH)}")
    else:
        print("⚠️  提醒: 找不到 VLC/PotPlayer，將使用系統預設模式。")
    
    keyword = input("\n請輸入 類別名稱 (如 '謝謝') 或 關鍵字: ").strip()
    if not keyword: return

    # 搜尋匹配的特徵檔
    all_npy = sorted(list(SOURCE_DIR.rglob("*.npy")))
    matches = [p for p in all_npy if keyword.lower() in str(p).lower()]
    
    if not matches:
        print(f"❌ 找不到包含 '{keyword}' 的檔案。")
        return

    print(f"🔎 找到 {len(matches)} 個檔案。開始遍歷...")
    
    for i, target_npy in enumerate(matches, 1):
        print(f"\n[{i}/{len(matches)}]", "-"*40)
        success = process_and_play(target_npy)
        
        if i < len(matches):
            cmd = input("\n按 [Enter] 下一部 | 輸入 [q] 退出: ").strip().lower()
            if cmd == 'q':
                break
    
    print("\n✅ 審查結束。")

if __name__ == "__main__":
    main()
