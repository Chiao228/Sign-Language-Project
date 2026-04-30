#!/usr/bin/env python3
"""
dynamic_energy_crop.py
======================
動態能量裁切與濃縮 (66D 版本 - 精簡手部特徵)
描述：此腳本將剔除開頭與結尾的靜態(發呆)幀，並將剩餘動態片段調整縮放為標準的 (30, 66) 時間軸尺寸。
"""

import os
import cv2
import numpy as np
from pathlib import Path

# ==========================================
# 1. Hyperparameters & Paths
# ==========================================
SOURCE_DIR = Path("tsl_features_66")   # features.py 產出的 66 維資料
OUTPUT_DIR = Path("processed_npy_66")  # 裁切兼縮放完成的輸出地

TARGET_FRAMES = 30
# 經過雙重歸一化後，坐標已經微調，這裡設定能量差值門檻。若發現裁掉太多，可降為 0.02
ENERGY_RATIO_THRESHOLD = 0.05 

def process_single_file(file_path, output_path):
    try:
        features = np.load(file_path).astype(np.float32)
        N, D = features.shape
        # 66 維度防護
        if D != 66 or N < 2:
            return False, f"無效格式(D:{D}, N:{N})"

        # 1. 重心偵測：計算兩手重心 (手腕到指尖 共11點)
        lh_pts = features[:, 0:33].reshape(-1, 11, 3)
        rh_pts = features[:, 33:66].reshape(-1, 11, 3)
        lh_center = np.mean(lh_pts, axis=1) 
        rh_center = np.mean(rh_pts, axis=1)
        centers = np.hstack([lh_center, rh_center]) # (N, 6)
        
        # 2. 滾動標準差能量偵測
        energy_list = []
        win_size = 5
        for i in range(len(centers)):
            start_w = max(0, i - win_size // 2)
            end_w = min(len(centers), i + win_size // 2 + 1)
            window = centers[start_w:end_w]
            energy_list.append(np.mean(np.std(window, axis=0)))
        energy = np.array(energy_list)
        
        # 3. 強化平滑處理 (視窗 11，讓波峰更平穩)
        kernel_size = min(11, len(energy))
        kernel = np.ones(kernel_size) / kernel_size
        smoothed_energy = np.convolve(energy, kernel, mode='same')
        
        # 4. 強制峰值偵測：使用 40% 的相對門檻，且地板值降到 0.01
        # 這樣即使影片動作極微弱，也能強迫找出相對最動的部分
        max_e = np.max(smoothed_energy)
        dyn_thresh = max(0.01, max_e * 0.40) 
        valid_indices = np.where(smoothed_energy > dyn_thresh)[0]
        
        if len(valid_indices) < 3: # 只要有幾格在動就開切
            start_idx, end_idx = 0, N
        else:
            # 最小緩衝區
            start_idx = max(0, valid_indices[0] - 2)
            end_idx = min(N, valid_indices[-1] + 3)
            
            # 【保護機制】維持 15 格保底
            current_dur = end_idx - start_idx
            if current_dur < 15 and N > 15:
                needed = 15 - current_dur
                start_idx = max(0, start_idx - needed // 2)
                end_idx = min(N, end_idx + (needed - needed // 2))

        duration = end_idx - start_idx
        # 換算秒數 (以 30 FPS 為主)
        t_start = start_idx / 30.0
        t_end = end_idx / 30.0
        crop_msg = f"原:{N:3d}格 -> 取[{start_idx:3d}:{end_idx:3d}] ({t_start:4.2f}s~{t_end:4.2f}s) | 留:{duration:3d}格 | MaxE:{max_e:.4f}"

        cropped = features[start_idx:end_idx]
        
        # 利用 OpenCV 進行時間差值拉伸或壓縮到 (30幀, 66維)
        resized = cv2.resize(cropped, (66, TARGET_FRAMES), interpolation=cv2.INTER_LINEAR)
        
        np.save(output_path, resized)
        return True, crop_msg
        
    except Exception as e:
        return False, str(e)

def main():
    print("=" * 60)
    print("🚀 啟動特徵動態裁切與核心濃縮管線")
    print("=" * 60)
    
    if not SOURCE_DIR.exists():
        print(f"❌ 找不到來源資料夾 (請確認 features.py 是否有跑完生出檔案): {SOURCE_DIR}")
        return
        
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    all_files = list(SOURCE_DIR.rglob("*.npy"))
    total_files = len(all_files)
    
    if total_files == 0:
        print("❌ 來源資料夾內目前沒有任何 .npy 檔案！")
        return
        
    print(f"📂 偵測到 {total_files} 個 .npy 檔案準備處理...")
    print(f"⚙️  目標純化維度: (30, 66)\n")
    
    success_count = 0
    fail_count = 0
    
    for idx, file_path in enumerate(all_files, 1):
        relative_path = file_path.relative_to(SOURCE_DIR)
        output_path = OUTPUT_DIR / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        is_success, msg = process_single_file(file_path, output_path)
        
        if is_success:
            success_count += 1
            # 每一筆都顯示詳細裁切資訊
            print(f"✅ [{idx:04d}/{total_files:04d}] {relative_path.parent.name}/{relative_path.name:<20} | {msg}")
        else:
            fail_count += 1
            print(f"⚠️  [{idx:04d}/{total_files:04d}] 略過壞檔: {relative_path} | 錯誤: {msg}")
            
    print("\n" + "=" * 60)
    print(f"🎉 任務完成！")
    print(f"✅ 成功純化: {success_count} 筆")
    if fail_count > 0:
        print(f"❌ 跳過極短壞檔: {fail_count} 筆")
    print(f"📁 最終標準資料庫: {OUTPUT_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()
