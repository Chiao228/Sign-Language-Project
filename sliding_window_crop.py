#!/usr/bin/env python3
"""
sliding_window_crop.py
======================
滑窗裁切與過濾 (66D 版本 - 精簡手部特徵)
描述：此腳本不再尋找單一動態區塊，而是使用固定長度的滑窗 (Sliding Window) 遍歷整個序列。
     若視窗內的平均動態能量高於門檻，則將其存為一個獨立的樣本。
"""

import os
import cv2
import numpy as np
from pathlib import Path

# ==========================================
# 1. Hyperparameters & Paths
# ==========================================
SOURCE_DIR = Path("tsl_features_66")   # features.py 產出的 66 維資料
OUTPUT_DIR = Path("sliding_window_66") # 滑窗產出的輸出地

WINDOW_SIZE = 30  # 視窗長度
STRIDE = 15      # 步長 (重疊度)
ENERGY_THRESHOLD = 0.02 # 視窗平均能量門檻，過低視為靜態不存檔

def process_single_file(file_path, output_dir):
    """
    處理單個檔案，將其切割成多個滑窗。
    """
    try:
        features = np.load(file_path).astype(np.float32)
        N, D = features.shape
        # 66 維度防護
        if D != 66:
            return 0, f"無效格式(D:{D})"
        
        if N < WINDOW_SIZE:
            # 如果不夠長，可以選擇補零或直接拉伸。這裡比照原架構，若太短則拉伸一次
            resized = cv2.resize(features, (66, WINDOW_SIZE), interpolation=cv2.INTER_LINEAR)
            save_path = output_dir / f"{file_path.stem}_full.npy"
            np.save(save_path, resized)
            return 1, "長度不足 WINDOW_SIZE，已拉伸存為單一檔案"

        # 1. 重心偵測：計算兩手重心 (手腕到指尖 共11點)
        lh_pts = features[:, 0:33].reshape(-1, 11, 3)
        rh_pts = features[:, 33:66].reshape(-1, 11, 3)
        lh_center = np.mean(lh_pts, axis=1) 
        rh_center = np.mean(rh_pts, axis=1)
        centers = np.hstack([lh_center, rh_center]) # (N, 6)
        
        # 2. 計算每幀的能量 (標準差)
        energy_list = []
        win_std = 5
        for i in range(len(centers)):
            start_w = max(0, i - win_std // 2)
            end_w = min(len(centers), i + win_std // 2 + 1)
            window = centers[start_w:end_w]
            energy_list.append(np.mean(np.std(window, axis=0)))
        energy = np.array(energy_list)

        # 3. 滑窗切割
        saved_count = 0
        for start_idx in range(0, N - WINDOW_SIZE + 1, STRIDE):
            end_idx = start_idx + WINDOW_SIZE
            window_features = features[start_idx:end_idx]
            window_energy = energy[start_idx:end_idx]
            
            avg_energy = np.mean(window_energy)
            
            # 只有當能量高於門檻時才存檔
            if avg_energy >= ENERGY_THRESHOLD:
                save_path = output_dir / f"{file_path.stem}_win{start_idx}.npy"
                np.save(save_path, window_features)
                saved_count += 1
        
        # 如果整段都沒有超過門檻的視窗，但 N 足夠大，強迫取能量最高的一個
        if saved_count == 0:
            # 找能量最高的一段
            max_energy_idx = 0
            max_e = -1
            for start_idx in range(0, N - WINDOW_SIZE + 1, STRIDE):
                avg_e = np.mean(energy[start_idx:start_idx+WINDOW_SIZE])
                if avg_e > max_e:
                    max_e = avg_e
                    max_energy_idx = start_idx
            
            save_path = output_dir / f"{file_path.stem}_best.npy"
            np.save(save_path, features[max_energy_idx:max_energy_idx+WINDOW_SIZE])
            saved_count = 1
            return saved_count, f"能量皆低於門檻，強迫取出最高能量視窗 (MaxE:{max_e:.4f})"

        return saved_count, f"成功切出 {saved_count} 個視窗"
        
    except Exception as e:
        return 0, str(e)

def main():
    print("=" * 60)
    print("🚀 啟動滑窗裁切 (Sliding Window) 處理管線")
    print(f"⚙️  視窗大小: {WINDOW_SIZE}, 步長: {STRIDE}, 門檻: {ENERGY_THRESHOLD}")
    print("=" * 60)
    
    if not SOURCE_DIR.exists():
        print(f"❌ 找不到來源資料夾: {SOURCE_DIR}")
        return
        
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    all_files = list(SOURCE_DIR.rglob("*.npy"))
    total_files = len(all_files)
    
    if total_files == 0:
        print("❌ 來源資料夾內目前沒有任何 .npy 檔案！")
        return
        
    print(f"📂 偵測到 {total_files} 個來源檔案...")
    
    total_saved = 0
    fail_count = 0
    
    for idx, file_path in enumerate(all_files, 1):
        # 注意：滑窗可能會產出多個檔案，所以輸出路徑處理方式稍有不同
        relative_parent = file_path.relative_to(SOURCE_DIR).parent
        target_subdir = OUTPUT_DIR / relative_parent
        target_subdir.mkdir(parents=True, exist_ok=True)
        
        count, msg = process_single_file(file_path, target_subdir)
        
        if count > 0:
            total_saved += count
            print(f"✅ [{idx:04d}/{total_files:04d}] {file_path.name:<20} | {msg}")
        else:
            fail_count += 1
            print(f"⚠️  [{idx:04d}/{total_files:04d}] 處理失敗: {file_path.name} | 錯誤: {msg}")
            
    print("\n" + "=" * 60)
    print(f"🎉 任務完成！")
    print(f"原始檔案數: {total_files}")
    print(f"產出視窗數: {total_saved}")
    if fail_count > 0:
        print(f"失敗檔案數: {fail_count}")
    print(f"📁 輸出目錄: {OUTPUT_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()
