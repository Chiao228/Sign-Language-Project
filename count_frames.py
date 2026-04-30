import cv2
import os

def count_frames(video_path):
    """
    計算影片的總影格數 (Total Frames)
    """
    if not os.path.exists(video_path):
        print(f"錯誤: 找不到檔案 {video_path}")
        return None

    # 使用 OpenCV 打開影片
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"錯誤: 無法打開影片 {video_path}")
        return None

    # 方法 1: 從影片屬性直接讀取 (速度最快)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 如果 OpenCV 回傳 0 或負數，嘗試手動讀取計算 (較慢但精準)
    if total_frames <= 0:
        print("警告: 無法從屬性讀取影格數，切換至手動計數模式...")
        total_frames = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            total_frames += 1
            
    cap.release()
    return total_frames

def count_frames_in_directory(directory_path, extensions=('.mp4', '.mkv', '.avi', '.mov')):
    """
    遞迴掃描資料夾及其子資料夾內所有影片的影格計數
    """
    if not os.path.isdir(directory_path):
        print(f"錯誤: {directory_path} 不是有效的資料夾")
        return

    print(f"正在掃描資料夾: {directory_path} ...")
    
    video_paths = []
    for root, dirs, files in os.walk(directory_path):
        for f in files:
            if f.lower().endswith(extensions):
                video_paths.append(os.path.join(root, f))
    
    if not video_paths:
        print(f"在 {directory_path} (及其子資料夾) 中找不到影片檔案 {extensions}")
        return

    print(f"{'影片路徑':<60} | {'總影格數':<10}")
    print("-" * 75)

    results = []
    for full_path in video_paths:
        # 顯示相對路徑比較好讀
        rel_path = os.path.relpath(full_path, directory_path)
        frames = count_frames(full_path)
        if frames is not None:
            print(f"{rel_path:<60} | {frames:<10}")
            results.append((rel_path, frames))
    
    print("-" * 75)
    print(f"處理完成，共檢查了 {len(results)} 個影片。")
    return results

if __name__ == "__main__":
    # --- 請修改這裡 ---
    # 你可以填入資料夾路徑，例如 "tsl" 或 "C:/Videos"
    target_dir = "tsl" 
    # ----------------
    
    if os.path.isdir(target_dir):
        all_results = count_frames_in_directory(target_dir)
        
        # --- 額外篩選：找出剛好 30 偵的影片 ---
        print("\n[ 篩選結果 ] 剛好 30 偵的影片：")
        target_frames = 30
        filtered = [path for path, frames in all_results if frames == target_frames]
        
        if filtered:
            for path in filtered:
                print(f"-> {path}")
            print(f"\n找到 {len(filtered)} 個影片剛好為 {target_frames} 偵。")
        else:
            print(f"沒找到剛好 {target_frames} 偵的影片。")
    
    else:
        # 如果不是資料夾，就當作單一檔案處理
        frames = count_frames(target_dir)
        if frames is not None:
            print(f"影片: {target_dir} | 總影格數: {frames}")
