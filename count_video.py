import os

def save_video_count_to_md(root_path, output_filename="video_count.md"):
    # 定義影片副檔名
    video_extensions = ('.mp4', '.mkv')
    
    # 建立 Markdown 表格標頭
    md_content = [
        "# 影片檔案數量統計報告\n",
        "| 詞彙 | 影片數量 |",
        "| :--- | :--- |"
    ]

    try:
        # 取得所有項目
        items = os.listdir(root_path)
        folder_counts = []
        
        for item in items:
            item_path = os.path.join(root_path, item)
            
            # 確保是資料夾才處理
            if os.path.isdir(item_path):
                # 計算該子資料夾內的影片數
                files = os.listdir(item_path)
                video_count = sum(1 for f in files if f.lower().endswith(video_extensions))
                folder_counts.append((item, video_count))
        
        # 按照影片數量遞減排序
        folder_counts.sort(key=lambda x: x[1], reverse=True)
        
        # 加入表格列
        for item, video_count in folder_counts:
            md_content.append(f"| {item} | {video_count} |")

        # 將內容寫入檔案
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write("\n".join(md_content))
            
        print(f"成功！報告已儲存至: {os.path.abspath(output_filename)}")

    except Exception as e:
        print(f"發生錯誤: {e}")

# --- 設定區 ---
target_folder = "tsl"  # 你的目標資料夾
save_video_count_to_md(target_folder)