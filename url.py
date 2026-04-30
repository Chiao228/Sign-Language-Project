import yt_dlp
import json
import time

# 您的 46 個詞彙清單
vocab_list = [
    "謝謝", "不客氣", "幫忙", "檢查", "高興", "不可以", "忘記", 
    "沒關係", "計程車", "記得", "認真", "中午", "你好", "名字", "喜歡", 
    "媽媽", "明天", "有沒有", "棒", "爸爸", "生氣", "蘋果", "不喜歡", 
    "不是", "今天(現在)", "去", "可以", "幾點", "找", "有", "朋友", 
    "機車", "飛機", "飲料", "休息", "公車", "告訴", 
    "我們", "放學", "是", "會", "要", "還沒", "好吃", "我", "說話",
    "火車", "再見", "對不起", "高鐵"
]

# 限定搜尋 K12EA 頻道的教保服務人員手語手冊
base_search_query = "ytsearch1:K12ea 教保服務人員手語手冊 "
output_filename = 'tsl_vocab_videos.json'

# 存放最終結果的陣列
final_data = []

ydl_opts = {
    'quiet': True,
    'extract_flat': True, # 只抓取元資料，不下載影片
    'force_generic_extractor': False
}

print("開始抓取影片連結，這可能需要幾分鐘的時間...\n")

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    for word in vocab_list:
        query = f"{base_search_query}{word}"
        print(f"正在搜尋: {word} ...", end=" ")
        
        # 直接以 tsl_ 加上中文詞彙作為標準語意命名的 ID
        vocab_obj = {
            "vocab_id": word,
            "word_zh": word,
            "video_url": None
        }
        
        try:
            info = ydl.extract_info(query, download=False)
            if 'entries' in info and len(info['entries']) > 0:
                video_url = info['entries'][0]['url']
                vocab_obj["video_url"] = video_url
                print(f"成功 -> {video_url}")
            else:
                print("找不到影片")
        except Exception as e:
            print(f"發生錯誤 -> {e}")
            
        final_data.append(vocab_obj)
        
        # 設定延遲，避免請求過於頻繁被 YouTube 伺服器阻擋
        time.sleep(1.5)

# 輸出成供前端讀取的 JSON 檔案
with open(output_filename, 'w', encoding='utf-8') as f:
    json.dump(final_data, f, ensure_ascii=False, indent=4)

print(f"\n抓取完成！檔案已存為：{output_filename}")