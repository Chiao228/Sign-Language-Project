import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from pathlib import Path

# ==========================================
# 1. 模型架構 (需與 train_transformer.py 一致)
# ==========================================
class CNNTransformerTSL(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim, num_layers, dropout=0.4):
        super(CNNTransformerTSL, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(128, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.pos_encoder = nn.Parameter(torch.randn(1, 30, hidden_dim))
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=8, dim_feedforward=hidden_dim * 2, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 128), nn.ReLU(), nn.Dropout(0.5), nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.cnn(x.transpose(1, 2)).transpose(1, 2)
        if x.size(1) <= self.pos_encoder.size(1):
            x = x + self.pos_encoder[:, :x.size(1), :]
        x = self.transformer(x)
        avg_pool = x.mean(dim=1)
        max_pool, _ = x.max(dim=1)
        return self.fc(avg_pool + max_pool)

# ==========================================
# 2. 視覺化核心邏輯
# ==========================================
def visualize_sample_importance(model, sample_path, label_name, label_idx, output_folder, device):
    """
    計算並視覺化單一樣本中，每一幀的重要性 (Saliency Map)
    """
    model.eval()
    try:
        data = np.load(sample_path).astype(np.float32)
        # 確保資料長度為 30
        if data.shape[0] != 30:
            return
            
        input_tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0).to(device)
        input_tensor.requires_grad = True # 開啟梯度追蹤
        
        # 前向傳播
        output = model(input_tensor)
        
        # 取得預測結果
        pred_idx = output.argmax(1).item()
        confidence = torch.softmax(output, dim=1)[0, pred_idx].item()
        
        # 使用正確標籤的分數進行反向傳播
        score = output[0, label_idx]
        
        model.zero_grad()
        score.backward()
        
        # 取得梯度 (Saliency)
        # 梯度絕對值的平均代表該幀對判斷該詞彙的「貢獻度」
        saliency = input_tensor.grad.data.abs().mean(dim=2).squeeze().cpu().numpy()
        
        # 歸一化到 0~1 方便觀察
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
        
        # 標註最高能量幀 (最關鍵時刻)
        max_idx = np.argmax(saliency)

        # === 核心動作驗證 (Ablation Test) ===
        # 實驗：如果我們把最關鍵的那一幀「抹除」(設為0)，模型會變笨嗎？
        test_tensor = input_tensor.clone()
        with torch.no_grad():
            # 抹除關鍵幀
            test_tensor[0, max_idx, :] = 0
            # 也可以試著抹除周圍的小區間 (例如前後1幀)，效果會更明顯
            if max_idx > 0: test_tensor[0, max_idx-1, :] = 0
            if max_idx < 29: test_tensor[0, max_idx+1, :] = 0
            
            output_after = model(test_tensor)
            conf_after = torch.softmax(output_after, dim=1)[0, label_idx].item()
            
        drop_rate = (confidence - conf_after) / (confidence + 1e-8)
        # =================================

        # 繪圖
        plt.figure(figsize=(10, 7))
        frames = np.arange(len(saliency))
        
        # 畫長條圖
        bars = plt.bar(frames, saliency, color='skyblue', alpha=0.7, label='Importance')
        # 畫折線圖
        plt.plot(frames, saliency, color='blue', marker='o', markersize=4, alpha=0.4)
        
        # 設定標題與資訊
        status_text = "真實核心" if drop_rate > 0.2 else "非唯一核心"
        plt.title(f"手語關鍵幀分析: 『{label_name}』\n原始信心度: {confidence:.2%} -> 抹除後: {conf_after:.2%}\n(信心度下降: {drop_rate:.2%} | 判定: {status_text})", 
                  fontsize=13, color='darkred' if drop_rate > 0.2 else 'black')
        
        plt.xlabel("時間軸 (Frame Index)", fontsize=12)
        plt.ylabel("影響力權重 (Contribution Score)", fontsize=12)
        plt.xticks(np.arange(0, 31, 5))
        plt.ylim(0, 1.2)
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        
        # 標註最高能量幀
        plt.annotate(f'核心動作 (第 {max_idx} 幀)\n抹除此處掉分: {drop_rate:.1%}', xy=(max_idx, saliency[max_idx]), 
                     xytext=(max_idx+2, saliency[max_idx]+0.1),
                     arrowprops=dict(facecolor='red', shrink=0.05, width=2, headwidth=8),
                     fontsize=10, color='red', fontweight='bold')
        
        plt.tight_layout()
        save_name = f"importance_{label_name}_{Path(sample_path).stem}.png"
        save_path = os.path.join(output_folder, save_name)
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"✅ 已生成分析圖表: {save_name} (掉分率: {drop_rate:.2%})")
        
    except Exception as e:
        import traceback
        print(f"❌ 分析樣本 {sample_path} 時出錯: {e}")
        traceback.print_exc()

# ==========================================
# 3. 主程式
# ==========================================
def main():
    # 根據您的環境設定路徑
    MODEL_FOLDER = "train_V16_Transformer_66(with asl weight + sliding window)"
    WEIGHTS_PATH = os.path.join(MODEL_FOLDER, "best_model.pth")
    LABEL_MAP_PATH = os.path.join(MODEL_FOLDER, "label_map.json")
    PARAMS_PATH = os.path.join(MODEL_FOLDER, "best_params.json")
    DATA_DIR = "sliding_window_66"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  使用設備: {device}")

    # 檢查必要檔案
    if not os.path.exists(WEIGHTS_PATH):
        print(f"❌ 找不到模型權重檔: {WEIGHTS_PATH}")
        return

    # 載入設定與模型
    with open(LABEL_MAP_PATH, 'r', encoding='utf-8') as f:
        label_map = json.load(f)
    with open(PARAMS_PATH, 'r') as f:
        bp = json.load(f)

    num_classes = len(label_map)
    model = CNNTransformerTSL(66, num_classes, bp['hidden_dim'], bp['num_layers'], bp['dropout']).to(device)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    print("🚀 模型與參數載入完成！")

    # 建立視覺化結果資料夾
    VIS_DIR = os.path.join(MODEL_FOLDER, "attention_visuals")
    os.makedirs(VIS_DIR, exist_ok=True)

    # 隨機挑選 10 個不同的詞彙來分析
    words_to_analyze = list(label_map.items())
    np.random.shuffle(words_to_analyze)
    
    analyzed_count = 0
    for word_idx_str, word_name in words_to_analyze:
        if analyzed_count >= 10: break # 只分析 10 個範例
        
        word_idx = int(word_idx_str)
        word_folder = os.path.join(DATA_DIR, word_name)
        if not os.path.exists(word_folder): continue
        
        files = [f for f in os.listdir(word_folder) if f.endswith('.npy')]
        if not files: continue
        
        # 挑選第一個滑窗檔案進行分析
        sample_path = os.path.join(word_folder, files[0])
        visualize_sample_importance(model, sample_path, word_name, word_idx, VIS_DIR, device)
        analyzed_count += 1

    print("\n" + "=" * 50)
    print(f"✨ 分析完成！")
    print(f"📁 視覺化圖片已存至: {VIS_DIR}")
    print("=" * 50)

if __name__ == "__main__":
    # 設定中文字體避免圖片亂碼
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial Unicode MS', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    main()
