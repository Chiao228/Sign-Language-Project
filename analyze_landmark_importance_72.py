import torch
import torch.nn as nn
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split

# ==========================================
# 設定與路徑
# ==========================================
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

DATA_PATH = "sliding_window_72"
MODEL_DIR = "train_V23_Transformer_72(with asl weight+ sliding window + K-fold + output F1-score)"
MODEL_PATH = os.path.join(MODEL_DIR, "Fold_4", "best_model.pth") 
OUTPUT_DIR = "landmark_analysis_72"
os.makedirs(OUTPUT_DIR, exist_ok=True)

INPUT_DIM = 72
TARGET_FRAMES = 30

# ==========================================
# 1. 特徵群組定義 (72 維：66 座標 + 6 幾何)
# ==========================================
def generate_all_72_features():
    groups = {}
    hand_names = [
        "手腕 Wrist(0)", "拇指關節 IP(3)", "拇指尖 Tip(4)",
        "食指關節 DIP(7)", "食指尖 Tip(8)", "中指關節 DIP(11)",
        "中指尖 Tip(12)", "無名指關節 DIP(15)", "無名指尖 Tip(16)",
        "小指關節 DIP(19)", "小指尖 Tip(20)"
    ]

    # 1. 左手座標 (0~32)
    for i, name in enumerate(hand_names):
        groups[f"左手 LH {name}"] = [i*3, i*3+1, i*3+2]
    # 幾何特徵 (33~35)
    groups["左手 LH 拇食距離(TI-ED)"] = [33]
    groups["左手 LH 指尖夾角(Angle)"] = [34]
    groups["左手 LH 掌面距離(Plane)"] = [35]

    # 2. 右手座標 (36~68)
    for i, name in enumerate(hand_names):
        groups[f"右手 RH {name}"] = [i*3+36, i*3+37, i*3+38]
    # 幾何特徵 (69~71)
    groups["右手 RH 拇食距離(TI-ED)"] = [69]
    groups["右手 RH 指尖夾角(Angle)"] = [70]
    groups["右手 RH 掌面距離(Plane)"] = [71]
    
    return groups

# --- 2. 模型架構 (需與 train_transformer_72.py 完全一致) ---
class CNNTransformerTSL(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim, num_layers, dropout):
        super(CNNTransformerTSL, self).__init__()
        # CNN 提取局部特徵
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
        
        # Positional Encoding
        self.pos_encoder = nn.Parameter(torch.randn(1, 30, hidden_dim))
        
        # Transformer
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=8, 
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 128), 
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # CNN: (Batch, Frames, Dim) -> (Batch, Dim, Frames)
        x = self.cnn(x.transpose(1, 2)).transpose(1, 2)
        
        # 加入 Positional Encoding
        if x.size(1) <= self.pos_encoder.size(1):
            x = x + self.pos_encoder[:, :x.size(1), :]
        
        # Transformer
        x = self.transformer(x)
        
        # 結合 Mean 與 Max Pooling
        avg_pool = x.mean(dim=1)
        max_pool, _ = x.max(dim=1)
        return self.fc(avg_pool + max_pool)

# 將 72 個特徵作為分析對象
FEATURE_GROUPS = generate_all_72_features()

# ==========================================
# 3. 資料讀取
# ==========================================
class SimpleDataset(Dataset):
    def __init__(self, data_dir, model_dir):
        self.samples, self.labels = [], []
        
        # 讀取標籤映射
        label_map_path = os.path.join(model_dir, "label_map.json")
        if not os.path.exists(label_map_path):
            raise FileNotFoundError(f"❌ 找不到標籤映射檔: {label_map_path}")
            
        with open(label_map_path, "r", encoding="utf-8") as f:
            temp_map = json.load(f)
            self.label_map = {int(k): v for k, v in temp_map.items()}
            name_to_idx = {v: k for k, v in self.label_map.items()}

        all_words = sorted([d for d in os.listdir(data_dir)
                           if os.path.isdir(os.path.join(data_dir, d)) and not d.startswith('.')])
        
        for word in all_words:
            if word not in name_to_idx: continue
            word_idx = name_to_idx[word]
            word_path = os.path.join(data_dir, word)
            files = sorted([f for f in os.listdir(word_path) if f.endswith('.npy')])
            for f in files:
                self.samples.append(os.path.join(word_path, f))
                self.labels.append(word_idx)
                
        self.samples = np.array(self.samples)
        self.labels = np.array(self.labels)

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        data = np.load(self.samples[idx])
        return torch.tensor(data, dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

def collate_fn(batch):
    data, labels = zip(*batch)
    return torch.stack(data), torch.stack(labels)

# ==========================================
# 4. 分析函式
# ==========================================

def evaluate_model(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / total if total > 0 else 0.0

def analyze_variance(dataset, val_idx):
    print("Analysis 1: Statistical Variance")
    all_data = []
    for idx in val_idx:
        data, _ = dataset[idx]
        all_data.append(data.numpy())
    all_data = np.array(all_data) 
    reshaped = all_data.reshape(-1, INPUT_DIM) 
    variances = np.var(reshaped, axis=0) 

    group_variances = {}
    for group_name, indices in FEATURE_GROUPS.items():
        group_variances[group_name] = np.mean(variances[indices])

    # 視覺化
    plt.figure(figsize=(12, 12))
    names = list(group_variances.keys())
    vals = list(group_variances.values())
    plt.barh(names, vals, color='skyblue')
    plt.title("各特徵群組平均變異數 (72維)")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "variance_analysis.png"))
    plt.close()
    return variances

def run_ablation_study(model, dataset, val_idx, device, baseline_acc):
    print("Analysis 2: Ablation Study")
    results = {}
    for group_name, indices in FEATURE_GROUPS.items():
        masked_batch = []
        for idx in val_idx:
            data, label = dataset[idx]
            data_np = data.numpy().copy()
            data_np[:, indices] = 0.0 
            masked_batch.append((torch.tensor(data_np), label))
        
        loader = DataLoader(masked_batch, batch_size=32, shuffle=False, collate_fn=collate_fn)
        masked_acc = evaluate_model(model, loader, device)
        drop = baseline_acc - masked_acc
        results[group_name] = {"masked_acc": masked_acc, "drop": drop}
        print(f"  {group_name:<35} | 準確率下降: {drop:+.2%}")

    plt.figure(figsize=(12, 14))
    names = list(results.keys())
    drops = [results[n]["drop"] * 100 for n in names]
    plt.barh(names, drops, color='salmon')
    plt.title("消融實驗：移除特徵後的準確率下降 (%)")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "ablation_study.png"))
    plt.close()
    return results

def analyze_importance():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Beginning 72-dimensional feature importance analysis...")
    
    # 1. 載入超參數
    params_path = os.path.join(MODEL_DIR, "best_params.json")
    if os.path.exists(params_path):
        with open(params_path, "r", encoding="utf-8") as f:
            bp = json.load(f)
    else:
        bp = {'hidden_dim': 128, 'num_layers': 1, 'dropout': 0.5}

    # 2. 載入資料
    dataset = SimpleDataset(DATA_PATH, MODEL_DIR)
    num_classes = len(dataset.label_map)
    _, val_idx = train_test_split(np.arange(len(dataset.labels)), test_size=0.3, stratify=dataset.labels, random_state=42)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=32, shuffle=False, collate_fn=collate_fn)
    
    # 3. 載入模型
    model = CNNTransformerTSL(INPUT_DIM, num_classes, bp['hidden_dim'], bp['num_layers'], bp['dropout']).to(device)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print(f"Successfully loaded model: {MODEL_PATH}")
    else:
        print(f"Cannot find model file: {MODEL_PATH}")
        return

    # 4. 執行分析
    baseline_acc = evaluate_model(model, val_loader, device)
    print(f"Baseline Accuracy: {baseline_acc:.2%}")
    
    analyze_variance(dataset, val_idx)
    run_ablation_study(model, dataset, val_idx, device, baseline_acc)
    
    print(f"Analysis complete! Charts saved to {OUTPUT_DIR}/")

if __name__ == "__main__":
    analyze_importance()
