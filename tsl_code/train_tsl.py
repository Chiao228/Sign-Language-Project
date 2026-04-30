import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import os
import random
import json
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import interp1d
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import optuna

# --- 1. 環境與輸出資料夾設定 ---
OUTPUT_DIR = "train_V6_GRU"
os.makedirs(OUTPUT_DIR, exist_ok=True)

#PRETRAINED_WEIGHTS = "asl_best_model.pth" 
PRETRAINED_WEIGHTS = " " 


plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"🌱 Random seed set to: {seed}")

# --- 引用擴增技術 ---
try:
    from augment_data import rotate_landmarks_safe, add_noise_safe, random_shift_safe, scale_hand_parts, finger_dropout ,horizontal_flip_safe
    print("✅ Successfully imported advanced augmentations")
except ImportError:
    def rotate_landmarks_safe(x): return x
    def add_noise_safe(x): return x
    def random_shift_safe(x): return x
    def scale_hand_parts(x): return x
    def finger_dropout(x): return x
    def horizontal_flip_safe(x): return x
    print("⚠️ Warning: augment_data.py missing.")

# --- 2. 固定參數 ---
INPUT_DIM = 132      # 132 (座標)
TARGET_FRAMES = 30   
DATA_PATH = "processed_npy_132"

# --- 3. 資料讀取器 ---
class TSLDataset(Dataset):
    def __init__(self, data_dir, augment=False, target_frames=30):
        self.samples, self.labels = [], []
        self.augment, self.target_frames = augment, target_frames
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"找不到資料路徑: {data_dir}")
            
        all_words = sorted([d for d in os.listdir(data_dir) 
                           if os.path.isdir(os.path.join(data_dir, d)) and not d.startswith('.')])
        self.label_map = {i: word for i, word in enumerate(all_words)}
        
        for word_idx, word in enumerate(all_words):
            word_path = os.path.join(data_dir, word)
            for f in os.listdir(word_path):
                if f.endswith('.npy'):
                    self.samples.append(os.path.join(word_path, f))
                    self.labels.append(word_idx)
        self.samples, self.labels = np.array(self.samples), np.array(self.labels)

    def __len__(self): return len(self.samples)
    
    def __getitem__(self, idx):
        data = np.load(self.samples[idx])
        
        if self.augment:
            # --- 1. 平移 + 縮放
            if random.random() < 0.6:  
                data, _ = random_shift_safe(data)  
                data, _ = scale_hand_parts(data)    

            # --- 2. 模擬左撇子 ---
            if random.random() < 0.5: 
                data = horizontal_flip_safe(data)   

            # --- 3. 破壞性/干擾性擴增 ---
            if random.random() < 0.3:
                data, _ = rotate_landmarks_safe(data)
            if random.random() < 0.3: 
                data, _ = finger_dropout(data)        
            if random.random() < 0.2: 
                data, _ = add_noise_safe(data)              
            
        return torch.tensor(data, dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

def collate_fn(batch):
    data, labels = zip(*batch)
    return torch.stack(data), torch.stack(labels)

# --- 4. 模型架構 ---
class CNNGRUTSL(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim, num_layers, dropout=0.4):
        super(CNNGRUTSL, self).__init__()
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
        
        # [修改點] 更換為雙向 GRU
        self.gru = nn.GRU(
            input_size=hidden_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True # 雙向捕捉動作軌跡
        )
        
        # 因為是雙向，輸入維度要 * 2
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128), 
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # CNN: (Batch, Frames, Dim) -> (Batch, Dim, Frames)
        x = self.cnn(x.transpose(1, 2)).transpose(1, 2)
        
        # GRU: (Batch, Frames, hidden_dim * 2)
        x, _ = self.gru(x)
        
        # 結合 Mean 與 Max Pooling
        avg_pool = x.mean(dim=1)
        max_pool, _ = x.max(dim=1)
        return self.fc(avg_pool + max_pool)

# --- 5. 視覺化函式 ---
def plot_training_curves(history, folder):
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss'); plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss Curve'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc'); plt.plot(history['val_acc'], label='Val Acc')
    plt.title('Accuracy Curve'); plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(folder, "full_training_report.png"))
    plt.close()

def plot_confusion_matrix(y_true, y_pred, classes, folder):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('TSL 混淆矩陣 (Confusion Matrix)'); plt.ylabel('真實標籤'); plt.xlabel('預測標籤')
    plt.savefig(os.path.join(folder, "final_confusion_matrix.png"))
    plt.close()

def safe_save_model(state_dict, path):
    """
    解決 Windows/OneDrive 檔案鎖定問題 (Error 32) 的安全儲存函式
    """
    import time
    for i in range(5):
        try:
            torch.save(state_dict, path)
            return
        except RuntimeError as e:
            if "error code: 32" in str(e) or "PermissionDenied" in str(e):
                print(f"⚠️ 偵測到 OneDrive 檔案鎖定，1秒後重試... ({i+1}/5)")
                time.sleep(1)
            else:
                raise e
    print(f"❌ 無法儲存模型至 {path}，請檢查檔案是否被其他程式開啟。")

# --- 6. 權重載入小工具 ---
def load_pretrained_weights(model, weights_path, device):
    """
    自動載入預訓練權重，並跳過形狀不符的層 (例如分類層)。
    """
    if os.path.exists(weights_path):
        state_dict = torch.load(weights_path, map_location=device)
        model_dict = model.state_dict()
        # 僅載入名稱與形狀皆相符的權重
        new_state_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
        model.load_state_dict(new_state_dict, strict=False)
        return len(new_state_dict)
    return 0

# --- 6. Optuna 目標函式 ---
def objective(trial):
    h_dim = trial.suggest_categorical('hidden_dim', [128, 256])
    n_layers = trial.suggest_int('num_layers', 1, 3) 
    lr = trial.suggest_float('learning_rate', 1e-4, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16])
    dropout = trial.suggest_float('dropout', 0.5, 0.6)
    wd = trial.suggest_float('weight_decay', 1e-2, 1e-1, log=True)
    ls = trial.suggest_float('label_smoothing', 0.05, 0.1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = TSLDataset(DATA_PATH, target_frames=TARGET_FRAMES)
    
    # 修改：依照需求調整為 70/30 切分 (test_size=0.3)
    train_idx, val_idx = train_test_split(
        np.arange(len(dataset.labels)), 
        test_size=0.3, 
        stratify=dataset.labels, 
        random_state=42
    )
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    model = CNNGRUTSL(INPUT_DIM, len(dataset.label_map), h_dim, n_layers, dropout).to(device)
    
    # 遷移學習：搜尋時也載入基礎權重
    load_pretrained_weights(model, PRETRAINED_WEIGHTS, device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.CrossEntropyLoss(label_smoothing=ls)

    for epoch in range(30):
        model.train(); dataset.augment = True
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad(); out = model(bx); loss = criterion(out, by); loss.backward(); optimizer.step()
        
        model.eval(); dataset.augment = False
        v_loss = 0
        with torch.no_grad():
            for vx, vy in val_loader:
                vx, vy = vx.to(device), vy.to(device)
                out = model(vx); v_loss += criterion(out, vy).item()
        
        avg_v_loss = v_loss / len(val_loader)
        trial.report(avg_v_loss, epoch)
        if trial.should_prune(): raise optuna.exceptions.TrialPruned()
    return avg_v_loss 

# 檢查函式
def save_augmentation_preview(dataset, output_path):
    import matplotlib.pyplot as plt
    dataset.augment = True # 開啟隨機擴增
    
    # 抓取同一筆原始資料兩次，觀察其隨機產生的變體
    data1, _ = dataset[0] 
    data2, _ = dataset[0]
    
    plt.figure(figsize=(12, 6))
    
    # 繪製曲線
    plt.plot(data1[0].numpy(), label='Random Augmentation 1 (隨機擴增 1)', alpha=0.7)
    plt.plot(data2[0].numpy(), label='Random Augmentation 2 (隨機擴增 2)', alpha=0.7, linestyle='--')
    
    # --- 加入軸標籤與說明 ---
    plt.title("Online Augmentation Check (動態擴增預覽)", fontsize=14)
    plt.xlabel("Feature Index (138-dim: Hands, Face, Shoulders)\n特徵索引 (138維：雙手、面部、雙肩)", fontsize=10)
    plt.ylabel("Normalized Coordinate Value (Relative Position)\n正規化座標值 (相對位置)", fontsize=10)
    
    # 加入格線輔助觀察偏移量
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    
    # 儲存檔案
    preview_file = os.path.join(output_path, "online_aug_preview.png")
    plt.savefig(preview_file, bbox_inches='tight')
    plt.close()
    print(f"📸 含有軸標籤的預覽圖已存至：{preview_file}")

# --- 7. 主程式 ---
def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    full_dataset = TSLDataset(DATA_PATH, target_frames=TARGET_FRAMES)

    # --------------------------
    save_augmentation_preview(full_dataset, OUTPUT_DIR)
    # --------------------------

    num_classes = len(full_dataset.label_map)
    class_names = [full_dataset.label_map[i] for i in range(num_classes)]
    
    # 儲存 Label Map
    label_map_path = os.path.join(OUTPUT_DIR, 'label_map.json')
    with open(label_map_path, 'w', encoding='utf-8') as f:
        json.dump(full_dataset.label_map, f, ensure_ascii=False, indent=4)
    print(f"✅ Label Map 已儲存至 {label_map_path}")

    # B. 超參數搜尋 (優先檢測現有參數)
    params_file = os.path.join(OUTPUT_DIR, "best_params.json")
    
    if os.path.exists(params_file):
        print(f"📄 檢測到現有的參數檔，直接讀取：{params_file}")
        with open(params_file, "r", encoding="utf-8") as f:
            bp = json.load(f)
    else:
        print("🔍 Step 1: Hyperparameter Optimization (70/30 Split)...")
        sampler = optuna.samplers.TPESampler(seed=42)
        study = optuna.create_study(direction="minimize", sampler=sampler)
        study.optimize(objective, n_trials=25)
        bp = study.best_params
        
        with open(params_file, "w", encoding="utf-8") as f:
            json.dump(bp, f, indent=4, ensure_ascii=False)
        print(f"✅ 最佳參數已存至 {params_file}")

    # C. 正式訓練
    # 修改：依照需求調整為 70/30 切分 (test_size=0.3)
    train_idx, val_idx = train_test_split(
        np.arange(len(full_dataset.labels)), 
        test_size=0.3, 
        stratify=full_dataset.labels, 
        random_state=42
    )
    train_loader = DataLoader(Subset(full_dataset, train_idx), batch_size=bp['batch_size'], shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(Subset(full_dataset, val_idx), batch_size=bp['batch_size'], shuffle=False, collate_fn=collate_fn)

    model = CNNGRUTSL(INPUT_DIM, num_classes, bp['hidden_dim'], bp['num_layers'], bp['dropout']).to(device)
    
    # --- 加載預訓練權重 (Transfer Learning / Warm Start) ---
    loaded_count = load_pretrained_weights(model, PRETRAINED_WEIGHTS, device)
    if loaded_count > 0:
        print(f"✅ 成功載入 {loaded_count} 個層的預訓練權重！")
    else:
        print(f"ℹ️ 未發現權重檔 {PRETRAINED_WEIGHTS}，將從隨機初始化開始訓練。")
    optimizer = optim.Adam(model.parameters(), lr=bp['learning_rate'], weight_decay=bp['weight_decay'])
    criterion = nn.CrossEntropyLoss(label_smoothing=bp['label_smoothing'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15)

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    min_val_loss, epochs_no_improve = float('inf'), 0
    PATIENCE, EPOCHS = 30, 300

    print(f"\n🚀 Final Training (70% Train, 30% Val)...")
    for epoch in range(EPOCHS):
        model.train(); full_dataset.augment = True
        t_loss, t_correct, t_total = 0, 0, 0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad(); out = model(bx); loss = criterion(out, by); loss.backward(); optimizer.step()
            t_loss += loss.item(); t_correct += (out.argmax(1) == by).sum().item(); t_total += by.size(0)

        model.eval(); full_dataset.augment = False
        v_loss, v_correct, v_total = 0, 0, 0
        with torch.no_grad():
            for vx, vy in val_loader:
                vx, vy = vx.to(device), vy.to(device)
                out = model(vx); v_loss += criterion(out, vy).item()
                v_correct += (out.argmax(1) == vy).sum().item(); v_total += vy.size(0)

        avg_v_loss = v_loss/len(val_loader)
        history['train_loss'].append(t_loss/len(train_loader)); history['val_loss'].append(avg_v_loss)
        history['train_acc'].append(t_correct/t_total); history['val_acc'].append(v_correct/v_total)
        scheduler.step(avg_v_loss)

        if avg_v_loss < min_val_loss:
            min_val_loss = avg_v_loss
            safe_save_model(model.state_dict(), os.path.join(OUTPUT_DIR, "best_model.pth"))
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:03d} | Val Loss: {avg_v_loss*100:.2f}% | Val Acc: {history['val_acc'][-1]:.2%}")

        if epochs_no_improve >= PATIENCE: 
            print(f"🛑 Early stopping at epoch {epoch+1}")
            break

    # D. 產出報告與 ONNX 轉換
    plot_training_curves(history, OUTPUT_DIR)
    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "best_model.pth")))
    model.eval()

    print("\n📦 Exporting Optimal Model to ONNX...")
    dummy_input = torch.randn(1, TARGET_FRAMES, INPUT_DIM).to(device)
    onnx_path = os.path.join(OUTPUT_DIR, "tsl_model.onnx")
    
    torch.onnx.export(
        model, dummy_input, onnx_path, 
        input_names=['input'], output_names=['output'],
        opset_version=16, 
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}} 
    )
    print(f"✅ ONNX model saved to: {onnx_path}")

    # 產出混淆矩陣
    all_preds, all_trues = [], []
    with torch.no_grad():
        for vx, vy in val_loader:
            vx, vy = vx.to(device), vy.to(device)
            out = model(vx); all_preds.extend(out.argmax(1).cpu().numpy()); all_trues.extend(vy.cpu().numpy())
    plot_confusion_matrix(all_trues, all_preds, class_names, OUTPUT_DIR)
    print(f"✨ All Tasks Completed! Files saved in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()


'''# 假設 bp 是你讀取進來的固定參數字典
base_lr = bp['learning_rate']

# --- [修正點] 先定義參數清單變數，確保後續階段性凍結可用 ---
backbone_params = list(model.cnn.parameters()) + list(model.transformer_encoder.parameters())
head_params = list(model.fc.parameters())

# --- 關鍵修改：差異化參數群組 ---
optimizer = optim.Adam([
    {
        'params': backbone_params, 
        'lr': base_lr * 0.1  # 保護預訓練知識
    },
    {
        'params': head_params, 
        'lr': base_lr        # 分類頭用 1.0 倍
    }
], weight_decay=bp['weight_decay'])

# 損失函數與排程器
criterion = nn.CrossEntropyLoss(label_smoothing=bp['label_smoothing'])
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15)

history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
min_val_loss, epochs_no_improve = float('inf'), 0
PATIENCE, EPOCHS = 50, 300

# --- [優化] 階段性凍結 ---
FREEZE_EPOCHS = 10  
print(f"\n❄️ Phase 1: Freezing Backbone for {FREEZE_EPOCHS} epochs...")

print(f"\n🚀 Final Training (70% Train, 30% Val)...")
for epoch in range(EPOCHS):
    # 階段性解凍邏輯
    if epoch < FREEZE_EPOCHS:
        model.train() 
        model.fc.train() # 只讓分類頭進入訓練模式
        for param in backbone_params: 
            param.requires_grad = False
    else:
        if epoch == FREEZE_EPOCHS:
            print("🔥 Phase 2: Unfreezing all layers for full fine-tuning...")
        model.train()
        for param in backbone_params: 
            param.requires_grad = True

    full_dataset.augment = True
    t_loss, t_correct, t_total = 0, 0, 0
    # ... (後續訓練 Loop 保持不變)'''