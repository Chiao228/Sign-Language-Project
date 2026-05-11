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
from sklearn.metrics import confusion_matrix, f1_score, classification_report
from sklearn.model_selection import train_test_split, StratifiedKFold, StratifiedShuffleSplit
import optuna

# --- 1. 環境與輸出資料夾設定 ---
OUTPUT_DIR = "train_V27_Transformer_66(with asl weight+ sliding window + K-fold + output F1-score + parameter_asl)"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PRETRAINED_WEIGHTS = "transformer_66_BS16_LR0.001_best.pth" 
#PRETRAINED_WEIGHTS = " " 

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
INPUT_DIM = 66       # 與預訓練權重一致 (66)
TARGET_FRAMES = 30   
DATA_PATH = "sliding_window_66"

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
        
        # --- 加速優化：將所有資料預載入記憶體 ---
        print(f"📦 正在將 {len(self.samples)} 筆資料載入記憶體以加速訓練...")
        self.data_cache = [np.load(s) for s in self.samples]
        print(f"✅ 載入完成！(減少 OneDrive/硬碟讀取延遲)")

    def __len__(self): return len(self.samples)
    
    def __getitem__(self, idx):
        # 從快取讀取資料並複製，避免擴增時修改到原始數據
        data = self.data_cache[idx].copy()
        
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
class CNNTransformerTSL(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim, num_layers, dropout, nhead=4):
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
        
        # Positional Encoding (針對時間序列的可學習位置編碼)
        self.pos_encoder = nn.Parameter(torch.randn(1, 30, hidden_dim))
        
        # [修改點] 更換為 Transformer
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=nhead, # 與預訓練權重一致 (4)
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True
        )
        # 為了與預訓練權重的 key 名稱對齊，改為 transformer_encoder
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # Transformer 沒有雙向的概念，因此輸出維度就是 hidden_dim
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
        
        # Transformer: (Batch, Frames, hidden_dim)
        x = self.transformer_encoder(x)
        
        # 結合 Mean 與 Max Pooling
        avg_pool = x.mean(dim=1)
        max_pool, _ = x.max(dim=1)
        return self.fc(avg_pool + max_pool)

# --- 5. 視覺化函式 ---
def plot_comprehensive_report(history, y_true, y_pred, classes, folder, fold_num):
    """
    將 Loss, Accuracy, F1 摺線圖與每類別的詳細分類報告 (Precision, Recall, F1, Support) 整合在同一張圖中。
    """
    # 1. 計算分類報告 (字典格式)
    report_dict = classification_report(y_true, y_pred, target_names=classes, output_dict=True, zero_division=0)
    
    # 提取資料 (排除統計摘要行)
    metrics_keys = ['precision', 'recall', 'f1-score', 'support']
    labels = [k for k in report_dict.keys() if k not in ['accuracy', 'macro avg', 'weighted avg']]
    data = [[report_dict[l][m] for m in metrics_keys] for l in labels]
    
    # 2. 建立畫布
    fig = plt.figure(figsize=(22, 18))
    # 上方 1/3 放摺線圖，下方 2/3 放熱圖 (適用於詞彙較多的情況)
    gs = plt.GridSpec(2, 3, height_ratios=[1, 2.5])
    
    # (A) Loss 摺線圖
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(history['train_loss'], label='Train Loss', color='blue', linewidth=2)
    ax1.plot(history['val_loss'], label='Val Loss', color='orange', linewidth=2)
    ax1.set_title(f'Fold {fold_num} - Loss Curve', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss'); ax1.legend(); ax1.grid(True, alpha=0.3)
    
    # (B) Accuracy 摺線圖
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(history['train_acc'], label='Train Acc', color='green', linewidth=2)
    ax2.plot(history['val_acc'], label='Val Acc', color='red', linewidth=2)
    ax2.set_title(f'Fold {fold_num} - Accuracy Curve', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy'); ax2.legend(); ax2.grid(True, alpha=0.3)
    
    # (C) F1-Score 摺線圖
    ax3 = fig.add_subplot(gs[0, 2])
    if 'val_f1' in history:
        ax3.plot(history['val_f1'], label='Val F1 (Macro)', color='purple', linewidth=2)
        ax3.set_title(f'Fold {fold_num} - F1 Score Curve', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Epoch'); ax3.set_ylabel('F1 Score'); ax3.legend(); ax3.grid(True, alpha=0.3)
    
    # (D) 每詞彙詳細指標熱圖 (Precision, Recall, F1, Support)
    ax4 = fig.add_subplot(gs[1, :])
    sns.heatmap(data, annot=True, fmt='.2f', cmap='YlGnBu', 
                xticklabels=['Precision', 'Recall', 'F1-Score', 'Support'],
                yticklabels=labels, ax=ax4, annot_kws={"size": 10}, cbar=True)
    ax4.set_title(f'Fold {fold_num} - Per-word Detailed Report', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    report_img_path = os.path.join(folder, "comprehensive_report.png")
    plt.savefig(report_img_path, dpi=120)
    plt.close()
    print(f"📊 綜合報告圖表已存至：{report_img_path}")

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
    h_dim = trial.suggest_categorical('hidden_dim', [128])
    n_layers = trial.suggest_categorical('num_layers', [2]) # 固定為 2 層以匹配權重
    lr = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16])
    dropout = trial.suggest_float('dropout', 0.5, 0.6)
    wd = trial.suggest_float('weight_decay', 1e-2, 1e-1, log=True)
    ls = trial.suggest_float('label_smoothing', 0.05, 0.1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = TSLDataset(DATA_PATH, target_frames=TARGET_FRAMES)
    
    # 修改：依照需求調整為 80/20 切分 (test_size=0.2)
    train_idx, val_idx = train_test_split(
        np.arange(len(dataset.labels)), 
        test_size=0.2, 
        stratify=dataset.labels, 
        random_state=42
    )
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    model = CNNTransformerTSL(INPUT_DIM, len(dataset.label_map), h_dim, n_layers, dropout, nhead=4).to(device)
    
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
    plt.xlabel(f"Feature Index ({INPUT_DIM}-dim: Hands/Skeleton)\n特徵索引 ({INPUT_DIM}維：手部/骨架點)", fontsize=10)
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
    print(f"\n🔥 目前正在使用的運算設備: {device}")
    if device.type == 'cuda':
        print(f"✨ 偵測到顯卡: {torch.cuda.get_device_name(0)}")
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
        print("🔍 Step 1: Hyperparameter Optimization (80/20 Split)...")
        sampler = optuna.samplers.TPESampler(seed=42)
        study = optuna.create_study(direction="minimize", sampler=sampler)
        study.optimize(objective, n_trials=25)
        bp = study.best_params
        
        with open(params_file, "w", encoding="utf-8") as f:
            json.dump(bp, f, indent=4, ensure_ascii=False)
        print(f"✅ 最佳參數已存至 {params_file}")

    # C. 正式訓練 (改為 K-Fold 交叉驗證)
    K_FOLDS = 5
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    
    all_fold_accuracies = []
    all_fold_f1_scores = []
    all_indices = np.arange(len(full_dataset.labels))
    all_labels = full_dataset.labels
    
    print(f"\n🔍 Step 2: Final Training with {K_FOLDS}-Fold Cross Validation...")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(all_indices, all_labels)):
        print(f"\n" + "="*40)
        print(f"🚀 Training Fold {fold+1}/{K_FOLDS}")
        print("="*40)

        # 建立該 Fold 的子資料夾
        FOLD_DIR = os.path.join(OUTPUT_DIR, f"Fold_{fold+1}")
        os.makedirs(FOLD_DIR, exist_ok=True)

        train_loader = DataLoader(Subset(full_dataset, train_idx), batch_size=bp['batch_size'], shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(Subset(full_dataset, val_idx), batch_size=bp['batch_size'], shuffle=False, collate_fn=collate_fn)

        # 每次 Fold 都重新初始化模型與優化器
        model = CNNTransformerTSL(INPUT_DIM, num_classes, bp['hidden_dim'], bp['num_layers'], bp['dropout'], nhead=4).to(device)
        
        # 加載預訓練權重
        loaded_count = load_pretrained_weights(model, PRETRAINED_WEIGHTS, device)
        if loaded_count > 0 and fold == 0:
            print(f"✅ Fold 1: 成功載入 {loaded_count} 個層的預訓練權重")

        optimizer = optim.Adam(model.parameters(), lr=bp['learning_rate'], weight_decay=bp['weight_decay'])
        criterion = nn.CrossEntropyLoss(label_smoothing=bp['label_smoothing'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15)

        history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'val_f1': []}
        min_val_loss, epochs_no_improve = float('inf'), 0
        best_fold_acc, best_fold_f1 = 0, 0
        PATIENCE, EPOCHS = 30, 500

        for epoch in range(EPOCHS):
            model.train(); full_dataset.augment = True
            t_loss, t_correct, t_total = 0, 0, 0
            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad(); out = model(bx); loss = criterion(out, by); loss.backward(); optimizer.step()
                t_loss += loss.item(); t_correct += (out.argmax(1) == by).sum().item(); t_total += by.size(0)

            model.eval(); full_dataset.augment = False
            v_loss, v_correct, v_total = 0, 0, 0
            all_v_preds, all_v_trues = [], []
            with torch.no_grad():
                for vx, vy in val_loader:
                    vx, vy = vx.to(device), vy.to(device)
                    out = model(vx); v_loss += criterion(out, vy).item()
                    preds = out.argmax(1)
                    v_correct += (preds == vy).sum().item(); v_total += vy.size(0)
                    all_v_preds.extend(preds.cpu().numpy()); all_v_trues.extend(vy.cpu().numpy())

            avg_v_loss = v_loss/len(val_loader)
            current_v_acc = v_correct/v_total
            current_v_f1 = f1_score(all_v_trues, all_v_preds, average='macro', zero_division=0)

            history['train_loss'].append(t_loss/len(train_loader)); history['val_loss'].append(avg_v_loss)
            history['train_acc'].append(t_correct/t_total); history['val_acc'].append(current_v_acc)
            history['val_f1'].append(current_v_f1)
            scheduler.step(avg_v_loss)

            if avg_v_loss < min_val_loss:
                min_val_loss = avg_v_loss
                best_fold_acc = current_v_acc
                best_fold_f1 = current_v_f1
                safe_save_model(model.state_dict(), os.path.join(FOLD_DIR, "best_model.pth"))
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if (epoch + 1) % 20 == 0:
                print(f"Fold {fold+1} | Ep {epoch+1:03d} | Val Loss: {avg_v_loss:.4f} | Val Acc: {current_v_acc:.2%}")

            if epochs_no_improve >= PATIENCE: 
                print(f"🛑 Fold {fold+1} Early stopping at epoch {epoch+1}")
                break

        # 記錄該 Fold 的最終最佳成績
        all_fold_accuracies.append(best_fold_acc)
        all_fold_f1_scores.append(best_fold_f1)
        
        # 載入該 Fold 最好的權重來產出混淆矩陣
        model.load_state_dict(torch.load(os.path.join(FOLD_DIR, "best_model.pth")))
        model.eval()
        
        # 匯出 ONNX (每一折都匯出，方便後續集成或挑選)
        print(f"📦 Exporting Fold {fold+1} Model to ONNX...")
        dummy_input = torch.randn(1, TARGET_FRAMES, INPUT_DIM).to(device)
        onnx_path = os.path.join(FOLD_DIR, f"tsl_model_fold{fold+1}.onnx")
        torch.onnx.export(model, dummy_input, onnx_path, 
                        input_names=['input'], output_names=['output'], opset_version=17,
                        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})

        all_preds, all_trues = [], []
        with torch.no_grad():
            for vx, vy in val_loader:
                vx, vy = vx.to(device), vy.to(device)
                out = model(vx); all_preds.extend(out.argmax(1).cpu().numpy()); all_trues.extend(vy.cpu().numpy())
        
        # 產出詳細分類報告 (含 Precision, Recall, F1, Support)
        report = classification_report(all_trues, all_preds, target_names=class_names, zero_division=0)
        report_path = os.path.join(FOLD_DIR, "classification_report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"Fold {fold+1} Best Model Report\n")
            f.write("="*30 + "\n")
            f.write(report)
        
        print(f"📄 Fold {fold+1} 分類報告：\n{report}")
        print(f"✅ 報告已儲存至：{report_path}")

        # 產出綜合圖表 (摺線圖 + 每詞彙詳細指標)
        plot_comprehensive_report(history, all_trues, all_preds, class_names, FOLD_DIR, fold+1)

        plot_confusion_matrix(all_trues, all_preds, class_names, FOLD_DIR)
        print(f"✅ Fold {fold+1} Completed. Best Val Acc: {best_fold_acc:.2%}")

    # D. 最終總結
    avg_acc = np.mean(all_fold_accuracies)
    std_acc = np.std(all_fold_accuracies)
    
    print("\n" + "⭐" * 30)
    print(f"🏆 K-Fold Cross Validation Summary ({K_FOLDS} Folds)")
    for i, acc in enumerate(all_fold_accuracies):
        print(f"   Fold {i+1}: {acc:.2%}")
    print("-" * 30)
    print(f"   Average Accuracy: {avg_acc:.2%}")
    print(f"   Standard Deviation: {std_acc:.4f}")
    print("⭐" * 30)

    # 將總結存入 JSON
    summary = {
        "fold_accuracies": all_fold_accuracies,
        "fold_f1_scores": all_fold_f1_scores,
        "average_accuracy": avg_acc,
        "average_f1_score": np.mean(all_fold_f1_scores),
        "std_deviation": std_acc,
        "best_params": bp
    }
    with open(os.path.join(OUTPUT_DIR, "kfold_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)
    
    print(f"✨ All Tasks Completed! Results saved in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
