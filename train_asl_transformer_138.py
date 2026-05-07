import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
# from models.asl_cnn_transformer import ASL_CNN_Transformer

# ==========================================
# 0. Model Architecture (Consistent with train_transformer.py style)
# ==========================================
class ASL_CNN_Transformer(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_layers, num_classes, dropout=0.3):
        super(ASL_CNN_Transformer, self).__init__()
        # CNN block
        self.cnn = nn.Sequential(
            nn.Conv1d(input_size, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(128, d_model, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Positional Encoding (Target frames = 30)
        self.pos_encoder = nn.Parameter(torch.randn(1, 30, d_model))
        
        # Transformer
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # Final FC
        self.fc = nn.Sequential(
            nn.Linear(d_model, 128), 
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x: (Batch, Frames, Dim) -> (Batch, Dim, Frames)
        x = self.cnn(x.transpose(1, 2)).transpose(1, 2)
        
        # Add Positional Encoding
        if x.size(1) <= self.pos_encoder.size(1):
            x = x + self.pos_encoder[:, :x.size(1), :]
        
        x = self.transformer(x)
        
        # Combine Mean & Max Pooling
        avg_pool = x.mean(dim=1)
        max_pool, _ = x.max(dim=1)
        return self.fc(avg_pool + max_pool)
# ==========================================
# 1. Hyperparameters & Settings
# ==========================================
OUTPUT_DIR = "train_V1_transformer_138"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATA_PATH = "sliding_window_138"
INPUT_SIZE = 138                            
D_MODEL = 128
NHEAD = 4
NUM_LAYERS = 2
DROPOUT = 0.3
BATCH_SIZE = 16        # ASL 資料較多，可以設為 16 或 32
LEARNING_RATE = 0.001
EPOCHS = 100

# MPS / CUDA / CPU
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("🚀 使用 Apple Silicon MPS 加速訓練！")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("🚀 使用 CUDA 加速訓練！")
else:
    device = torch.device("cpu")
    print("⚠️ 未偵測到 GPU，使用 CPU 訓練。")

# ==========================================
# 2. Data Preparation
# ==========================================
def scan_classes(data_dir):
    classes = [d for d in sorted(os.listdir(data_dir)) 
               if os.path.isdir(os.path.join(data_dir, d)) and any(f.endswith('.npy') for f in os.listdir(os.path.join(data_dir, d)))]
    print(f"📊 總共偵測到 {len(classes)} 個類別於 {os.path.basename(data_dir)} 中。")
    return classes

def prepare_dataset(data_dir, classes):
    train_data = [] 
    val_data = []
    
    label_map = {word: idx for idx, word in enumerate(classes)}
    
    for word in classes:
        label = label_map[word]
        word_dir = os.path.join(data_dir, word)
        files = sorted([os.path.join(word_dir, f) for f in os.listdir(word_dir) if f.endswith('.npy')])
        
        # 固定種子打亂確保每次 Train/Val 切分一致
        random.seed(42)  
        random.shuffle(files)
        # 80% Train, 20% Val
        split_idx = int(len(files) * 0.8)
        
        if len(files) < 2:
            train_data.extend([(f, label) for f in files])
        else:
            for f in files[:split_idx]: train_data.append((f, label))
            for f in files[split_idx:]: val_data.append((f, label))
            
    print(f"📦 Train: {len(train_data)} 筆, Val: {len(val_data)} 筆")
    return train_data, val_data, label_map

# ==========================================
# 3. Custom Dataset
# ==========================================
class FixedASLDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list
        
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        file_path, label = self.data_list[idx]
        feature = np.load(file_path).astype(np.float32) # (30, 138)
        feature_tensor = torch.tensor(feature)
        return feature_tensor, label

# ==========================================
# 4. Training Loop
# ==========================================
def train():
    if not os.path.exists(DATA_PATH):
        print(f"❌ 找不到訓練集目錄: {DATA_PATH}")
        return
        
    active_classes = scan_classes(DATA_PATH)
    num_classes = len(active_classes)
    
    train_data, val_data, label_map = prepare_dataset(DATA_PATH, active_classes)
    
    if len(train_data) == 0:
        print("❌ 訓練集是空的，無法進行訓練。")
        return
        
    train_dataset = FixedASLDataset(train_data)
    val_dataset = FixedASLDataset(val_data)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False) if len(val_data) > 0 else []
    
    # 建立模型
    model = ASL_CNN_Transformer(
        input_size=INPUT_SIZE, 
        d_model=D_MODEL, 
        nhead=NHEAD, 
        num_layers=NUM_LAYERS, 
        num_classes=num_classes, 
        dropout=DROPOUT
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'val_f1': []}
    
    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_val_f1 = 0.0
    best_epoch = 0
    
    # 加上超參數後綴以區分不同測試
    suffix = f"BS{BATCH_SIZE}_LR{LEARNING_RATE}"
    save_path = os.path.join(OUTPUT_DIR, f"transformer_138_{suffix}_best.pth")
    csv_log_path = os.path.join(OUTPUT_DIR, f"training_log_{suffix}.csv")
    
    # 初始化 CSV 檔案
    with open(csv_log_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Train Loss', 'Train Acc', 'Val Loss', 'Val Acc', 'Val F1'])
    
    print(f"\n🚀 模型啟動 (ASL 138D Transformer) | 類別數: {num_classes} | Epochs: {EPOCHS}")
    for epoch in range(EPOCHS):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        
        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * features.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
        train_loss /= train_total
        train_acc = train_correct / train_total
        
        # Validation
        val_loss, val_correct, val_total = 0.0, 0, 0
        val_acc = 0.0
        val_f1 = 0.0
        
        all_val_labels = []
        all_val_preds = []
        
        if len(val_loader) > 0:
            model.eval()
            with torch.no_grad():
                for features, labels in val_loader:
                    features = features.to(device)
                    labels = labels.to(device)
                    
                    outputs = model(features)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item() * features.size(0)
                    _, predicted = torch.max(outputs, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
                    
                    all_val_labels.extend(labels.cpu().numpy())
                    all_val_preds.extend(predicted.cpu().numpy())
                    
            val_loss /= val_total if val_total > 0 else 1
            val_acc = val_correct / val_total if val_total > 0 else 0
            val_f1 = f1_score(all_val_labels, all_val_preds, average='macro', zero_division=0)
            
            # 使用 Acc 與 Loss 來判斷是否儲存 Best Model
            if val_acc > best_val_acc or (val_acc == best_val_acc and val_loss < best_val_loss):
                best_val_acc = val_acc
                best_val_loss = min(val_loss, best_val_loss)
                best_val_f1 = val_f1
                best_epoch = epoch + 1
                torch.save(model.state_dict(), save_path)
                
                # 儲存最佳模型的表現報告與混淆矩陣
                report_path = os.path.join(OUTPUT_DIR, f"report_{suffix}.txt")
                with open(report_path, "w", encoding='utf-8') as rf:
                    rf.write(f"Best Epoch: {best_epoch}\n")
                    rf.write(f"Best Validation Accuracy: {best_val_acc:.4f}\n")
                    rf.write(f"Best Validation Loss: {best_val_loss:.4f}\n")
                    rf.write(f"Best Validation F1 (Macro): {best_val_f1:.4f}\n\n")
                    
                    # 生成分類報告
                    inverse_label = {v: k for k, v in label_map.items()}
                    labels_present = np.unique(all_val_labels)
                    target_names = [inverse_label.get(i, str(i)) for i in labels_present]
                    rf.write(classification_report(all_val_labels, all_val_preds, target_names=target_names, labels=labels_present, zero_division=0))
                    
                cm = confusion_matrix(all_val_labels, all_val_preds)
                cm_path = os.path.join(OUTPUT_DIR, f"cm_{suffix}.npy")
                np.save(cm_path, cm)
        else:
            val_loss = train_loss
            val_acc = train_acc
            torch.save(model.state_dict(), save_path)
            
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        # 寫入 CSV
        with open(csv_log_path, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, train_loss, train_acc, val_loss, val_acc, val_f1])
            
        print(f"Epoch [{epoch+1:03d}/{EPOCHS}] "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
                
    # 繪製圖表
    plot_path = os.path.join(OUTPUT_DIR, f"history_transformer_138_asl_{suffix}.png")
    plt.figure(figsize=(18, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    if len(val_loader) > 0:
        plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss Curve (ASL 138D Transformer)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    if len(val_loader) > 0:
        plt.plot(history['val_acc'], label='Val Acc')
    plt.title('Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    if len(val_loader) > 0:
        plt.plot(history['val_f1'], label='Val F1 Score', color='green')
    plt.title('Validation F1-Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"\n📈 訓練完成，損耗圖已儲存為 {os.path.basename(plot_path)}")
    print(f"💾 最佳權重儲存為 {os.path.basename(save_path)}")

if __name__ == '__main__':
    train()
