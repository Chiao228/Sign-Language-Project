"""
analyze_landmark_importance.py
==============================
分析 TSL 手語辨識中每個關鍵點群組的重要性。
提供三種分析方法：
  1. 統計變異分析 — 快速找出無用的維度
  2. 消融實驗 (Ablation Study) — 遮蔽某群特徵，觀察準確率變化
  3. 排列重要性 (Permutation Importance) — 打亂某群特徵，觀察準確率變化

使用方式：
  python analyze_landmark_importance.py
"""

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
# ==========================================
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

DATA_PATH = "processed_npy_66"
MODEL_DIR = "train_V9_GRU_66(earlyStop)"
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pth")
OUTPUT_DIR = "landmark_analysis_66"
os.makedirs(OUTPUT_DIR, exist_ok=True)

INPUT_DIM = 66
TARGET_FRAMES = 30

# ==========================================
# 1. 特徵群組定義 (22點全拆解：精簡版手勢點位)
# ==========================================
def generate_all_22_landmarks():
    groups = {}
    
    # 重新定義保留的接點名稱 (共 11 個點)
    hand_names = [
        "手腕 Wrist(0)", 
        "拇指關節 IP(3)", "拇指尖 Tip(4)",
        "食指關節 DIP(7)", "食指尖 Tip(8)",
        "中指關節 DIP(11)", "中指尖 Tip(12)",
        "無名指關節 DIP(15)", "無名指尖 Tip(16)",
        "小指關節 DIP(19)", "小指尖 Tip(20)"
    ]

    # 1. 左手 11 點 (0~32維)
    for i, name in enumerate(hand_names):
        groups[f"左手 LH {name}"] = [i*3, i*3+1, i*3+2]

    # 2. 右手 11 點 (33~65維)
    for i, name in enumerate(hand_names):
        base_idx = 33 + i*3
        groups[f"右手 RH {name}"] = [base_idx, base_idx+1, base_idx+2]

    return groups

# 將所有 22 個點直接作為變異分析與消融實驗的對象
ALL_22_LANDMARKS = generate_all_22_landmarks()

FEATURE_GROUPS_COORD = ALL_22_LANDMARKS
ABLATION_GROUPS = ALL_22_LANDMARKS

# ==========================================
# 2. 模型定義 (必須和 train_tsl.py 一致)
# ==========================================
class CNNGRUTSL(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim, num_layers, dropout=0.4):
        super(CNNGRUTSL, self).__init__()
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
        self.gru = nn.GRU(
            input_size=hidden_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128), 
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.cnn(x.transpose(1, 2)).transpose(1, 2)
        x, _ = self.gru(x)
        avg_pool = x.mean(dim=1)
        max_pool, _ = x.max(dim=1)
        return self.fc(avg_pool + max_pool)

# ==========================================
# 3. 資料讀取
# ==========================================
class SimpleDataset(Dataset):
    def __init__(self, data_dir, model_dir):
        self.samples, self.labels = [], []
        
        # 1. 強制讀取訓練時存下的標籤映射以確保索引完全正確
        label_map_path = os.path.join(model_dir, "label_map.json")
        if not os.path.exists(label_map_path):
            raise FileNotFoundError(f"❌ 找不到標籤映射檔: {label_map_path} \n這會導致分類索引錯誤，請確認模型目錄。")
            
        with open(label_map_path, "r", encoding="utf-8") as f:
            # JSON 的 key 永遠是字串，需轉回 int: {"0": "謝謝"} -> {0: "謝謝"}
            temp_map = json.load(f)
            self.label_map = {int(k): v for k, v in temp_map.items()}
            # 建立反向映射供讀取檔案時使用
            name_to_idx = {v: k for k, v in self.label_map.items()}

        all_words = sorted([d for d in os.listdir(data_dir)
                           if os.path.isdir(os.path.join(data_dir, d)) and not d.startswith('.')])
        
        for word in all_words:
            if word not in name_to_idx:
                print(f"⚠️ 警告: 資料夾 '{word}' 不在訓練時的標籤清單中，將跳過。")
                continue
                
            word_idx = name_to_idx[word]
            word_path = os.path.join(data_dir, word)
            
            # 2. 檔案進行排序，增加資料切分的一致性
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
    """評估模型在給定 DataLoader 上的準確率。"""
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
    """分析每個維度的變異數，找出可能無用的維度。"""
    print("\n" + "=" * 60)
    print("📊 分析 1：統計變異分析 (Statistical Variance)")
    print("=" * 60)

    # 收集所有驗證集數據
    all_data = []
    for idx in val_idx:
        data, _ = dataset[idx]
        all_data.append(data.numpy())
    all_data = np.array(all_data)  # (N, 30, 138)

    # 計算每個維度在所有樣本+所有幀上的變異數
    reshaped = all_data.reshape(-1, INPUT_DIM)  # (N*30, 138)
    variances = np.var(reshaped, axis=0)  # (138,)

    # 按群組統計
    print(f"\n{'群組名稱':<35} {'平均變異數':>12} {'最大變異數':>12} {'最小變異數':>12}")
    print("-" * 75)

    group_variances = {}
    for group_name, indices in FEATURE_GROUPS_COORD.items():
        group_var = variances[indices]
        avg_var = np.mean(group_var)
        max_var = np.max(group_var)
        min_var = np.min(group_var)
        group_variances[group_name] = avg_var
        print(f"{group_name:<35} {avg_var:>12.6f} {max_var:>12.6f} {min_var:>12.6f}")

    # 找出近乎恆為 0 的維度
    near_zero = np.where(variances < 1e-8)[0]
    if len(near_zero) > 0:
        print(f"\n⚠️ 發現 {len(near_zero)} 個幾乎恆為零的維度（變異數 < 1e-8）：")
        print(f"  索引：{near_zero.tolist()}")
    else:
        print(f"\n✅ 所有維度都有足夠的變異量")

    # 視覺化 (加大圖片尺寸以容納 46 個群組)
    fig, axes = plt.subplots(2, 1, figsize=(16, 16))

    # 上圖：所有維度的變異數
    axes[0].bar(range(INPUT_DIM), variances, color='#3b82f6', alpha=0.7)
    axes[0].set_title(f"每個維度的變異數 ({INPUT_DIM} 維座標)", fontsize=14)
    axes[0].set_xlabel("特徵維度索引")
    axes[0].set_ylabel("變異數")
    axes[0].axvline(x=33, color='red', linestyle='--', alpha=0.5, label='左右手分界')
    axes[0].legend()

    # 下圖：群組平均變異數
    group_names = list(group_variances.keys())
    group_vals = list(group_variances.values())
    
    # 動態產生顏色，避免長度不匹配
    colors = ['#3b82f6'] * len(group_names)
    
    axes[1].barh(group_names, group_vals, color=colors, alpha=0.8)
    axes[1].set_title("各點平均變異數", fontsize=14)
    axes[1].set_xlabel("平均變異數")
    axes[1].invert_yaxis()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "variance_analysis.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n📊 變異數分析圖已儲存至 {OUTPUT_DIR}/variance_analysis.png")

    return variances


def run_ablation_study(model, dataset, val_idx, device, baseline_acc):
    """消融實驗：逐一遮蔽每組特徵，觀察準確率變化。"""
    print("\n" + "=" * 60)
    print("🔬 分析 2：消融實驗 (Ablation Study)")
    print("=" * 60)
    print(f"📌 基線準確率 (Baseline): {baseline_acc:.2%}\n")

    results = {}

    for group_name, indices in ABLATION_GROUPS.items():
        # 建立一個遮蔽版的資料集
        masked_data = []
        masked_labels = []
        for idx in val_idx:
            data, label = dataset[idx]
            data_np = data.numpy().copy()
            data_np[:, indices] = 0.0  # 將該群組特徵全部歸零
            masked_data.append(torch.tensor(data_np, dtype=torch.float32))
            masked_labels.append(label)

        # 建立臨時 DataLoader
        masked_batch = list(zip(masked_data, masked_labels))
        masked_loader = DataLoader(masked_batch, batch_size=32, shuffle=False, collate_fn=collate_fn)

        # 評估
        masked_acc = evaluate_model(model, masked_loader, device)
        drop = baseline_acc - masked_acc
        results[group_name] = {
            "masked_acc": masked_acc,
            "drop": drop,
            "drop_pct": (drop / baseline_acc * 100) if baseline_acc > 0 else 0
        }

        emoji = "🔴" if drop > 0.05 else "🟡" if drop > 0.01 else "🟢"
        print(f"  {emoji} {group_name:<35} | 遮蔽後準確率: {masked_acc:.2%} | 下降: {drop:+.2%}")

    # 視覺化 (加大高度以容納 46 個群組)
    fig, ax = plt.subplots(figsize=(12, 14))

    names = list(results.keys())
    drops = [results[n]["drop_pct"] for n in names]
    colors = ['#ef4444' if d > 5 else '#f59e0b' if d > 1 else '#10b981' for d in drops]

    bars = ax.barh(names, drops, color=colors, alpha=0.85, edgecolor='white', linewidth=0.5)
    ax.set_title("消融實驗：移除單一特徵點後的準確率下降幅度 (%)", fontsize=14, fontweight='bold')
    ax.set_xlabel("準確率下降 (%)")
    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.invert_yaxis()

    # 加上數值標籤
    for bar, drop_val in zip(bars, drops):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f'{drop_val:.1f}%', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "ablation_study.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n📊 消融實驗圖已儲存至 {OUTPUT_DIR}/ablation_study.png")

    return results


def run_permutation_importance(model, dataset, val_idx, device, baseline_acc, n_repeats=5):
    """排列重要性分析：隨機打亂每組特徵，觀察準確率變化。"""
    print("\n" + "=" * 60)
    print("🎲 分析 3：排列重要性 (Permutation Importance)")
    print("=" * 60)
    print(f"📌 基線準確率 (Baseline): {baseline_acc:.2%}")
    print(f"📌 重複次數: {n_repeats}\n")

    # 先收集所有驗證資料
    all_data, all_labels = [], []
    for idx in val_idx:
        data, label = dataset[idx]
        all_data.append(data.numpy())
        all_labels.append(label.item())
    all_data = np.array(all_data)
    all_labels = np.array(all_labels)

    results = {}

    for group_name, indices in ABLATION_GROUPS.items():
        drops = []
        for _ in range(n_repeats):
            perturbed_data = all_data.copy()
            # 在樣本維度上打亂該群組的特徵
            for dim_idx in indices:
                perm = np.random.permutation(len(perturbed_data))
                perturbed_data[:, :, dim_idx] = perturbed_data[perm, :, dim_idx]

            # 評估
            batch_data = [torch.tensor(d, dtype=torch.float32) for d in perturbed_data]
            batch_labels = [torch.tensor(l, dtype=torch.long) for l in all_labels]
            batch = list(zip(batch_data, batch_labels))
            loader = DataLoader(batch, batch_size=32, shuffle=False, collate_fn=collate_fn)

            acc = evaluate_model(model, loader, device)
            drops.append(baseline_acc - acc)

        mean_drop = np.mean(drops)
        std_drop = np.std(drops)
        results[group_name] = {"mean_drop": mean_drop, "std_drop": std_drop}

        emoji = "🔴" if mean_drop > 0.05 else "🟡" if mean_drop > 0.01 else "🟢"
        print(f"  {emoji} {group_name:<35} | 平均下降: {mean_drop:+.2%} ± {std_drop:.2%}")

    # 視覺化 (加大高度以容納 46 個群組)
    fig, ax = plt.subplots(figsize=(12, 14))

    names = list(results.keys())
    means = [results[n]["mean_drop"] * 100 for n in names]
    stds = [results[n]["std_drop"] * 100 for n in names]
    colors = ['#ef4444' if m > 5 else '#f59e0b' if m > 1 else '#10b981' for m in means]

    ax.barh(names, means, xerr=stds, color=colors, alpha=0.85, capsize=3)
    ax.set_title("排列重要性：打亂單一特徵點後的準確率下降 (%)", fontsize=14, fontweight='bold')
    ax.set_xlabel("平均準確率下降 (%)")
    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "permutation_importance.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n📊 排列重要性圖已儲存至 {OUTPUT_DIR}/permutation_importance.png")

    return results


# ==========================================
# 5. 主程式
# ==========================================
def main():
    print("=" * 60)
    print("🔍 TSL 手語特徵點重要性分析 (22 點純手部精簡版)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📱 裝置: {device}")

    # 載入資料
    if not os.path.exists(DATA_PATH):
        print(f"❌ 找不到資料路徑: {DATA_PATH}")
        print(f"   請先執行 dynamic_energy_crop.py 產生處理後的資料")
        return

    dataset = SimpleDataset(DATA_PATH, MODEL_DIR)
    num_classes = len(dataset.label_map)
    class_names = [dataset.label_map[i] for i in range(num_classes)]
    print(f"📂 資料集: {len(dataset)} 筆, {num_classes} 類")
    print(f"📋 類別: {class_names}")

    # 切分驗證集
    train_idx, val_idx = train_test_split(
        np.arange(len(dataset.labels)),
        test_size=0.3,
        stratify=dataset.labels,
        random_state=42
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=32, shuffle=False, collate_fn=collate_fn
    )

    # 載入超參數
    params_file = os.path.join(MODEL_DIR, "best_params.json")
    if os.path.exists(params_file):
        with open(params_file, "r", encoding="utf-8") as f:
            bp = json.load(f)
        print(f"✅ 讀取超參數: {params_file}")
    else:
        print("⚠️ 找不到 best_params.json，使用預設超參數")
        bp = {"hidden_dim": 128, "num_layers": 2, "dropout": 0.5}

    # 載入模型
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 找不到模型權重: {MODEL_PATH}")
        print(f"   請先訓練模型 (python train_tsl.py)")
        return

    model = CNNGRUTSL(
        INPUT_DIM, num_classes,
        bp['hidden_dim'], bp['num_layers'],
        bp.get('dropout', 0.4)
    ).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print(f"✅ 模型載入成功: {MODEL_PATH}")

    # 計算基線準確率
    baseline_acc = evaluate_model(model, val_loader, device)
    print(f"\n🎯 基線驗證準確率 (Baseline): {baseline_acc:.2%}")

    # --- 執行三項分析 ---
    # 1. 統計變異分析
    variances = analyze_variance(dataset, val_idx)

    # 2. 消融實驗
    ablation_results = run_ablation_study(model, dataset, val_idx, device, baseline_acc)

    # 3. 排列重要性
    perm_results = run_permutation_importance(model, dataset, val_idx, device, baseline_acc, n_repeats=5)

    # --- 綜合報告 ---
    print("\n" + "=" * 60)
    print("📋 綜合分析報告")
    print("=" * 60)

    report_lines = [
        "# TSL 手語特徵點重要性分析報告 (22 點純手部精簡版)\n",
        f"- 基線準確率: {baseline_acc:.2%}",
        f"- 驗證集大小: {len(val_idx)} 筆",
        f"- 類別數: {num_classes}\n",
        "## 消融實驗結果\n",
        "| 特徵點 | 遮蔽後準確率 | 下降幅度 | 重要性 |",
        "|---|---|---|---|",
    ]

    for group_name, res in sorted(ablation_results.items(), key=lambda x: x[1]["drop"], reverse=True):
        importance = "🔴 高" if res["drop"] > 0.05 else "🟡 中" if res["drop"] > 0.01 else "🟢 低"
        report_lines.append(
            f"| {group_name} | {res['masked_acc']:.2%} | {res['drop']:+.2%} | {importance} |"
        )

    report_lines.extend([
        "\n## 排列重要性結果\n",
        "| 特徵點 | 平均下降 | 標準差 | 重要性 |",
        "|---|---|---|---|",
    ])

    for group_name, res in sorted(perm_results.items(), key=lambda x: x[1]["mean_drop"], reverse=True):
        importance = "🔴 高" if res["mean_drop"] > 0.05 else "🟡 中" if res["mean_drop"] > 0.01 else "🟢 低"
        report_lines.append(
            f"| {group_name} | {res['mean_drop']:+.2%} | ±{res['std_drop']:.2%} | {importance} |"
        )

    report_lines.extend([
        "\n## 建議\n",
        "- 🔴 **高重要性**特徵點：移除後準確率下降 > 5%，為核心特徵，必須保留",
        "- 🟡 **中重要性**特徵點：移除後準確率下降 1-5%，建議保留",
        "- 🟢 **低重要性**特徵點：移除後準確率幾乎不變，可考慮在未來資料前處理時剔除以減少運算負擔",
    ])

    report_path = os.path.join(OUTPUT_DIR, "importance_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    # 存成 JSON 供後續使用
    summary = {
        "baseline_acc": baseline_acc,
        "ablation": {k: v for k, v in ablation_results.items()},
        "permutation": {k: {"mean_drop": v["mean_drop"], "std_drop": v["std_drop"]} for k, v in perm_results.items()}
    }
    with open(os.path.join(OUTPUT_DIR, "importance_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    print(f"\n✅ 報告已儲存至 {report_path}")
    print(f"✅ JSON 摘要已儲存至 {OUTPUT_DIR}/importance_summary.json")
    print(f"✅ 圖表已儲存至 {OUTPUT_DIR}/")
    print("\n✨ 分析完成！請查看上述檔案以決定要保留或移除哪些特徵點。")


if __name__ == "__main__":
    main()