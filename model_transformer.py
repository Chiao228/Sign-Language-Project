import torch
import torch.nn as nn

# --- 4. 模型架構 ---
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
        
        # Positional Encoding (針對時間序列的可學習位置編碼)
        self.pos_encoder = nn.Parameter(torch.randn(1, 30, hidden_dim))
        
        # [修改點] 更換為 Transformer
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=8, # 若 hidden_dim=128或256，皆可整除8
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
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
        x = self.transformer(x)
        
        # 結合 Mean 與 Max Pooling
        avg_pool = x.mean(dim=1)
        max_pool, _ = x.max(dim=1)
        return self.fc(avg_pool + max_pool)
