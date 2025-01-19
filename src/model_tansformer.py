import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

train_df = pd.read_csv("train_data.csv")
train_df.sort_values(by=["dteday", "hr"], inplace=True)
train_df.reset_index(drop=True, inplace=True)
print("train_data 行数:", len(train_df))

feature_cols = [
    "temp", "atemp", "hum", "windspeed",
    "season", "holiday", "workingday", "weathersit",
    "cnt"
]
train_features = train_df[feature_cols].values  # (T_train, 9)

scaler = MinMaxScaler()
train_features_scaled = scaler.fit_transform(train_features)

target_idx = feature_cols.index("cnt")
train_target_scaled = train_features_scaled[:, target_idx].reshape(-1, 1)

I = 96   # 过去 96 小时
O = 240   # 未来 96 小时 (可改为240)

def create_dataset(features, target, look_back=96, pred_len=96):
    """
    features: shape (N, num_features)
    target: shape   (N, 1)
    返回:
      X: shape (M, look_back, num_features)
      y: shape (M, pred_len)
    """
    X_list, y_list = [], []
    length = len(features)
    for i in range(length - look_back - pred_len + 1):
        X_list.append(features[i : i + look_back])
        y_list.append(target[i + look_back : i + look_back + pred_len])
    X_arr = np.array(X_list)
    y_arr = np.array(y_list).squeeze(-1)  # (M, pred_len)
    return X_arr, y_arr

X_all, y_all = create_dataset(
    train_features_scaled,
    train_target_scaled,
    look_back=I,
    pred_len=O
)
print("X_all:", X_all.shape)  # (M, 96, 9)
print("y_all:", y_all.shape)  # (M, 96)

# 简单划分 80%-20% 做训练/验证
train_size = int(0.8 * len(X_all))
X_train = X_all[:train_size]
y_train = y_all[:train_size]

X_val   = X_all[train_size:]
y_val   = y_all[train_size:]

print("训练样本数:", X_train.shape[0])
print("验证样本数:", X_val.shape[0])

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        """
        X: (N, I, num_features)
        y: (N, O)
        """
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        X_item = torch.FloatTensor(self.X[idx])  # (I, num_features)
        y_item = torch.FloatTensor(self.y[idx])  # (O,)
        return X_item, y_item

batch_size = 64
train_dataset = TimeSeriesDataset(X_train, y_train)
val_dataset   = TimeSeriesDataset(X_val,   y_val)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # pe shape: (max_len, d_model)
        # unsqueeze(1) -> (max_len, 1, d_model)  for batch_first=False usage
        self.register_buffer('pe', pe.unsqueeze(0))  # shape: (1, max_len, d_model)

    def forward(self, x):
        """
        x shape: (batch, seq_len, d_model)
        """
        seq_len = x.size(1)
        # self.pe[:, :seq_len] shape: (1, seq_len, d_model)
        x = x + self.pe[:, :seq_len, :]
        return x

class TransformerTimeSeries(nn.Module):
    def __init__(
            self,
            input_size=9,
            d_model=64,
            nhead=4,
            num_encoder_layers=2,
            num_decoder_layers=2,
            dim_feedforward=128,
            output_size=96,  # O
            max_len=500,
            batch_first=True
    ):
        super().__init__()

        self.d_model = d_model
        self.output_size = output_size

        # 1) 用线性把 input_size -> d_model
        self.input_fc = nn.Linear(input_size, d_model)
        # 2) 位置编码
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len)

        # 3) Transformer本体
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            batch_first=batch_first
        )

        # 4) Decoder端的输出投影: d_model -> 1 (因为我们只预测cnt)
        self.output_fc = nn.Linear(d_model, 1)

    def forward(self, src, tgt):
        """
        src: (batch, I, input_size)
        tgt: (batch, O, input_size) or (batch, O, d_model) ...

        但是如果我们只是给一串 zeros 让 Decoder decode O步，也可以:
        在外部先构造 shape (batch, O, d_model) 的 zero embed, or O-step "queries".
        """
        # 1) fc -> pos encode
        src_embed = self.input_fc(src)  # (batch, I, d_model)
        src_embed = self.pos_encoder(src_embed)

        tgt_embed = self.input_fc(tgt)  # (batch, O, d_model)
        tgt_embed = self.pos_encoder(tgt_embed)

        # 2) pass into transformer
        # out shape: (batch, O, d_model) if batch_first=True
        out = self.transformer(
            src=src_embed,
            tgt=tgt_embed
        )

        # 3) map d_model -> 1
        # out: (batch, O, 1)
        out = self.output_fc(out)
        # 返回 (batch, O)
        return out.squeeze(-1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

d_model = 64
nhead = 4
num_encoder_layers = 2
num_decoder_layers = 2
dim_feedforward = 128

model = TransformerTimeSeries(
    input_size=len(feature_cols),
    d_model=d_model,
    nhead=nhead,
    num_encoder_layers=num_encoder_layers,
    num_decoder_layers=num_decoder_layers,
    dim_feedforward=dim_feedforward,
    output_size=O,  # O=96
    max_len=500,    # 足够大即可
    batch_first=True
)
model.to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
epochs = 100  # 调试

def generate_tgt_zeros(batch_size, seq_len, input_dim):
    # 生成全 0 的 (batch_size, seq_len, input_dim)
    return torch.zeros(batch_size, seq_len, input_dim)

for epoch in range(epochs):
    model.train()
    train_mse_sum = 0.0
    train_mae_sum = 0.0
    total_train_samples = 0

    for X_batch, y_batch in train_loader:
        # X_batch: (batch, 96, 9)
        # y_batch: (batch, 96)
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        # 生成 decoder 输入 (batch, O, 9) 全0
        tgt_zero = generate_tgt_zeros(X_batch.size(0), O, X_batch.size(2)).to(device)

        optimizer.zero_grad()
        # 输出: (batch, O)
        pred = model(X_batch, tgt_zero)

        loss = criterion(pred, y_batch)
        loss.backward()
        optimizer.step()

        batch_size_ = X_batch.size(0)
        train_mse_sum += torch.mean((pred - y_batch)**2).item() * batch_size_
        train_mae_sum += torch.mean(torch.abs(pred - y_batch)).item() * batch_size_
        total_train_samples += batch_size_

    train_mse_epoch = train_mse_sum / total_train_samples
    train_mae_epoch = train_mae_sum / total_train_samples

    # 验证
    model.eval()
    val_mse_sum = 0.0
    val_mae_sum = 0.0
    total_val_samples = 0
    with torch.no_grad():
        for X_val_b, y_val_b in val_loader:
            X_val_b = X_val_b.to(device)
            y_val_b = y_val_b.to(device)

            tgt_zero_val = generate_tgt_zeros(X_val_b.size(0), O, X_val_b.size(2)).to(device)
            pred_val = model(X_val_b, tgt_zero_val)  # (batch, O)

            val_mse_sum += torch.mean((pred_val - y_val_b)**2).item() * X_val_b.size(0)
            val_mae_sum += torch.mean(torch.abs(pred_val - y_val_b)).item() * X_val_b.size(0)
            total_val_samples += X_val_b.size(0)

    val_mse_epoch = val_mse_sum / total_val_samples
    val_mae_epoch = val_mae_sum / total_val_samples

    print(
        f"[Epoch {epoch+1}/{epochs}] "
        f"Train MSE: {train_mse_epoch:.6f}, MAE: {train_mae_epoch:.6f} | "
        f"Val MSE: {val_mse_epoch:.6f}, MAE: {val_mae_epoch:.6f} | "
        f"Loss: {loss.item():.6f}"
    )

test_df = pd.read_csv("test_data.csv")
test_df.sort_values(by=["dteday", "hr"], inplace=True)
test_df.reset_index(drop=True, inplace=True)

test_features = test_df[feature_cols].values  # (T_test, 9)
test_features_scaled = scaler.transform(test_features)

test_target_scaled = test_features_scaled[:, target_idx].reshape(-1, 1)

def create_dataset(features, target, look_back, pred_len):
    X_list, y_list = [], []
    length = len(features)
    for i in range(length - look_back - pred_len + 1):
        X_list.append(features[i : i + look_back])
        y_list.append(target[i + look_back : i + look_back + pred_len])
    return np.array(X_list), np.array(y_list).squeeze(-1)

X_test_all, y_test_all = create_dataset(
    test_features_scaled, test_target_scaled,
    look_back=I,  # 96
    pred_len=O    # 96
)
print("X_test_all:", X_test_all.shape)  # (N_test, 96, 9)
print("y_test_all:", y_test_all.shape)  # (N_test, 96)

model.eval()
pred_list = []

with torch.no_grad():
    for i in range(len(X_test_all)):
        x_input = torch.FloatTensor(X_test_all[i]).unsqueeze(0).to(device)  # (1, 96, 9)

        # decoder输入全 0
        tgt_zeros = torch.zeros(1, O, x_input.size(2)).to(device)  # (1, 96, 9)

        pred = model(x_input, tgt_zeros)  # (1, 96)
        pred_list.append(pred.cpu().numpy().squeeze(0))  # (96,)

preds_test = np.array(pred_list)  # (N_test, 96)

def inverse_cnt_scaling(y_scaled_2d, scaler, target_col_idx, num_feats=9):
    """
    y_scaled_2d: (N, O)
    返回反归一化后的 shape (N, O)
    """
    N, seq_len = y_scaled_2d.shape
    dummy = np.zeros((N * seq_len, num_feats))
    dummy[:, target_col_idx] = y_scaled_2d.reshape(-1)
    inv_dummy = scaler.inverse_transform(dummy)
    cnt_values = inv_dummy[:, target_col_idx].reshape(N, seq_len)
    return cnt_values

preds_test_inv = inverse_cnt_scaling(preds_test, scaler, target_idx, len(feature_cols))
trues_test_inv = inverse_cnt_scaling(y_test_all, scaler, target_idx, len(feature_cols))

# 全局 MSE/MAE
flatten_preds = preds_test_inv.reshape(-1)
flatten_trues = trues_test_inv.reshape(-1)
test_mse = mean_squared_error(flatten_trues, flatten_preds)
test_mae = mean_absolute_error(flatten_trues, flatten_preds)
print(f"Test MSE: {test_mse:.4f}, Test MAE: {test_mae:.4f}")

sample_idx = 0  # 任选一个
pred_seq = preds_test_inv[sample_idx]
true_seq = trues_test_inv[sample_idx]

plt.figure(figsize=(10,5))
plt.plot(range(O), true_seq, label='True', marker='o')
plt.plot(range(O), pred_seq, label='Predicted', marker='x')
plt.title(f"Test Sample {sample_idx} - Future {O} steps")
plt.xlabel("Hour Ahead")
plt.ylabel("cnt")
plt.legend()
plt.show()
