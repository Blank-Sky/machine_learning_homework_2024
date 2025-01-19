import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ---- 读取 train_data ----
train_df = pd.read_csv("train_data.csv")
train_df.sort_values(by=["dteday", "hr"], inplace=True)
train_df.reset_index(drop=True, inplace=True)

feature_cols = [
    "temp", "atemp", "hum", "windspeed",
    "season", "holiday", "workingday", "weathersit",
    "cnt"
]
train_features = train_df[feature_cols].values
scaler = MinMaxScaler()
train_features_scaled = scaler.fit_transform(train_features)

target_idx = feature_cols.index("cnt")
train_target_scaled = train_features_scaled[:, target_idx].reshape(-1, 1)

# ---- 滑动窗口函数 ----
def create_dataset(features, target, look_back=96, pred_len=96):
    X_list, y_list = [], []
    length = len(features)
    for i in range(length - look_back - pred_len + 1):
        X_list.append(features[i : i+look_back])
        y_list.append(target[i+look_back : i+look_back+pred_len])
    X_arr = np.array(X_list)                # (N, I, num_features)
    y_arr = np.array(y_list).squeeze(-1)    # (N, O)
    return X_arr, y_arr

# ---- 设置 I=96, O=96 (或者240) ----
I = 96
O = 240

X_all, y_all = create_dataset(train_features_scaled, train_target_scaled, I, O)
train_size = int(0.8 * len(X_all))

X_train = X_all[:train_size]
y_train = y_all[:train_size]
X_val   = X_all[train_size:]
y_val   = y_all[train_size:]

print("X_train:", X_train.shape, "y_train:", y_train.shape)
print("X_val:",   X_val.shape,   "y_val:",   y_val.shape)

batch_size = 64

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return torch.FloatTensor(self.X[idx]), torch.FloatTensor(self.y[idx])

train_dataset = TimeSeriesDataset(X_train, y_train)
val_dataset   = TimeSeriesDataset(X_val,   y_val)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

class MultiScaleCNN(nn.Module):
    """
    并行的多尺度1D卷积模块
    """
    def __init__(
            self,
            in_channels,   # = num_features
            out_channels,  # 每个分支的通道数
            num_scales=3,  # 分支数
            kernel_size=3,
            dilations=[1, 2, 4]
    ):
        super().__init__()
        assert num_scales == len(dilations), "num_scales和dilations长度要匹配"

        # 分支列表
        self.convs = nn.ModuleList()
        for d in dilations:
            conv = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                dilation=d,
                padding=(kernel_size-1)*d // 2
            )
            self.convs.append(conv)

        self.relu = nn.ReLU()

    def forward(self, x):
        # (B, T, C_in) -> (B, C_in, T)
        x = x.permute(0, 2, 1)

        outs = []
        for conv in self.convs:
            y = conv(x)  # (B, out_channels, T)
            y = self.relu(y)
            outs.append(y)

        # 拼接通道: (B, out_channels * num_scales, T)
        out = torch.cat(outs, dim=1)

        # 再转回 (B, T, channels)
        out = out.permute(0, 2, 1)
        return out

class LSTMEncoder(nn.Module):
    """
    一层或多层LSTM，将输入序列编码成 (batch, T, hidden_size)
    """
    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
    def forward(self, x):
        # x: (batch, T, input_size)
        out, (h_n, c_n) = self.lstm(x)
        # out: (batch, T, hidden_size)
        return out

class Attention(nn.Module):
    """
    简易单头注意力: 学习一个查询向量q，对每个时间步做 q·h_t 的打分 -> softmax -> 加权和
    """
    def __init__(self, hidden_size):
        super().__init__()
        # 可学习的query
        self.query = nn.Parameter(torch.randn(hidden_size))  # (hidden_size,)

    def forward(self, encoder_outputs):
        """
        encoder_outputs: (batch, T, hidden_size)
        返回: (batch, hidden_size), (batch, T)  # 最终上下文, 注意力分布
        """
        batch_size, T, H = encoder_outputs.shape
        # (batch, T, hidden_size) dot (hidden_size,) -> (batch, T)
        # 需要先扩展 query -> (batch, hidden_size)
        query_b = self.query.unsqueeze(0).expand(batch_size, H)  # (batch, hidden_size)

        # 点积: sum over hidden_size
        scores = torch.sum(encoder_outputs * query_b.unsqueeze(1), dim=2)  # (batch, T)

        # softmax
        alpha = torch.softmax(scores, dim=1)  # (batch, T)

        # 加权和
        context = torch.sum(encoder_outputs * alpha.unsqueeze(-1), dim=1)  # (batch, hidden_size)
        return context, alpha

class MSCLANet(nn.Module):
    """
    Multi-Scale CNN + LSTM + Attention Network
    """
    def __init__(
            self,
            num_features=9,
            cnn_out_channels=16,
            dilations=[1, 2, 4],
            lstm_hidden=64,
            lstm_layers=1,
            output_size=96
    ):
        super().__init__()
        self.num_scales = len(dilations)

        # 1) 多尺度卷积
        # 每个分支输出通道=cnn_out_channels，共有len(dilations)个分支 -> total_cnn_out = cnn_out_channels * num_scales
        self.multi_scale_cnn = MultiScaleCNN(
            in_channels=num_features,
            out_channels=cnn_out_channels,
            num_scales=self.num_scales,
            dilations=dilations
        )

        # 2) LSTM
        self.lstm_hidden = lstm_hidden
        self.lstm_encoder = LSTMEncoder(
            input_size=cnn_out_channels * self.num_scales,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers
        )

        # 3) Attention
        self.attn = Attention(lstm_hidden)

        # 4) 输出层: hidden_size -> O
        self.fc_out = nn.Linear(lstm_hidden, output_size)

    def forward(self, x):
        """
        x: (batch, I, num_features)
        """
        # 1) 多尺度卷积
        cnn_out = self.multi_scale_cnn(x)
        # shape: (batch, I, cnn_out_channels * num_scales)

        # 2) LSTM
        lstm_out = self.lstm_encoder(cnn_out)
        # shape: (batch, I, lstm_hidden)

        # 3) Attention
        context, alpha = self.attn(lstm_out)
        # context: (batch, lstm_hidden)

        # 4) 输出 O步
        preds = self.fc_out(context)  # (batch, O)
        return preds

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MSCLANet(
    num_features=len(feature_cols),
    cnn_out_channels=16,
    dilations=[1,2,4],
    lstm_hidden=64,
    lstm_layers=1,
    output_size=O  # 96
)
model.to(device)

class WeightedMSELoss(nn.Module):
    """
    对真实值 target > high_thresh 的位置使用更大的权重，放大峰值处的误差。
    """
    def __init__(self, high_thresh=0.7, weight_high=2.0):
        super().__init__()
        self.high_thresh = high_thresh
        self.weight_high = weight_high

    def forward(self, pred, target):
        # pred, target: (batch, O)
        # 1) 找到 target > high_thresh 的mask
        mask_high = (target > self.high_thresh).float()
        # 2) 基础weight=1，对高值区域的weight=weight_high
        #    weight = 1 + (weight_high-1)*mask
        weight = torch.ones_like(target) + (self.weight_high - 1.0) * mask_high
        # 3) MSE
        mse = (pred - target) ** 2
        weighted_mse = weight * mse
        return torch.mean(weighted_mse)

criterion = WeightedMSELoss(high_thresh=0.7, weight_high=2.0)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

epochs = 30
for epoch in range(epochs):
    model.train()
    train_mse_sum = 0
    train_mae_sum = 0
    total_train_samples = 0

    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)  # (batch, I, 9)
        y_batch = y_batch.to(device)  # (batch, O)

        optimizer.zero_grad()
        preds = model(X_batch)  # (batch, O)
        loss = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()

        batch_size_ = X_batch.size(0)
        train_mse_sum += torch.mean((preds - y_batch)**2).item() * batch_size_
        train_mae_sum += torch.mean(torch.abs(preds - y_batch)).item() * batch_size_
        total_train_samples += batch_size_

    train_mse_epoch = train_mse_sum / total_train_samples
    train_mae_epoch = train_mae_sum / total_train_samples

    # 验证
    model.eval()
    val_mse_sum = 0
    val_mae_sum = 0
    total_val_samples = 0
    with torch.no_grad():
        for X_val_b, y_val_b in val_loader:
            X_val_b = X_val_b.to(device)
            y_val_b = y_val_b.to(device)

            val_preds = model(X_val_b)
            val_mse_sum += torch.mean((val_preds - y_val_b)**2).item() * X_val_b.size(0)
            val_mae_sum += torch.mean(torch.abs(val_preds - y_val_b)).item() * X_val_b.size(0)
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

test_features = test_df[feature_cols].values
test_features_scaled = scaler.transform(test_features)
test_target_scaled = test_features_scaled[:, target_idx].reshape(-1, 1)

X_test_all, y_test_all = create_dataset(test_features_scaled, test_target_scaled, I, O)
print("X_test_all:", X_test_all.shape, "y_test_all:", y_test_all.shape)

model.eval()
preds_list = []
with torch.no_grad():
    for i in range(len(X_test_all)):
        x_input = torch.FloatTensor(X_test_all[i]).unsqueeze(0).to(device)  # (1, I, 9)
        pred = model(x_input)  # (1, O)
        preds_list.append(pred.cpu().numpy().squeeze(0))

preds_test = np.array(preds_list)  # (N_test, O)

# ---- 反归一化 (仅对cnt列) ----
def inverse_cnt_scaling(y_scaled_2d, scaler, target_col_idx, num_feats=9):
    N, seq_len = y_scaled_2d.shape
    dummy = np.zeros((N * seq_len, num_feats))
    dummy[:, target_col_idx] = y_scaled_2d.reshape(-1)
    inv_dummy = scaler.inverse_transform(dummy)
    return inv_dummy[:, target_col_idx].reshape(N, seq_len)

preds_test_inv = inverse_cnt_scaling(preds_test, scaler, target_idx, len(feature_cols))
trues_test_inv = inverse_cnt_scaling(y_test_all, scaler, target_idx, len(feature_cols))

# ---- 计算整体 MSE/MAE ----
flatten_preds = preds_test_inv.reshape(-1)
flatten_trues = trues_test_inv.reshape(-1)
test_mse = mean_squared_error(flatten_trues, flatten_preds)
test_mae = mean_absolute_error(flatten_trues, flatten_preds)
print(f"Test MSE: {test_mse:.4f}, Test MAE: {test_mae:.4f}")

# ---- 可视化某个样本 ----
sample_idx = 0
plt.figure(figsize=(10,5))
plt.plot(range(O), trues_test_inv[sample_idx], label="True", marker='o')
plt.plot(range(O), preds_test_inv[sample_idx], label="Pred", marker='x')
plt.title(f"Test Sample {sample_idx} - Next {O} Hours")
plt.xlabel("Hour Ahead")
plt.ylabel("cnt")
plt.legend()
plt.show()
