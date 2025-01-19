import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
# 读取 train_data
train_df = pd.read_csv("train_data.csv")
# print("train_data行数:", len(train_df))

train_df.sort_values(by=["dteday", "hr"], inplace=True)
train_df.reset_index(drop=True, inplace=True)
# 特征列
feature_cols = [
    "temp", "atemp", "hum", "windspeed",
    "season", "holiday", "workingday", "weathersit",
    "cnt"
]
train_data_features = train_df[feature_cols].values  # shape: (T_train, num_features)

# 归一化
scaler = MinMaxScaler()
train_data_features_scaled = scaler.fit_transform(train_data_features)

# 定位 'cnt' 在 feature_cols 中的下标
target_idx = feature_cols.index("cnt")
train_data_target_scaled = train_data_features_scaled[:, target_idx].reshape(-1, 1)
def create_dataset(features, target, look_back=96, pred_len=96):
    """
    features: shape (N, num_features)
    target:   shape (N, 1)
    return:
      X_arr: (M, look_back, num_features)
      y_arr: (M, pred_len)
    """
    X_list, y_list = [], []
    length = len(features)
    for i in range(length - look_back - pred_len + 1):
        X_list.append(features[i : i+look_back])
        y_list.append(target[i+look_back : i+look_back+pred_len])
    X_arr = np.array(X_list)
    y_arr = np.array(y_list).squeeze(-1)  # (M, pred_len)
    return X_arr, y_arr

I = 96
O = 240
X_all, y_all = create_dataset(
    train_data_features_scaled,
    train_data_target_scaled,
    look_back=I,
    pred_len=O
)
# print("X_all shape:", X_all.shape)  # (M, 96, 9)
# print("y_all shape:", y_all.shape)  # (M, 96)
train_size = int(0.8 * len(X_all))

X_train = X_all[:train_size]    # 前 80%
y_train = y_all[:train_size]

X_val = X_all[train_size:]      # 后 20%
y_val = y_all[train_size:]

print("训练样本数:", X_train.shape[0])
print("验证样本数:", X_val.shape[0])
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        """
        X: (N, seq_len=I, num_features)
        y: (N, pred_len=O)
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
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        """
        X: (N, seq_len=I, num_features)
        y: (N, pred_len=O)
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
class LSTMPredictor(nn.Module):
    def __init__(self, input_size=9, hidden_size=64, output_size=96, num_layers=1):
        super(LSTMPredictor, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,   # 9
            hidden_size=hidden_size, # 64
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, output_size)  # 输出维度 O=96

    def forward(self, x):
        # x: (batch, seq_len=I=96, input_size=9)
        out, (h_n, c_n) = self.lstm(x)     # out: (batch, seq_len, hidden_size=64)
        out_last = out[:, -1, :]          # 取最后一个时间步 (batch, hidden_size)
        out_fc = self.fc(out_last)        # (batch, 96)
        return out_fc
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMPredictor(input_size=len(feature_cols), hidden_size=64, output_size=O)
model.to(device)

criterion = nn.MSELoss()  # 用于反向传播的损失
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

epochs = 30
for epoch in range(epochs):
    model.train()
    train_mse_sum = 0.0
    train_mae_sum = 0.0
    total_train_samples = 0

    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)  # (batch, 96, 9)
        y_batch = y_batch.to(device)  # (batch, 96)

        optimizer.zero_grad()
        outputs = model(X_batch)      # (batch, 96)

        # 1) 计算训练损失 (MSE for backprop)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        # 2) 累加本批次的 MSE & MAE
        batch_size_ = X_batch.size(0)
        train_mse_sum += torch.mean((outputs - y_batch) ** 2).item() * batch_size_
        train_mae_sum += torch.mean(torch.abs(outputs - y_batch)).item() * batch_size_
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

            pred_val_b = model(X_val_b)  # (batch, 96)

            batch_size_val = X_val_b.size(0)
            val_mse_sum += torch.mean((pred_val_b - y_val_b) ** 2).item() * batch_size_val
            val_mae_sum += torch.mean(torch.abs(pred_val_b - y_val_b)).item() * batch_size_val
            total_val_samples += batch_size_val

    val_mse_epoch = val_mse_sum / total_val_samples
    val_mae_epoch = val_mae_sum / total_val_samples

    print(
        f"[Epoch {epoch+1}/{epochs}] "
        f"Train MSE: {train_mse_epoch:.6f}, Train MAE: {train_mae_epoch:.6f} | "
        f"Val MSE: {val_mse_epoch:.6f},   Val MAE: {val_mae_epoch:.6f} | "
        f"Loss: {loss.item():.6f}"
    )
test_df = pd.read_csv("test_data.csv")
print("test_data行数:", len(test_df))

test_df.sort_values(by=["dteday", "hr"], inplace=True)
test_df.reset_index(drop=True, inplace=True)

# 提取特征
test_data_features = test_df[feature_cols].values  # shape: (T_test, 9)
test_data_features_scaled = scaler.transform(test_data_features)

# cnt 在 feature_cols 中同一位置
test_data_target_scaled = test_data_features_scaled[:, target_idx].reshape(-1, 1)
X_test_all, y_test_all = create_dataset(
    test_data_features_scaled,  # shape: (T_test, 9)
    test_data_target_scaled,    # shape: (T_test, 1)
    look_back=I,
    pred_len=O
)

print("X_test_all shape:", X_test_all.shape)  # (N_test, 96, 9)
print("y_test_all shape:", y_test_all.shape)  # (N_test, 96)

# 用训练好的模型做预测
model.eval()
all_preds_test = []

with torch.no_grad():
    for i in range(len(X_test_all)):
        x_input = torch.FloatTensor(X_test_all[i]).unsqueeze(0).to(device)
        # x_input shape: (1, 96, 9)
        pred = model(x_input)  # shape: (1, 96)
        all_preds_test.append(pred.squeeze(0).cpu().numpy())  # (96,)

all_preds_test = np.array(all_preds_test)  # shape: (N_test, 96)
def inverse_cnt_scaling(y_scaled_2d, scaler, target_idx, num_features=9):
    """
    y_scaled_2d: shape (N, 96),  N = (test_data样本数)
    返回反归一化后的 shape (N, 96)
    """
    N, seq_len = y_scaled_2d.shape
    dummy = np.zeros((N * seq_len, num_features))
    dummy[:, target_idx] = y_scaled_2d.reshape(-1)

    inv_dummy = scaler.inverse_transform(dummy)  # (N*seq_len, num_features)
    cnt_values = inv_dummy[:, target_idx].reshape(N, seq_len)
    return cnt_values

preds_test_inv = inverse_cnt_scaling(all_preds_test, scaler, target_idx, len(feature_cols))
trues_test_inv = inverse_cnt_scaling(y_test_all,     scaler, target_idx, len(feature_cols))
flatten_preds_test = preds_test_inv.reshape(-1)
flatten_trues_test = trues_test_inv.reshape(-1)

test_mse = mean_squared_error(flatten_trues_test, flatten_preds_test)
test_mae = mean_absolute_error(flatten_trues_test, flatten_preds_test)

print(f"Test MSE: {test_mse:.4f}, Test MAE: {test_mae:.4f}")
sample_idx = 500
pred_seq = preds_test_inv[sample_idx]  # (96,)
true_seq = trues_test_inv[sample_idx]  # (96,)

plt.figure(figsize=(10, 5))
plt.plot(range(O), true_seq, label="True", marker='o')
plt.plot(range(O), pred_seq, label="Predicted", marker='x')
plt.title(f"Test Sample {sample_idx} - Future {O} Hours Prediction")
plt.xlabel("Hour Ahead")
plt.ylabel("cnt")
plt.legend()
plt.show()
