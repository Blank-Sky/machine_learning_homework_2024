import pandas as pd
import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

# 检查 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def preprocess_data(file_path):
    # 加载数据
    data = pd.read_csv(file_path)

    # 日期字段处理
    data['dteday'] = pd.to_datetime(data['dteday'])
    data['year'] = data['dteday'].dt.year
    data['month'] = data['dteday'].dt.month
    data['day'] = data['dteday'].dt.day
    data = data.drop(columns=['dteday'])

    # 归一化数值特征
    scaler = MinMaxScaler()
    data[['temp', 'atemp', 'hum', 'windspeed']] = scaler.fit_transform(data[['temp', 'atemp', 'hum', 'windspeed']])

    # 独热编码：使用固定的取值范围
    categories = {
        'season': [1, 2, 3, 4],  # 春、夏、秋、冬
        'weathersit': [1, 2, 3, 4],  # 天气情况
        'holiday': [0, 1],  # 是否假日
        'workingday': [0, 1],  # 是否工作日
        'weekday': [0, 1, 2, 3, 4, 5, 6]  # 一周的天数
    }

    for col, values in categories.items():
        data[col] = data[col].astype(pd.CategoricalDtype(categories=values))  # 设置固定的类别范围

    # 独热编码
    data = pd.get_dummies(data, columns=categories.keys(), drop_first=False)  # 保留所有类别

    return data


# 滑动窗口
def create_time_series(data, input_steps, output_steps, stride=1):
    """
    创建时间序列数据集
    :param data: 数据集 (DataFrame)
    :param input_steps: 输入窗口大小
    :param output_steps: 输出窗口大小
    :param stride: 滑动窗口的步长
    :return: 特征数组 X 和目标数组 y
    """
    X, y = [], []
    for i in range(0, len(data) - input_steps - output_steps, stride):
        X.append(data.iloc[i:i + input_steps].values.astype(np.float32))
        y.append(data.iloc[i + input_steps:i + input_steps + output_steps]['cnt'].values.astype(np.float32))
    return np.array(X), np.array(y)



# PyTorch Dataset
class BikeSharingDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# LSTM 模型（增加神经网络层数）
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, fc_layers=2, fc_hidden_size=64):
        """
        :param input_size: 输入特征数
        :param hidden_size: LSTM 隐藏层大小
        :param num_layers: LSTM 层数
        :param output_size: 输出特征数
        :param fc_layers: 全连接层数量（默认为2）
        :param fc_hidden_size: 每个全连接层的隐藏层大小（默认为64）
        """
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # 构建多层全连接网络
        fc_layers_list = []
        fc_input_size = hidden_size
        for _ in range(fc_layers - 1):
            fc_layers_list.append(nn.Linear(fc_input_size, fc_hidden_size))
            fc_layers_list.append(nn.ReLU())
            fc_input_size = fc_hidden_size
        fc_layers_list.append(nn.Linear(fc_input_size, output_size))
        self.fc = nn.Sequential(*fc_layers_list)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = lstm_out[:, -1, :]  # 取最后一个时间步的输出
        out = self.fc(out)
        return out


# 模型训练与评估
def train_model(model, train_loader, criterion, optimizer, epochs=100):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(train_loader)}")


def evaluate_model(model, X_input, y_true):
    model.eval()
    with torch.no_grad():
        X_input = torch.tensor(X_input, dtype=torch.float32).unsqueeze(0).to(device)
        y_pred = model(X_input).cpu().numpy().flatten()
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    return y_pred, mse, mae


# 数据加载
train_data = preprocess_data("train_data.csv")
test_data = preprocess_data("test_data.csv")
# 检查列是否一致
assert list(train_data.columns) == list(test_data.columns), "Train and test columns are not consistent!"

# 创建短期预测数据
input_steps, output_steps_short = 96, 96
X_train, y_train = create_time_series(train_data, input_steps, output_steps_short)
X_test, y_test = create_time_series(test_data, input_steps, output_steps_short)

# 数据加载器
train_dataset = BikeSharingDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 模型实例化（短期预测）
input_size = X_train.shape[2]
hidden_size, num_layers, output_size = 128, 3, output_steps_short
model_short = LSTMModel(input_size, hidden_size, num_layers, output_size).to(device)

# 训练短期模型
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model_short.parameters(), lr=0.001)
train_model(model_short, train_loader, criterion, optimizer)

index = 0
# 保存模型
torch.save(model_short.state_dict(), f"model_lstm_{input_steps}_{index}.pth")

# 测试短期预测
sample_index = 0
X_input = X_test[sample_index]
y_true = y_test[sample_index]
y_pred, mse, mae = evaluate_model(model_short, X_input, y_true)

# 结果可视化
plt.figure(figsize=(10, 6))
plt.plot(range(output_steps_short), y_true, label='True', marker='o')
plt.plot(range(output_steps_short), y_pred, label='Predicted', marker='x')
plt.title('Short-term Prediction (96 Hours)')
plt.xlabel('Hour')
plt.ylabel('Rental Count')
plt.legend()
plt.grid()
plt.show()

print(f"Short-term Prediction MSE: {mse}, MAE: {mae}")
