import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch import nn

# 检查 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 预处理数据
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
def create_time_series(data, input_steps, output_steps):
    X, y = [], []
    for i in range(len(data) - input_steps - output_steps):
        X.append(data.iloc[i:i+input_steps].values.astype(np.float32))
        y.append(data.iloc[i+input_steps:i+input_steps+output_steps]['cnt'].values.astype(np.float32))
    return np.array(X), np.array(y)

# LSTM 模型定义
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = lstm_out[:, -1, :]  # 取最后一个时间步的输出
        out = self.fc(out)
        return out

# 加载保存的模型
def load_model(model_path, input_size, hidden_size, num_layers, output_size):
    model = LSTMModel(input_size, hidden_size, num_layers, output_size).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# 评估模型
def evaluate_model(model, X_input, y_true):
    model.eval()
    with torch.no_grad():
        X_input = torch.tensor(X_input, dtype=torch.float32).unsqueeze(0).to(device)
        y_pred = model(X_input).cpu().numpy().flatten()
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    return y_pred, mse, mae

# 读取并处理数据
test_data = preprocess_data("test_data.csv")

# 创建测试数据
input_steps, output_steps_short = 96, 96
X_test, y_test = create_time_series(test_data, input_steps, output_steps_short)

# 加载模型
input_size = X_test.shape[2]
hidden_size, num_layers, output_size = 128, 2, output_steps_short
model_path = "model_lstm_96_1.pth"
model = load_model(model_path, input_size, hidden_size, num_layers, output_size)

# 测试模型
sample_index = 0
X_input = X_test[sample_index]
y_true = y_test[sample_index]
y_pred, mse, mae = evaluate_model(model, X_input, y_true)

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
