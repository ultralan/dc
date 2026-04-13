import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tcn import TCN # 确保已安装 keras-tcn

# --- 1. 模拟数据 (Makeup Data) ---

# 模拟 10 天的数据，每 10 分钟采集一次
# (10 天 * 24 小时/天 * 6 个点/小时) = 1440 个数据点
data_points = 1440

# 1.1 创建时间戳
timestamps = pd.date_range(start='2025-01-01', periods=data_points, freq='10min')

# 1.2 创建日周期性 (使用 sin 函数模拟)
# 24小时 * 6个点/小时 = 144个点一个周期
daily_cycle = np.sin(np.linspace(0, 10 * 2 * np.pi, data_points)) * 15 + 55
# (波动范围 15dB, 基础值 55dB)

# 1.3 创建随机波动
random_noise = np.random.randn(data_points) * 2.5 # 标准差 2.5dB

# 1.4 组合成最终的噪声数据
noise_db = daily_cycle + random_noise
noise_db = np.clip(noise_db, 30, 100) # 假设噪声在 30dB 到 100dB 之间

# 1.5 创建 DataFrame
data = pd.DataFrame({'timestamp': timestamps, 'noise_db': noise_db})
data = data.set_index('timestamp')

print("--- 模拟数据 (前5条) ---")
print(data.head())

# 1.6 模拟数据可视化
print("\n--- 正在生成模拟数据可视化图... ---")
plt.figure(figsize=(15, 5))
plt.plot(data['noise_db'])
plt.title("模拟的噪声时间序列 (10天)")
plt.ylabel("Noise (dB)")
plt.xlabel("Time")
plt.show()

# --- 2. TCN 预测模型搭建 ---

# 2.1 提取数据
noise_values = data['noise_db'].values.reshape(-1, 1)

# 2.2 数据预处理 (归一化)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(noise_values)

# 2.3 创建时间序列
# 定义序列长度：用过去多少个点来预测未来
# 比如用过去 2 小时 (12个点) 的数据
LOOKBACK_PERIOD = 12
# 预测未来 1 个点 (10分钟后)
PREDICTION_HORIZON = 1

def create_sequences(data, lookback, horizon):
    X, y = [], []
    for i in range(len(data) - lookback - horizon + 1):
        X.append(data[i:(i + lookback)])
        y.append(data[i + lookback + horizon - 1])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_data, LOOKBACK_PERIOD, PREDICTION_HORIZON)

# 2.4 划分训练集和测试集 (时间序列不能随机打乱!)
# 保持数据的时间顺序
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

print(f"\n数据形状 (X_train): {X_train.shape}")
print(f"数据形状 (y_train): {y_train.shape}")
print(f"数据形状 (X_test): {X_test.shape}")
print(f"数据形状 (y_test): {y_test.shape}")

# --- 3. 构建 TCN 模型 ---

# TCN 的输入形状需要是 (样本数, 时间步长, 特征数)
# 我们的 X_train 已经是 (n_samples, 12, 1) 的形状
input_layer = Input(shape=(LOOKBACK_PERIOD, 1))

# 搭建 TCN 层
tcn_layer = TCN(
    nb_filters=64,     # 过滤器（或通道数）
    kernel_size=3,     # 卷积核大小
    dilations=[1, 2, 4, 8], # 膨胀因子列表
    padding='causal',  # 'causal' 保证时间上的因果性
    return_sequences=False # 只需要最后的输出来预测
)(input_layer)

# 输出层 (预测1个值)
output_layer = Dense(1)(tcn_layer)

model = Model(inputs=input_layer, outputs=output_layer)

# 编译模型 (使用 MSE 损失)
model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

# --- 4. 训练模型 ---
print("\n--- 开始训练模型 ---")
history = model.fit(
    X_train, 
    y_train,
    epochs=50,       # 训练轮数
    batch_size=32,   # 批量大小
    validation_data=(X_test, y_test),
    verbose=2
)

# --- 5. 评估模型 ---
print("\n--- 开始评估模型 ---")
# 1. 获取预测值 (归一化后的)
y_pred_scaled = model.predict(X_test)

# 2. 将预测值和真实值 反归一化 (变回原始的 dB 值)
y_pred = scaler.inverse_transform(y_pred_scaled)
y_test_orig = scaler.inverse_transform(y_test)

# 3. 计算 MAE 和 RMSE
mae = mean_absolute_error(y_test_orig, y_pred)
rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred))

print(f"--- 评估结果 (在测试集上) ---")
print(f"MAE (平均绝对误差): {mae:.2f} dB")
print(f"RMSE (均方根误差): {rmse:.2f} dB")

# --- 6. 可视化预测结果 ---
print("\n--- 正在生成模型预测结果可视化图... ---")
plt.figure(figsize=(15, 6))
plt.plot(y_test_orig, label="真实值 (Actual Noise)", color='blue')
plt.plot(y_pred, label="预测值 (Predicted Noise)", color='red', linestyle='--')
plt.title("TCN 模型预测结果 vs 真实值")
plt.ylabel("Noise (dB)")
plt.legend()
plt.show()

print("\n--- 完整流程结束 ---")