import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy.signal import correlate
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

# 读取数据
data_MFPT_10c = pd.read_csv("data_mfpt.csv")

# 选择 '100_Inner' 和 '200_Inner' 进行比较
signal_100 = data_MFPT_10c['100_Inner'].values
signal_200 = data_MFPT_10c['200_Inner'].values

# 采样频率 (假设 48 kHz, 你可以根据真实情况调整)
fs = 48000  # 48 kHz
N = len(signal_100)  # 采样点数

# 时域波形对比
plt.figure(figsize=(12, 5))
plt.plot(signal_100[:2000], label='100 Inner', alpha=0.8)
plt.plot(signal_200[:2000], label='200 Inner', alpha=0.8)
plt.xlabel('Time Step')
plt.ylabel('Amplitude')
plt.title('Time-Domain Vibration Signal Comparison')
plt.legend()
plt.show()

# 频谱分析 (FFT)
def compute_fft(signal, fs):
    N = len(signal)
    fft_values = fft(signal)
    fft_magnitude = np.abs(fft_values)[:N // 2]  # 取前半部分
    freq_axis = np.linspace(0, fs / 2, N // 2)  # 频率刻度
    return freq_axis, fft_magnitude

freq_100, fft_100 = compute_fft(signal_100, fs)
freq_200, fft_200 = compute_fft(signal_200, fs)

plt.figure(figsize=(12, 5))
plt.plot(freq_100, fft_100, label='100 Inner', alpha=0.8)
plt.plot(freq_200, fft_200, label='200 Inner', alpha=0.8)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.title('Frequency-Domain Vibration Signal Comparison (FFT)')
plt.legend()
plt.xlim([0, 5000])  # 关注 0-5000Hz 频段
plt.show()

# 计算皮尔逊相关系数
corr_coefficient = np.corrcoef(signal_100, signal_200)[0, 1]
print(f"Pearson Correlation Coefficient: {corr_coefficient:.4f}")

# 计算互相关
cross_corr = correlate(signal_100, signal_200, mode='full')
lag = np.arange(-len(signal_100) + 1, len(signal_100))

plt.figure(figsize=(12, 5))
plt.plot(lag, cross_corr)
plt.xlabel("Lag")
plt.ylabel("Cross-Correlation")
plt.title("Cross-Correlation between 100_Inner and 200_Inner")
plt.show()

# 计算 DTW 距离
def custom_euclidean(x, y):
    return np.abs(x - y)  # 计算绝对值距离

distance, path = fastdtw(signal_100, signal_200, dist=custom_euclidean)
print("DTW Distance:", distance)