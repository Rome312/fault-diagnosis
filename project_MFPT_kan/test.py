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

print(signal_100.shape)
print(signal_200.shape)
print(np.isnan(signal_100).sum())  # 如果 > 0，说明有 NaN
print(np.isinf(signal_100).sum())  # 如果 > 0，说明有 Inf
