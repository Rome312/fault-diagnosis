import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
 
 
# 读取CSV文件的前1000行
df = pd.read_csv('data_mfpt.csv', nrows=1000)
 
# 获取列名
columns = df.columns
 
# 设置子图布局
fig, axs = plt.subplots(5, 2, figsize=(12, 15), sharex=True)
 
# 将列名按顺序分配给子图
for i in range(5):
    for j in range(2):
        index = i * 2 + j
        if index < len(columns):
            axs[i, j].plot(df[columns[index]])
            axs[i, j].set_title(columns[index])
            axs[i, j].set_xlabel('Time')
            axs[i, j].set_ylabel('Vibration Signal')
 
plt.savefig('sequence_chart/my_sequence_plot.png')
            
# 调整布局
plt.tight_layout()