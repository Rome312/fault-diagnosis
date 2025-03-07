import numpy as np
import pandas as pd
from scipy.io import loadmat
 
# Assuming your filenames are as follows:
file_names = ['base','Outer_27','Inner_1', 'Inner_3', 'Inner_5', 'Inner_7', 'Outer_1', 'Outer_3', 'Outer_5', 'Outer_7']
data_columns = [f'X{filename}_DE_time' for filename in file_names]
# columns_name = [f'de_{filename}' for filename in file_names]
columns_name = ['normal','270_Outer','0_Inner','100_Inner','200_Inner','300_Inner','25_Outer','100_Outer','200_Outer','300_Outer']
 
data_MFPT_10c = pd.DataFrame()  # 名称表示10类
 
 

for index in range(10):
    file_path = (f"C:\\Users\\rome.luo\\python\\毕设\\project_MFPT\\MFPT\\{file_names[index]}.mat")
    data = loadmat(file_path)
    
    # ✅ 确保 'bearing' 存在
    if "bearing" not in data:
        print(f"⚠️ 警告: {file_names[index]} 文件中没有 'bearing' 键！")
        continue
    
    bearing_data = data["bearing"][0, 0]  # 取出 (1,1) 里的 tuple
    if index==0:
        vib_data = bearing_data[1].reshape(-1)  # 取出振动数据并展平
    else :
        vib_data = bearing_data[2].reshape(-1)
    data_MFPT_10c[columns_name[index]] = vib_data[:146484]  # 截取 119808 长度（可调整）

print("数据维度:", data_MFPT_10c.shape)
data_MFPT_10c.to_csv("data_mfpt.csv", index=False)
print("CSV 文件已保存！")