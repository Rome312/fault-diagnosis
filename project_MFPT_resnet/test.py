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

    bearing_data = data["bearing"][0, 0]  # 提取 (1,1) 结构里的数据
    print(type(bearing_data))  # 看看是不是 tuple
    print(len(bearing_data))   # 看看有几个元素
    for i, item in enumerate(bearing_data):
        print(f"Element {i}: {type(item)}, shape: {item.shape if isinstance(item, np.ndarray) else 'N/A'}")
