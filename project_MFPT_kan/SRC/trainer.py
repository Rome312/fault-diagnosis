import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import confusion_matrix

# --------------------------
# 1. 数据预处理部分
# --------------------------

# 确保 result 目录存在
result_dir = "result"
os.makedirs(result_dir, exist_ok=True)

# 读取 CSV 文件
df = pd.read_csv('data_mfpt.csv')

# 定义信号间隔长度和每块样本点数
interval_length = 512
samples_per_block = 128

# 数据前处理函数
def Sampling(signal, interval_length, samples_per_block):
    num_samples = len(signal)
    num_blocks = num_samples // samples_per_block
    samples = []
    for i in range(num_blocks):
        start = i * samples_per_block  # 计算起始索引
        end = start + interval_length    # 计算结束索引    
        if end <= num_samples:  # 确保不会超出 signal 长度
            samples.append(signal[start:end])
    return np.array(samples)

def DataPreparation(df, interval_length, samples_per_block):
    X, LabelPositional, Label = None, None, None
    for count, column in enumerate(df.columns):
        SplitData = Sampling(df[column].values, interval_length, samples_per_block)
        y = np.zeros([len(SplitData), 10])
        y[:, count] = 1
        y1 = np.zeros([len(SplitData), 1])
        y1[:, 0] = count
        # 堆叠并标记数据
        if X is None:
            X = SplitData
            LabelPositional = y
            Label = y1
        else:
            X = np.append(X, SplitData, axis=0)
            LabelPositional = np.append(LabelPositional, y, axis=0)
            Label = np.append(Label, y1, axis=0)
    return X, LabelPositional, Label

# 数据前处理
X, Y_CNN, Y = DataPreparation(df, interval_length, samples_per_block)
print('Shape of Input Data =', X.shape)
print('Shape of Label Y_CNN =', Y_CNN.shape)
print('Shape of Label Y =', Y.shape)

# Reshape数据：原始为 (N,512) ，转换为 (N,1,512)
Input_1D = X.reshape(-1, 1, 512)  # 结果形状 (N,1,512)

# 数据集划分（训练集75%，测试集25%）
X_1D_train, X_1D_test, y_1D_train, y_1D_test = train_test_split(Input_1D, Y_CNN, 
                                                                  train_size=0.75, 
                                                                  test_size=0.25, 
                                                                  random_state=101)
# 注意：y_1D_train, y_1D_test 目前为 one-hot 编码，转换为类别索引
y_1D_train_idx = np.argmax(y_1D_train, axis=1)
y_1D_test_idx = np.argmax(y_1D_test, axis=1)

# 转换为Tensor
X_1D_train_tensor = torch.tensor(X_1D_train, dtype=torch.float32)
X_1D_test_tensor = torch.tensor(X_1D_test, dtype=torch.float32)
y_1D_train_tensor = torch.tensor(y_1D_train_idx, dtype=torch.long)
y_1D_test_tensor = torch.tensor(y_1D_test_idx, dtype=torch.long)

# 构造数据集（训练集用于后续K折交叉验证）
train_dataset = data.TensorDataset(X_1D_train_tensor, y_1D_train_tensor)
test_dataset = data.TensorDataset(X_1D_test_tensor, y_1D_test_tensor)

# --------------------------
# 2. 模型定义部分
# --------------------------

# 定义1D残差块（仿照Keras代码）
class ResNetBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResNetBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        # 若输入通道与输出通道不一致，则用1x1卷积调整
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, padding=0),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
            
    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = self.relu(out)
        return out

# --------------------------
# 实现KAN模块（基于Kolmogorov-Arnold Networks） 
# --------------------------
# 可学习的激活函数，采用多项式（默认二次多项式）
class LearnableActivation(nn.Module):
    def __init__(self, degree=2):
        super(LearnableActivation, self).__init__()
        self.degree = degree
        # 初始化多项式系数：a0=0, a1=1, 其余为0
        init_vals = [0.0] * (degree + 1)
        init_vals[1] = 1.0
        self.coeffs = nn.Parameter(torch.tensor(init_vals, dtype=torch.float32))
        
    def forward(self, x):
        out = 0
        # 计算多项式表达式：sum_{i=0}^{degree} coeffs[i] * x^i
        for i in range(self.degree + 1):
            out = out + self.coeffs[i] * (x ** i)
        return out

# 定义KAN层（2层全连接，中间采用可学习的激活函数）
class KAN(nn.Module):
    def __init__(self, dims):
        """
        dims: list of dimensions, e.g. [input_dim, hidden_dim, output_dim]
        """
        super(KAN, self).__init__()
        self.fc1 = nn.Linear(dims[0], dims[1])
        self.act = LearnableActivation(degree=2)  # 可学习激活函数，不再固定ReLU
        self.fc2 = nn.Linear(dims[1], dims[2])
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

# 定义融合KAN的ResNet模型
class ResNet1D_KAN(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet1D_KAN, self).__init__()
        # 第一层Conv1d
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, stride=2, padding=1)  # 输出: (16, 256)
        self.relu = nn.ReLU()
        # 堆叠残差块及MaxPool1d层
        self.block1 = ResNetBlock1D(16, 16)   # (16,256)
        self.pool1 = nn.MaxPool1d(kernel_size=2)  # -> (16,128)
        
        self.block2 = ResNetBlock1D(16, 32)   # -> (32,128)
        self.pool2 = nn.MaxPool1d(kernel_size=2)  # -> (32,64)
        
        self.block3 = ResNetBlock1D(32, 64)   # -> (64,64)
        self.pool3 = nn.MaxPool1d(kernel_size=2)  # -> (64,32)
        
        self.block4 = ResNetBlock1D(64, 128)   # -> (128,32)
        self.pool4 = nn.MaxPool1d(kernel_size=2)  # -> (128,16)
        
        # 经过卷积部分后，特征图尺寸：(batch, 128, 16) => flatten 2048 维
        self.flatten_dim = 128 * 16
        
        # 融合KAN：使用真正的KAN模块，其激活函数为可学习的多项式形式
        self.kan = KAN([self.flatten_dim, 64, num_classes])
        
    def forward(self, x):
        # x: (batch, 1, 512)
        x = self.relu(self.conv1(x))   # -> (batch,16,256)
        x = self.block1(x)             # -> (batch,16,256)
        x = self.pool1(x)              # -> (batch,16,128)
        x = self.block2(x)             # -> (batch,32,128)
        x = self.pool2(x)              # -> (batch,32,64)
        x = self.block3(x)             # -> (batch,64,64)
        x = self.pool3(x)              # -> (batch,64,32)
        x = self.block4(x)             # -> (batch,128,32)
        x = self.pool4(x)              # -> (batch,128,16)
        x = x.view(x.size(0), -1)      # flatten to (batch, 2048)
        x = self.kan(x)              # 经过KAN模块进行非线性映射和分类
        return x

# --------------------------
# 3. 训练及K折交叉验证
# --------------------------

# 设置超参数
epochs = 12
batch_size = 32
learning_rate = 0.001
kSplits = 5

criterion = nn.CrossEntropyLoss()

# KFold划分（对训练集部分）
kfold = KFold(n_splits=kSplits, shuffle=True, random_state=32)

# 用于记录各折的验证准确率，以及每个epoch的训练loss和acc
fold_train_losses = []   # 每个折每个epoch的训练loss
fold_train_accs = []     # 每个折每个epoch的训练accuracy
accuracy_1D = []         # 每个折在验证集上的accuracy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# K折交叉验证
for fold, (train_idx, val_idx) in enumerate(kfold.split(X_1D_train_tensor)):
    print(f"Fold {fold+1}/{kSplits}")
    # 创建折内训练与验证数据集
    train_subsampler = data.Subset(train_dataset, train_idx)
    val_subsampler = data.Subset(train_dataset, val_idx)
    
    train_loader = data.DataLoader(train_subsampler, batch_size=batch_size, shuffle=True)
    val_loader = data.DataLoader(val_subsampler, batch_size=batch_size, shuffle=False)
    
    # 初始化模型，每个折重新初始化
    model = ResNet1D_KAN(num_classes=10).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    epoch_train_losses = []
    epoch_train_accs = []
    
    # 每折训练 epochs 轮
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        epoch_train_losses.append(epoch_loss)
        epoch_train_accs.append(epoch_acc)
        print(f"  Epoch {epoch+1}/{epochs}: Loss={epoch_loss:.4f}, Acc={epoch_acc:.4f}")
    
    fold_train_losses.append(epoch_train_losses)
    fold_train_accs.append(epoch_train_accs)
    
    # 在验证集上评估当前折模型
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)
    fold_acc = val_correct / val_total
    accuracy_1D.append(fold_acc)
    print(f"  Validation Accuracy for fold {fold+1}: {fold_acc:.4f}")

# 计算每个epoch在所有折上的平均训练loss和accuracy
avg_loss = np.mean(fold_train_losses, axis=0)
avg_accuracy = np.mean(fold_train_accs, axis=0)

# 计算训练集总体平均准确率（K折交叉验证结果）
ResNet_1D_train_accuracy = np.mean(accuracy_1D) * 100
print('ResNet 1D train accuracy (KFold) =', ResNet_1D_train_accuracy)

# 用最后一折的模型作为最终模型，对测试集进行评估
test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
model.eval()
test_loss = 0.0
test_correct = 0
test_total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        test_correct += (predicted == labels).sum().item()
        test_total += labels.size(0)
test_loss = test_loss / test_total
ResNet_1D_test_accuracy = (test_correct / test_total) * 100
print('ResNet 1D test accuracy =', ResNet_1D_test_accuracy)

# --------------------------
# 4. 绘图部分
# --------------------------

# (1) 绘制 Loss 和 Accuracy 曲线（训练过程中每个 epoch 的平均loss和accuracy）
plt.figure(figsize=(8, 6))
plt.plot(avg_loss, label="Train Loss", color="red", linestyle="-")
plt.plot(avg_accuracy, label="Train Accuracy", color="blue", linestyle="--")
plt.xlabel("Epoch")
plt.ylabel("Value")
plt.title("Training Loss and Accuracy over Epochs")
plt.legend()
plt.grid()
loss_acc_path = os.path.join(result_dir, "loss_accuracy_curve.png")
plt.savefig(loss_acc_path, dpi=300, bbox_inches="tight")
plt.close()

# 定义函数：利用模型对整个数据集进行预测
def get_predictions(model, dataset, batch_size=32):
    loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    all_preds = []
    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
    return np.array(all_preds)

# (2) 绘制并保存训练集混淆矩阵
train_preds = get_predictions(model, train_dataset)
train_true = y_1D_train_idx  # 原始训练标签（one-hot转成索引）
cm_train = confusion_matrix(train_true, train_preds)
plt.figure(figsize=(8,6))
plt.title('Confusion Matrix - ResNet 1D Train')
sns.heatmap(cm_train, annot=True, fmt='d', cmap="YlGnBu", annot_kws={"fontsize": 8})
train_cm_path = os.path.join(result_dir, "confusion_matrix_train.png")
plt.savefig(train_cm_path, dpi=300, bbox_inches='tight')
plt.close()

# (3) 绘制并保存测试集混淆矩阵
test_preds = get_predictions(model, test_dataset)
test_true = y_1D_test_idx
cm_test = confusion_matrix(test_true, test_preds)
plt.figure(figsize=(8,6))
plt.title('Confusion Matrix - ResNet 1D Test')
sns.heatmap(cm_test, annot=True, fmt='d', cmap="YlGnBu", annot_kws={"fontsize": 8})
test_cm_path = os.path.join(result_dir, "confusion_matrix_test.png")
plt.savefig(test_cm_path, dpi=300, bbox_inches='tight')
plt.close()

# (4) 绘制并保存 K 折交叉验证的训练准确率柱状图
plt.figure(figsize=(8,6))
plt.title('Train - Accuracy - ResNet 1D')
plt.bar(np.arange(1, kSplits + 1), [acc * 100 for acc in accuracy_1D])
plt.ylabel('Accuracy (%)')
plt.xlabel('Fold')
plt.ylim([0, 100])
train_acc_path = os.path.join(result_dir, "train_accuracy_resnet_1d.png")
plt.savefig(train_acc_path, dpi=300, bbox_inches='tight')
plt.close()

# (5) 绘制并保存训练集 vs 测试集的准确率比较图
plt.figure(figsize=(8,6))
plt.title('Train vs Test Accuracy - ResNet 1D')
plt.bar([1, 2], [ResNet_1D_train_accuracy, ResNet_1D_test_accuracy], color=['blue', 'green'])
plt.ylabel('Accuracy (%)')
plt.xlabel('Dataset')
plt.xticks([1, 2], ['Train', 'Test'])
plt.ylim([0, 100])
train_vs_test_acc_path = os.path.join(result_dir, "train_vs_test_accuracy_resnet.png")
plt.savefig(train_vs_test_acc_path, dpi=300, bbox_inches='tight')
plt.close()

print("All figures saved in the 'result' directory.")
