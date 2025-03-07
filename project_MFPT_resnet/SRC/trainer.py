import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
import tensorflow as tf
from tensorflow import keras
from keras import models, layers
import os
import matplotlib.pyplot as plt
import seaborn as sns

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
        end = start + interval_length  # 计算结束索引    
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

# k折交叉验证
from sklearn.model_selection import train_test_split, KFold
kSplits = 5
kfold = KFold(n_splits=kSplits, random_state=32, shuffle=True)

# Reshape数据
Input_1D = X.reshape([-1,512,1])

# 数据集划分
X_1D_train, X_1D_test, y_1D_train, y_1D_test = train_test_split(Input_1D, Y_CNN, train_size=0.75,test_size=0.25, random_state=101)

# 定义ResNet残差块
def resnet_block(x, filters):
    # Main path
    res = layers.Conv1D(filters=filters, kernel_size=3, padding='same', activation='relu')(x)
    res = layers.Conv1D(filters=filters, kernel_size=3, padding='same', activation='relu')(res)
    
    # Skip connection (matching the number of channels)
    skip = layers.Conv1D(filters=filters, kernel_size=1, padding='same')(x)
    
    # Add the skip connection to the main path
    x = layers.Add()([res, skip])
    return x

# 定义ResNet模型
class ResNet_1D():
    def __init__(self):
        self.model = self.CreateModel()

    def CreateModel(self):
        inputs = layers.Input(shape=(512, 1))
        x = layers.Conv1D(16, 3, strides=2, padding='same', activation='relu')(inputs)
        x = resnet_block(x, 16)
        x = layers.MaxPool1D(pool_size=2)(x)
        x = resnet_block(x, 32)
        x = layers.MaxPool1D(pool_size=2)(x)
        x = resnet_block(x, 64)
        x = layers.MaxPool1D(pool_size=2)(x)
        x = resnet_block(x, 128)
        x = layers.MaxPool1D(pool_size=2)(x)
        x = layers.Flatten()(x)
        x = layers.Dense(100, activation='relu')(x)
        x = layers.Dense(50, activation='relu')(x)
        x = layers.Dense(10)(x)
        outputs = layers.Softmax()(x)

        model = models.Model(inputs, outputs)
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.CategoricalCrossentropy(),
                      metrics=['accuracy'])
        return model

accuracy_1D = []


all_train_loss = []
all_train_accuracy = []
# 训练结果
for train, test in kfold.split(X_1D_train,y_1D_train):
    Classification_1D = ResNet_1D()
    history = Classification_1D.model.fit(X_1D_train[train], y_1D_train[train], verbose=1, epochs=12)

    # 记录 Loss 和 Accuracy
    all_train_loss.append(history.history['loss'])
    all_train_accuracy.append(history.history['accuracy'])

    kf_loss, kf_accuracy = Classification_1D.model.evaluate(X_1D_train[test], y_1D_train[test]) 
    accuracy_1D.append(kf_accuracy)

ResNet_1D_train_accuracy = np.average(accuracy_1D)*100
print('ResNet 1D train accuracy =', ResNet_1D_train_accuracy)

ResNet_1D_test_loss, ResNet_1D_test_accuracy = Classification_1D.model.evaluate(X_1D_test, y_1D_test)
ResNet_1D_test_accuracy *= 100
print('ResNet 1D test accuracy =', ResNet_1D_test_accuracy)

avg_loss = np.mean(all_train_loss, axis=0)
avg_accuracy = np.mean(all_train_accuracy, axis=0)
# 绘制 Loss 和 Accuracy 曲线
plt.figure(figsize=(8, 6))
plt.plot(avg_loss, label="Train Loss", color="red", linestyle="-")
plt.plot(avg_accuracy, label="Train Accuracy", color="blue", linestyle="--")
plt.xlabel("Epoch")
plt.ylabel("Value")
plt.title("Training Loss and Accuracy over Epochs")
plt.legend()
plt.grid()

# 保存 Loss 和 Accuracy 图像
loss_acc_path = os.path.join(result_dir, "loss_accuracy_curve.png")
plt.savefig(loss_acc_path, dpi=300, bbox_inches="tight")



# 定义混淆矩阵
from sklearn.metrics import confusion_matrix

def ConfusionMatrix(Model, X, y):
    y_pred = np.argmax(Model.model.predict(X), axis=1)
    ConfusionMat = confusion_matrix(np.argmax(y, axis=1), y_pred)
    return ConfusionMat



# 绘制并保存训练集混淆矩阵
plt.figure(1)
plt.title('Confusion Matrix - ResNet 1D Train')
sns.heatmap(ConfusionMatrix(Classification_1D, X_1D_train, y_1D_train),
            annot=True, fmt='d', annot_kws={"fontsize": 8}, cmap="YlGnBu")
train_cm_path = os.path.join(result_dir, "confusion_matrix_train.png")
plt.savefig(train_cm_path, dpi=300, bbox_inches='tight')

# 绘制并保存测试集混淆矩阵
plt.figure(2)
plt.title('Confusion Matrix - ResNet 1D Test')
sns.heatmap(ConfusionMatrix(Classification_1D, X_1D_test, y_1D_test),
            annot=True, fmt='d', annot_kws={"fontsize": 8}, cmap="YlGnBu")
test_cm_path = os.path.join(result_dir, "confusion_matrix_test.png")
plt.savefig(test_cm_path, dpi=300, bbox_inches='tight')

# 绘制并保存 K 折交叉验证的训练准确率
plt.figure(3)
plt.title('Train - Accuracy - ResNet 1D')
plt.bar(np.arange(1, kSplits + 1), [i * 100 for i in accuracy_1D])
plt.ylabel('accuracy')
plt.xlabel('folds')
plt.ylim([0, 100])
train_acc_path = os.path.join(result_dir, "train_accuracy_resnet_1d.png")
plt.savefig(train_acc_path, dpi=300, bbox_inches='tight')

# 绘制并保存训练集 vs 测试集的准确率比较图
plt.figure(4)
plt.title('Train vs Test Accuracy - ResNet 1D')
plt.bar([1, 2], [ResNet_1D_train_accuracy, ResNet_1D_test_accuracy])
plt.ylabel('accuracy')
plt.xlabel('folds')
plt.xticks([1, 2], ['Train', 'Test'])
plt.ylim([0, 100])
train_vs_test_acc_path = os.path.join(result_dir, "train_vs_test_accuracy_resnet.png")
plt.savefig(train_vs_test_acc_path, dpi=300, bbox_inches='tight')
