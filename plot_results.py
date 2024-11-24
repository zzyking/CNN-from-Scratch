import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# 假设你有以下数据
data = {
  'model_types': [val for val in ["simple", "deep"] for i in range(96)],
  'normalizations': [val for val in ["none", "batch", "layer"] for i in range(32)] * 2,
  'data_augmentations': [val for val in ["True", "False"] for i in range(16)] * 6,
  'weight_decays': [val for val in [0.0, 0.0001, 0.001, 0.01] for i in range(4)] * 12,
  'dropout_rates': [0.0, 0.2, 0.5, 0.8] * 48,
  'accuracy': [0.7531, 0.7518, 0.7334, 0.6682, 0.7477, 0.7516, 0.7267, 0.6811, 0.7419, 0.7351, 0.7269, 0.6903, 0.6155, 0.6081, 0.5810, 0.5278, \
    0.7201, 0.7296, 0.7133, 0.6813, 0.7230, 0.7153, 0.7315, 0.6825, 0.7345, 0.7379, 0.7257, 0.6553, 0.6640, 0.6462, 0.6318, 0.5629, \
    0.7531, 0.7529, 0.7034, 0.4131, 0.7534, 0.7569, 0.7370, 0.6478, 0.7374, 0.7304, 0.7118, 0.6625, 0.6809, 0.6674, 0.6355, 0.5803, \
    0.7372, 0.7250, 0.7214, 0.4197, 0.7239, 0.7425, 0.7226, 0.5856, 0.7243, 0.7386, 0.7196, 0.6337, 0.6643, 0.6638, 0.6720, 0.6076, \
    0.7239, 0.7269, 0.6530, 0.2159, 0.7395, 0.7148, 0.7050, 0.6061, 0.6963, 0.6901, 0.6689, 0.5783, 0.5969, 0.5768, 0.5290, 0.5104, \
    0.7128, 0.7076, 0.6764, 0.1935, 0.7171, 0.7136, 0.6816, 0.5246, 0.7081, 0.6955, 0.6565, 0.1, 0.6154, 0.6179, 0.5814, 0.5161, \
    0.7958, 0.7896, 0.7622, 0.6716, 0.7973, 0.7816, 0.7784, 0.7025, 0.7872, 0.7819, 0.7570, 0.6500, 0.6364, 0.5804, 0.5467, 0.2842, \
    0.7579, 0.7559, 0.7574, 0.6793, 0.7503, 0.7668, 0.7526, 0.6705, 0.7742, 0.7667, 0.7513, 0.6792, 0.6503, 0.6285, 0.5899, 0.3332, \
    0.8164, 0.8158, 0.7961, 0.4920, 0.7992, 0.8163, 0.7911, 0.6808, 0.7962, 0.7994, 0.7706, 0.6921, 0.7052, 0.6791, 0.6225, 0.4605, \
    0.7756, 0.7793, 0.7737, 0.4659, 0.7678, 0.7875, 0.7545, 0.5566, 0.7865, 0.7838, 0.7537, 0.6166, 0.7335, 0.6689, 0.6850, 0.5046, \
    0.7841, 0.7785, 0.7162, 0.1, 0.7693, 0.7777, 0.7479, 0.1, 0.7345, 0.7267, 0.6955, 0.5237, 0.5867, 0.5836, 0.5477, 0.2466, \
    0.7484, 0.7723, 0.7247, 0.1, 0.7314, 0.7560, 0.7195, 0.1, 0.7337, 0.7175, 0.6920, 0.1, 0.6596, 0.6362, 0.6115, 0.3296]
}

df = pd.DataFrame(data)
'''
# 绘制分类变量与 accuracy 之间的关系图
plt.figure(figsize=(15, 10))

# 绘制 model_types 与 accuracy 的关系
plt.subplot(2, 3, 1)
sns.boxplot(x='model_types', y='accuracy', data=df)
plt.title('Model Types vs Accuracy')

# 绘制 normalizations 与 accuracy 的关系
plt.subplot(2, 3, 2)
sns.boxplot(x='normalizations', y='accuracy', data=df)
plt.title('Normalizations vs Accuracy')

# 绘制 data_augmentations 与 accuracy 的关系
plt.subplot(2, 3, 3)
sns.boxplot(x='data_augmentations', y='accuracy', data=df)
plt.title('Data Augmentations vs Accuracy')

# 绘制 weight_decays 与 accuracy 的关系
plt.subplot(2, 3, 4)
sns.boxplot(x='weight_decays', y='accuracy', data=df)
plt.title('Weight Decays vs Accuracy')

# 绘制 dropout_rates 与 accuracy 的关系
plt.subplot(2, 3, 5)
sns.boxplot(x='dropout_rates', y='accuracy', data=df)
plt.title('Dropout Rates vs Accuracy')

plt.tight_layout()
plt.show()
'''
plt.figure(figsize=(12, 9))

plt.subplot(3, 2, 1)
sns.violinplot(x='model_types', y='accuracy', data=df)
plt.title('Model Types')

# 绘制 normalizations 与 accuracy 的关系
plt.subplot(3, 2, 2)
sns.violinplot(x='normalizations', y='accuracy', data=df)
plt.title('Normalizations')

# 绘制 data_augmentations 与 accuracy 的关系
plt.subplot(3, 2, 3)
sns.violinplot(x='data_augmentations', y='accuracy', data=df)
plt.title('Data Augmentations')

# 绘制 weight_decays 与 accuracy 的关系
plt.subplot(3, 2, 4)
sns.violinplot(x='weight_decays', y='accuracy', data=df)
plt.title('Weight Decays')

# 绘制 dropout_rates 与 accuracy 的关系
plt.subplot(3, 2, 5)
sns.violinplot(x='dropout_rates', y='accuracy', data=df)
plt.title('Dropout Rates')

plt.tight_layout()
plt.savefig('figure.png', dpi=300)
plt.show()
