# CNN-from-Scratch
> ML Assignment 2

这是一个用于训练卷积神经网络（CNN）的项目，主要用于CIFAR-10数据集的分类任务。项目包含了从数据预处理、模型定义、训练、测试到结果可视化的完整流程。

## 如何运行代码

1. **环境准备**：
   
   - 确保安装了Python 3.x。
   - 安装所需的Python包：`torch`, `torchvision`, `matplotlib`。可以通过以下命令安装：
     ```bash
     pip install torch torchvision matplotlib
     ```
   
2. **数据准备**：
   - 运行`save_datasets.py`脚本来下载并保存CIFAR-10数据集。
     ```bash
     python save_datasets.py
     ```
   - 该脚本会将数据集保存到`data`目录下。

3. **训练模型**：
   - 运行`main.py`脚本来训练模型。可以通过命令行参数来配置模型类型、正则化方法、数据增强、权重衰减和dropout率。
     ```bash
     python main.py --model simple --normalization batch --data_augmentation --weight_decay 0.001 --dropout_rate 0.5
     ```
   - 可以通过`run.sh`脚本来批量运行不同的配置。
     ```bash
     bash run.sh
     ```

4. **查看结果**：
   - 训练过程中的损失和测试准确率会被保存到`figs`目录下，并以图像和文本文件的形式记录。

## 文件说明

- **README.md**：项目的说明文档。
- **config.py**：配置文件，定义了训练过程中的参数，如设备类型、批量大小、学习率等。
- **main.py**：主程序，负责解析命令行参数、加载数据、定义模型、训练和测试模型，并保存结果。
- **models/deep_cnn.py**：定义了一个深度CNN模型。
- **models/simple_cnn.py**：定义了一个简单的CNN模型。
- **run.sh**：批量运行不同配置的脚本。
- **save_datasets.py**：下载并保存CIFAR-10数据集。
- **utils/data_loader.py**：加载数据集并生成数据加载器。
- **utils/test.py**：定义了测试模型的函数。
- **utils/train.py**：定义了训练模型的函数。
- **utils/visualize.py**：定义了可视化训练损失和测试准确率的函数。
