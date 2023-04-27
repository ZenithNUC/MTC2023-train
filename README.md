# MTC2023-train

## 简介 (Introduction)

这是一个用于分类19类28x28像素图片的深度学习项目。在这个项目中，我们使用PyTorch实现了LeNet-5、VGGNet和ResNet等不同的卷积神经网络模型，并提供了训练和测试脚本。数据集采用[USTC-TFC2016](https://github.com/yungshenglu/USTC-TFC2016) ，数据预处理工具为[MTC2023](https://github.com/ZenithNUC/MTC2023) 。

This is a deep learning project for classifying 19 classes of 28x28 pixel images. In this project, we implement various convolutional neural networks models such as LeNet-5, VGGNet, and ResNet using PyTorch and provide training and testing scripts. The dataset we used is [USTC-TFC2016](https://github.com/yungshenglu/USTC-TFC2016) and the tool to preprocess the dataset is [MTC2023](https://github.com/ZenithNUC/MTC2023).

## 开始 (Getting Started)

### 环境需求 (Environment Requirements)

- Python 3.6+
- PyTorch 1.9+
- torchvision 0.10+

### 安装 (Installation)

1. 克隆仓库 (Clone the repository)

```shell
git clone git@github.com:ZenithNUC/MTC2023-train.git
```

2. 进入项目文件夹 (Enter the project folder)

```shell
cd ./MTC2023-train
```

3. 安装依赖 (Install dependencies)

```shell
pip install -r requirements.txt
```

### 使用方法 (Usage)

1. 准备数据 (Prepare data)

将训练集和测试集分别放入`./train`和`./test`文件夹。

Put the training and testing datasets into the folders named `./train` and `./test`.


2. 训练模型 (Train the model)

修改`main.py`中的代码选择相应的神经网络

Edit the code in `main.py` to choose the neural network.

```shell
python ./main.py
```

## 贡献 (Contributing)

欢迎提交拉取请求以提供新功能或改进。在提交请求之前，请确保您已更新文档并编写相应的测试。

Pull requests for new features or improvements are welcome. Please make sure you have updated the documentation and written the corresponding tests before submitting the request.

## 许可证 (License)

本项目采用 [MIT 许可证](LICENSE)。

This project is licensed under the [MIT License](LICENSE).