import torch
from matplotlib import pyplot as plt
from torch import nn
from d2l import torch as d2l
import MyTools
import warnings

warnings.filterwarnings("ignore")


# 处理输入图像，通道数为1，尺寸为 28x28
class Reshape(torch.nn.Module):
    def forward(self, x):
        return x.view(-1, 1, 28, 28)


# 网络模型
net1 = torch.nn.Sequential(
    Reshape(),
    nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
    nn.Linear(120, 84), nn.Sigmoid(),
    nn.Linear(84, 10))

# 添加了批量归一化的网络模型
# batchNorm, 批量归一化：
#     对于每一个隐藏层的输出进行归一化操作
#       BatchNorm(x) = γ · ((x - μ) / σ^2) + β
#     为每一个隐藏层的输出训练出合适的均值和方差，其作用有如下三点：
#           1. 数据归一化后，可以使参数分布更加均匀，从而使梯度下降更加迅速
#           2. 可以减少本层输入与上层输出之间的联系，从而使训练中，各隐藏层权重变化更加独立，
#                   使得下层不在过分依赖于上层，即在上层权重发生变化时，下层权重不至于完全改变。
#           3. 在归一过程中会引入一些噪音，这些噪音可以起到正则化的作用，
#                   故在引入batchNorm后不需要做正则化操作。
net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5), nn.BatchNorm2d(6), nn.Sigmoid(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), nn.BatchNorm2d(16), nn.Sigmoid(),
    nn.MaxPool2d(kernel_size=2, stride=2), nn.Flatten(),
    nn.Linear(256, 120), nn.BatchNorm1d(120), nn.Sigmoid(),
    nn.Linear(120, 84), nn.BatchNorm1d(84), nn.Sigmoid(),
    nn.Linear(84, 10))

# 检查模型
# X = torch.rand(size=(28, 28), dtype=torch.float32)
# for layer in net:
#     X = layer(X)
#     print(layer.__class__.__name__, 'output shape: \t', X.shape)


# ============================================================
# 训练
batch_size = 256
# 获取mnist数据集
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)

lr, num_epochs = 0.9, 10

if __name__ == '__main__':
    MyTools.train_ch6(net1, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
