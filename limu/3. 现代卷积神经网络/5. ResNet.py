import torch
from torch import nn
from d2l import torch as d2l
import warnings
import MyTools
from torch.nn import functional as F

warnings.filterwarnings("ignore")


# 残差神经网络
# 引入了跳转的概念，有两大作用：
#   1. 对于深层的模型会有更好的效果：
#       在引入残差后，g‘ = g + a, 即深层模型的输出包含了浅层模型的输出，说以至少深层模型的效果不会差于浅层模型
#   2， 使深层模型的训练成为现实：
#       若现有网络 y = f(x), 称其导数为dy
#       对于深层网络 y' = g(y), 对其求导数,dy' = dg(y) · dy
#               若dg(y)很小时，dy'也会变小，在经过很多层处理后，会发生梯度消失
#       对于残差网络 y' = g(y) + y, 对其求导数,dy' = dg(y) · dy + dy
#               因为后面加了一个dy，所以避免了梯度消失的发生，使深层模型的训练成为可能

# 残差块
class Residual(nn.Module):
    #
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            # 1x1 卷积用于原始参数与跳转处通道数不同时使用
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        # 跳转叠加
        Y += X
        return F.relu(Y)


# 残差模块
def resnet_block(input_channels, num_channels, num_residuals,
                 first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels,
                                use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk


b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                   nn.BatchNorm2d(64), nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
# 由于b1进行了两次减半，所以b2不进行减半
b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
b3 = nn.Sequential(*resnet_block(64, 128, 2))
b4 = nn.Sequential(*resnet_block(128, 256, 2))
b5 = nn.Sequential(*resnet_block(256, 512, 2))

net = nn.Sequential(b1, b2, b3, b4, b5,
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(), nn.Linear(512, 10))

lr, num_epochs, batch_size = 0.05, 10, 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)

if __name__ == '__main__':
    MyTools.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())