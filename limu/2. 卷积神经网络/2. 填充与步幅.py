import torch
from torch import nn


# 计算卷积
# 由于我们的矩阵没有给定通道数和批量数，所以需要对数据进行处理
# 先在x上添加通道数和批量数,后在y上去掉通道数和批量数
def comp_conv2d(conv2d, X):
    # 这里的（1，1）表示批量大小和通道数都是1
    X = X.reshape((1, 1) + X.shape)
    Y = conv2d(X)
    # 省略前两个维度：批量大小和通道
    return Y.reshape(Y.shape[2:])


# =========================================================
# 填充
# 请注意，这里每边都填充了1行或1列，因此总共添加了2行或2列
conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1)
X = torch.rand(size=(8, 8))
# 输入输出图像尺寸相同
print(comp_conv2d(conv2d, X).shape)

# =========================================================
# 步幅
conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=2)
print(comp_conv2d(conv2d, X).shape)

# =========================================================
# 行和列上赋予不同的填充和步长
conv2d = nn.Conv2d(1, 1, kernel_size=(3, 5), padding=(0, 1), stride=(3, 4))
print(comp_conv2d(conv2d, X).shape)
