import torch
from torch import nn
from d2l import torch as d2l
import warnings
import imageTrain

warnings.filterwarnings("ignore")

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# PyTorch不会隐式地调整输入的形状。因此，
# 我们在线性层前定义了展平层（flatten），来调整网络输入的形状
# 保留第0维度，即将像素矩阵压缩为向量
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))


# 初始化
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)


# 对所有层进行应用
net.apply(init_weights);

# 损失函数
loss = nn.CrossEntropyLoss()

# 优化算法
trainer = torch.optim.SGD(net.parameters(), lr=0.1)

num_epochs = 10

if __name__ == "__main__":
    imageTrain.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
