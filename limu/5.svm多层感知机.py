import torch
from torch import nn
from d2l import torch as d2l
import imageTrain
import warnings

warnings.filterwarnings("ignore")
#
# net = nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU(),
#                     nn.Linear(256, 10))

# 添加dropout正则化
net = nn.Sequential(nn.Flatten(),
                    nn.Linear(784, 256), nn.ReLU(), nn.Dropout(0.5),
                    nn.Linear(256, 10))


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)


net.apply(init_weights)

batch_size, lr, num_epochs, wd = 256, 0.1, 10, 0.01

loss = nn.CrossEntropyLoss()

# 正常参数更新
trainer = torch.optim.SGD(net.parameters(), lr=lr)

# 权重衰退-正则化
# trainer = torch.optim.SGD([{"params": net[1].weight, 'weight_decay': wd},  # 添加权重衰退（正则化）
#                            {"params": net[1].bias}],
#                           lr=lr)
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

if __name__ == '__main__':
    imageTrain.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
