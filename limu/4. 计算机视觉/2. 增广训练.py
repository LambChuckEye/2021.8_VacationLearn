import torch
import torchvision
from matplotlib import pyplot as plt
from torch import nn
from d2l import torch as d2l
import MyTools

# 下载数据集
# all_images = torchvision.datasets.CIFAR10(train=True, root="../data",
#                                           download=True)
# d2l.show_images([all_images[i][0] for i in range(32)], 4, 8, scale=0.8)
# plt.show()

# 训练变换方法：随机横向反转
train_augs = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor()])
# 测试无变换
test_augs = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()])


# 应用增广
def load_cifar10(is_train, augs, batch_size):
    dataset = torchvision.datasets.CIFAR10(root="../data", train=is_train,
                                           transform=augs, download=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=is_train, num_workers=2)
    return dataloader


# 初始化
def init_weights(m):
    if type(m) in [nn.Linear, nn.Conv2d]:
        nn.init.xavier_uniform_(m.weight)


batch_size, devices, net = 256, d2l.try_all_gpus(), d2l.resnet18(10, 3)

def train_with_data_aug(train_augs, test_augs, net, lr=0.001):
    train_iter = load_cifar10(True, train_augs, batch_size)
    test_iter = load_cifar10(False, test_augs, batch_size)
    loss = nn.CrossEntropyLoss(reduction="none")
    trainer = torch.optim.Adam(net.parameters(), lr=lr)
    MyTools.train_ch13(net, train_iter, test_iter, loss, trainer, 10, devices)


if __name__ == '__main__':
    net.apply(init_weights)
    train_with_data_aug(train_augs, test_augs, net)
