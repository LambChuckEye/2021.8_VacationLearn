# cmd中查询gpu命令
# !nvidia-smi
import torch
from torch import nn
import torchvision

# 查询可用gpu数量
print(torch.cuda.device_count())


def try_gpu(i=0):  # @save
    """如果存在，则返回gpu(i)，否则返回cpu()。"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def try_all_gpus():  # @save
    """返回所有可用的GPU，如果没有GPU，则返回[cpu(),]。"""
    devices = [torch.device(f'cuda:{i}')
               for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]


print(try_gpu(), try_gpu(10), try_all_gpus())

# 默认生成张量在cpu上
x = torch.tensor([1, 2, 3])
print(x.device)

# 声明在gpu上生成张量，对于该张量的计算默认发生在其生成的位置中
X = torch.ones(2, 3, device=try_gpu())
print(X.device)

print('===============================')

# 复制对象至指定设备
Z = X.cuda(0)
print(X)
print(Z)

# 在gpu上做神经网络
net = nn.Sequential(nn.Linear(3, 1))
# 指定gpu
net = net.to(device=try_gpu())
print(net(X))

# 查询权重参数所在设备
print(net[0].weight.data.device)