import torch
from torch import nn
import time
import numpy as np


# 计时器
class Timer:  # @save
    """记录多次运行时间。"""

    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """启动计时器。"""
        self.tik = time.time()

    def stop(self):
        """停止计时器并将时间记录在列表中。"""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """返回平均时间。"""
        return sum(self.times) / len(self.times)

    def sum(self):
        """返回时间总和。"""
        return sum(self.times)

    def cumsum(self):
        """返回累计时间。"""
        return np.array(self.times).cumsum().tolist()


# 累加器
class Accumulator:  # @save
    """在`n`个变量上累加。"""

    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# 准确数量计算
def accuracy(y_hat, y):  # @save
    """计算预测正确的数量。"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


# 评估函数
def evaluate_accuracy_gpu(net, data_iter, device=None):  # @save
    """使用GPU计算模型在数据集上的精度。"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # 设置为评估模式
        if not device:
            device = next(iter(net.parameters())).device
    # 正确预测的数量，总预测的数量
    metric = Accumulator(2)
    for X, y in data_iter:
        # 将数据转移到gpu上
        if isinstance(X, list):
            X = [x.to(device) for x in X]
        else:
            X = X.to(device)
        y = y.to(device)
        metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


# 训练函数
def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):
    """用GPU训练模型(在第六章定义)。"""

    # 初始化权重参数
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)

    net.apply(init_weights)
    # 模型移动到GPU上训练
    print('training on', device)
    net.to(device)
    # 梯度下降
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    # 损失函数
    loss = nn.CrossEntropyLoss()
    # 时间戳
    timer, num_batches = Timer(), len(train_iter)

    for epoch in range(num_epochs):
        # 训练损失之和，训练准确率之和，范例数
        metric = Accumulator(3)
        # 训练模式
        net.train()
        # for i , x in enumerate(X):    返回序号和对象
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()  # 梯度清零
            X, y = X.to(device), y.to(device)  # 数据移至GPU
            y_hat = net(X)  # 正向传播
            l = loss(y_hat, y)  # 计算损失
            l.backward()  # 向后传播
            optimizer.step()  # 迭代赋值
            # 每个mini-batch后评估并绘图
            with torch.no_grad():
                metric.add(l * X.shape[0], accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            # if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
            #     print(f'    epoch {epoch + 1},batch {i + 1}, loss {train_l:f},train acc {train_acc:f}')
        # 每一个epoch后进行评估
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        print(f'epoch {epoch + 1}, loss {train_l:f},train acc {train_acc:f},test acc {test_acc}')
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')
