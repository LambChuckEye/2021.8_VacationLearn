# y = w1x1 + w2x2 + w3x3 + b
# y = <w,x> + b
import Timer
import random
import torch
from d2l import torch as d2l


# 生成数据集 y = 2 * x1 + -3.4 * x2 + 4.2
def synthetic_data(w, b, num_examples):
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    # 随机噪音
    y += torch.normal(0, 0.01, y.shape)
    # 列向量
    return X, y.reshape((-1, 1))


features, labels = synthetic_data(torch.tensor([2, -3.4]).reshape(-1, 1), 4.2, 1000)

# 画图
d2l.set_figsize()
d2l.plt.scatter(features[:, 1].detach().numpy(),
                labels.detach().numpy(),
                1)
d2l.plt.show()


# 生成小批量
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    # 生成下标list
    indices = list(range(num_examples))
    # 打乱下标
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i:min(i + batch_size, num_examples)])  # 最后一次不够取，则直接取到最后一位
        yield features[batch_indices], labels[batch_indices]  # yield：循环return


# 初始化模型参数，设置自动求导
w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)


# 定义模型
def linreg(X, w, b):
    """线性回归模型。"""
    return torch.matmul(X, w) + b


# 定义损失函数
def squared_loss(y_hat, y):
    """均方损失。"""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


# 优化算法
def sgd(params, lr, batch_size):
    """小批量随机梯度下降。"""
    # 更新参数时，不参与梯度计算
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


# 训练
lr = 0.03
num_epochs = 3
# 模型类型
net = linreg
# 损失函数
loss = squared_loss
batch_size = 10

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):  # 生成随机小批量
        l = loss(net(X, w, b), y)  # 计算当前损失
        # 因为`l`形状是(`batch_size`, 1)，而不是一个标量。`l`中的所有元素被加到一起，
        # 并以此计算关于[`w`, `b`]的梯度
        l.sum().backward() # 计算梯度
        sgd([w, b], lr, batch_size)  # 使用参数的梯度更新参数
    # 阶段性评价
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')

print(w,b)