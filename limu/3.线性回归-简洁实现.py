import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)


# =======================================================================
# 将数据传入pytorch的数据模块中
def load_array(data_arrays, batch_size, is_train=True):
    # 构造一个PyTorch数据迭代器。
    dataset = data.TensorDataset(*data_arrays)
    # 使用data.DataLoader获取mini-batch
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


batch_size = 10
data_iter = load_array((features, labels), batch_size)


# =======================================================================
# 定义模型
from torch import nn

# 导入torch中写好的线性层，Sequential是神经网络容器，list of layers
net = nn.Sequential(nn.Linear(2, 1))

# 传入模型参数初始值
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)

# 定义loss，均方误差，L2范数
loss = nn.MSELoss()

# 定义优化方法，sgd随机梯度下降
trainer = torch.optim.SGD(net.parameters(), lr=0.03)

# =======================================================================
# 训练
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:  # 获取mini-batch
        l = loss(net(X), y)  # 计算损失
        trainer.zero_grad()  # 清空梯度
        l.backward()  # 计算梯度
        trainer.step()  # 更新模型
    l = loss(net(features), labels)  # 评估新损失（全局）
    print(f'epoch {epoch + 1}, loss {l}')

w = net[0].weight.data
print('w的估计误差：', true_w - w.reshape(true_w.shape))
b = net[0].bias.data
print('b的估计误差：', true_b - b)


