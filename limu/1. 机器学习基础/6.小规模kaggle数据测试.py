import hashlib
import os
import tarfile
import zipfile
import requests
from torch.utils import data

# @save
DATA_HUB = dict()
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'


def download(name, cache_dir=os.path.join('../..', 'data')):
    """下载一个DATA_HUB中的文件，返回本地文件名。"""
    assert name in DATA_HUB, f"{name} 不存在于 {DATA_HUB}."
    url, sha1_hash = DATA_HUB[name]
    os.makedirs(cache_dir, exist_ok=True)
    fname = os.path.join(cache_dir, url.split('/')[-1])
    if os.path.exists(fname):
        sha1 = hashlib.sha1()
        with open(fname, 'rb') as f:
            while True:
                data = f.read(1048576)
                if not data:
                    break
                sha1.update(data)
        if sha1.hexdigest() == sha1_hash:
            return fname
    print(f'正在从{url}下载{fname}...')
    r = requests.get(url, stream=True, verify=True)
    with open(fname, 'wb') as f:
        f.write(r.content)
    return fname


def download_extract(name, folder=None):
    """下载并解压zip/tar文件。"""
    fname = download(name)
    base_dir = os.path.dirname(fname)
    data_dir, ext = os.path.splitext(fname)
    if ext == '.zip':
        fp = zipfile.ZipFile(fname, 'r')
    elif ext in ('.tar', '.gz'):
        fp = tarfile.open(fname, 'r')
    else:
        assert False, '只有zip/tar文件可以被解压缩。'
    fp.extractall(base_dir)
    return os.path.join(base_dir, folder) if folder else data_dir


def download_all():
    """下载DATA_HUB中的所有文件。"""
    for name in DATA_HUB:
        download(name)


import numpy as np
import pandas as pd
import torch
from torch import nn
from d2l import torch as d2l

# ============ 数据下载 ==========================================================
DATA_HUB['kaggle_house_train'] = (
    DATA_URL + 'kaggle_house_pred_train.csv',
    '585e9cc93e70b39160e7921475f9bcd7d31219ce')

DATA_HUB['kaggle_house_test'] = (
    DATA_URL + 'kaggle_house_pred_test.csv',
    'fa19780a7b011d9b009e8bff8e99922a8ee2eb90')

train_data = pd.read_csv(download('kaggle_house_train'))
test_data = pd.read_csv(download('kaggle_house_test'))

# 训练集包含标签（房价），测试集没有
print(train_data.shape)
print(test_data.shape)

# 删除id
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))

# ============ 数据处理 ==========================================================
# 数据标准化
# 获取数字标签，默认类型object为非数字
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
# 对所有数字项进行标准化
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std()))
# 对缺失值填充0，在标准化数据之后，所有非数字数据都消失，因此我们可以将缺失值设置为0
all_features[numeric_features] = all_features[numeric_features].fillna(0)

# 独热编码处理离散值
all_features = pd.get_dummies(all_features, dummy_na=True)

# 生成numpy格式的数据用于训练
n_train = train_data.shape[0]
train_features = torch.tensor(all_features[:n_train].values,
                              dtype=torch.float32)
test_features = torch.tensor(all_features[n_train:].values,
                             dtype=torch.float32)
train_labels = torch.tensor(train_data.SalePrice.values.reshape(-1, 1),
                            dtype=torch.float32)

# ============ 训练 ==========================================================
# 均方损失函数
loss = nn.MSELoss()
# 特征数量
in_features = train_features.shape[1]


# 单层网络
def get_net():
    net = nn.Sequential(nn.Linear(in_features, 100), nn.ReLU(), nn.Linear(100, 1))
    return net


# 均方根误差
def log_rmse(net, features, labels):
    # 为了在取对数时进一步稳定该值，将小于1的值设置为1
    # torch.clamp(x,a,b) 将x中小于a的设为a，大于b的设为b
    clipped_preds = torch.clamp(net(features), 1, float('inf'))
    # 均方根误差
    rmse = torch.sqrt(loss(torch.log(clipped_preds), torch.log(labels)))
    return rmse.item()  # 直接取内容元素


# 训练。返回均方误差记录
def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    # 生成tensor数据迭代器
    dataset = data.TensorDataset(train_features, train_labels)
    train_iter = data.DataLoader(dataset, batch_size, shuffle=True)
    # 这里使用的是Adam优化算法
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate,
                                 weight_decay=weight_decay)
    for epoch in range(num_epochs):
        for X, y in train_iter:
            optimizer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            optimizer.step()
        # 记录均方根误差
        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))
    return train_ls, test_ls


# K折交叉验证
# K折，第i折为验证集，对Xy进行折
def get_k_fold_data(k, i, X, y):
    assert k > 1
    fold_size = X.shape[0] // k  # 每折大小
    X_train, y_train = None, None
    for j in range(k):
        # slice裁剪(开启，结束，步长)
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        # j为验证集
        if j == i:
            X_valid, y_valid = X_part, y_part
        # j为训练集
        # 训练集为空时赋值
        elif X_train is None:
            X_train, y_train = X_part, y_part
        # 训练集不空时拼接
        else:
            X_train = torch.cat([X_train, X_part], 0)
            y_train = torch.cat([y_train, y_part], 0)
    return X_train, y_train, X_valid, y_valid


# K折训练
def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay,
           batch_size):
    train_l_sum, valid_l_sum = 0, 0
    # 对每一折设为一遍验证集
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        net = get_net()
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate,
                                   weight_decay, batch_size)
        # 叠加训练结果
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        if i == 0:
            d2l.plot(list(range(1, num_epochs + 1)), [train_ls, valid_ls],
                     xlabel='epoch', ylabel='rmse', xlim=[1, num_epochs],
                     legend=['train', 'valid'], yscale='log')
        print(f'fold {i + 1}, train log rmse {float(train_ls[-1]):f}, '
              f'valid log rmse {float(valid_ls[-1]):f}')
        d2l.plt.show()
    return train_l_sum / k, valid_l_sum / k


if __name__ == '__main__':
    k, num_epochs, lr, weight_decay, batch_size = 5, 100, 5, 0, 64
    train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr,
                              weight_decay, batch_size)
    print(f'{k}-折验证: 平均训练log rmse: {float(train_l):f}, '
          f'平均验证log rmse: {float(valid_l):f}')
