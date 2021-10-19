import collections
import math
import os
import shutil
import pandas as pd
import torch
import torchvision
from torch import nn
from d2l import torch as d2l

data_dir = 'data/'


def read_csv_labels(fname):
    """读取 `fname` 来给标签字典返回一个文件名。"""
    with open(fname, 'r') as f:
        # 跳过文件头行 (列名)
        lines = f.readlines()[1:]
    # rstrip()删除末尾空格
    tokens = [l.rstrip().split(',') for l in lines]
    return dict(((name, label) for name, label in tokens))


labels = read_csv_labels(os.path.join(data_dir, 'trainLabels.csv'))
print('# 训练示例 :', len(labels))
print('# 类别 :', len(set(labels.values())))


def copyfile(filename, target_dir):
    """将文件复制到目标目录。"""
    os.makedirs(target_dir, exist_ok=True)
    shutil.copy(filename, target_dir)


# 图片归类并生成验证集
def reorg_train_valid(data_dir, labels, valid_ratio):
    # 训练数据集中示例最少的类别中的示例数
    # collections.Counter(labels.values()).most_common()：返回每一个类别以及类别出现的次数，从大到小排序
    # [('a', 5), ('r', 2), ('b', 2)]，第一个[-1]拿最小，第二个[1]拿类名
    n = collections.Counter(labels.values()).most_common()[-1][1]
    # 验证集中每个类别的示例数
    # math.floor 向下取整
    n_valid_per_label = max(1, math.floor(n * valid_ratio))
    label_count = {}
    for train_file in os.listdir(os.path.join(data_dir, 'train')):
        # 去后缀
        label = labels[train_file.split('.')[0]]
        fname = os.path.join(data_dir, 'train', train_file)
        # 图片归类
        copyfile(fname, os.path.join(data_dir, 'train_valid_test',
                                     'train_valid', label))
        # 验证集中不存在或者验证集数量不足时存到‘valid’中
        if label not in label_count or label_count[label] < n_valid_per_label:
            copyfile(fname, os.path.join(data_dir, 'train_valid_test',
                                         'valid', label))
            # 若存在label返回label的数值，不存在则返回0
            label_count[label] = label_count.get(label, 0) + 1
        else:
            copyfile(fname, os.path.join(data_dir, 'train_valid_test',
                                         'train', label))
    return n_valid_per_label


# 在预测期间整理测试集，以方便读取？
def reorg_test(data_dir):
    for test_file in os.listdir(os.path.join(data_dir, 'test')):
        copyfile(os.path.join(data_dir, 'test', test_file),
                 os.path.join(data_dir, 'train_valid_test', 'test',
                              'unknown'))


def reorg_cifar10_data(data_dir, valid_ratio):
    labels = read_csv_labels(os.path.join(data_dir, 'trainLabels.csv'))
    reorg_train_valid(data_dir, labels, valid_ratio)
    reorg_test(data_dir)


batch_size = 128
valid_ratio = 0.1
# 分类
# reorg_cifar10_data(data_dir, valid_ratio)
reorg_test(data_dir)
print("ok")
# 数据增广及预处理：
# 训练集
transform_train = torchvision.transforms.Compose([
    # 在高度和宽度上将图像放大到40像素的正方形
    torchvision.transforms.Resize(40),
    # 随机裁剪出一个高度和宽度均为40像素的正方形图像，
    # 生成一个面积为原始图像面积0.64到1倍的小正方形，
    # 然后将其缩放为高度和宽度均为32像素的正方形
    torchvision.transforms.RandomResizedCrop(32, scale=(0.64, 1.0),
                                             ratio=(1.0, 1.0)),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    # 标准化图像的每个通道
    torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                     [0.2023, 0.1994, 0.2010])])
# 测试集
transform_test = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                     [0.2023, 0.1994, 0.2010])])

# train_ds, train_valid_ds = [torchvision.datasets.ImageFolder(
#     os.path.join(data_dir, 'train_valid_test', folder),
#     transform=transform_train) for folder in ['train', 'train_valid']]
#
# valid_ds, test_ds = [torchvision.datasets.ImageFolder(
#     os.path.join(data_dir, 'train_valid_test', folder),
#     transform=transform_test) for folder in ['valid', 'test']]
