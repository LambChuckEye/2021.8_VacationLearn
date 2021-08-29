import torch
import torchvision
from matplotlib import pyplot as plt
from torch import nn
from d2l import torch as d2l

d2l.set_figsize()
img = d2l.Image.open('cat.jpg')
d2l.plt.imshow(img)
plt.show()


# aug:数据增广方法
# 图像放大1.5倍，重复2*4=8次
def apply(img, aug, num_rows=2, num_cols=4, scale=1.5):
    Y = [aug(img) for _ in range(num_rows * num_cols)]
    d2l.show_images(Y, num_rows, num_cols, scale=scale)
    plt.show()


# ===================图像尺寸类===================================
# 水平翻转
apply(img, torchvision.transforms.RandomHorizontalFlip())
# 上下反转
apply(img, torchvision.transforms.RandomVerticalFlip())
# 随机剪裁
# 输出图像 200*200，裁剪比例 10%~100%，高宽比例1:2或2:1
shape_aug = torchvision.transforms.RandomResizedCrop(
    (200, 200), scale=(0.1, 1), ratio=(0.5, 2))
apply(img, shape_aug)

# ===================图像色彩类===================================
# 改变亮度，对比度，饱和度，色温，0.5：上下浮动50%
color_aug = torchvision.transforms.ColorJitter(
    brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
apply(img, color_aug)

# ===================综合使用===================================
augs = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(), color_aug, shape_aug])
apply(img, augs)