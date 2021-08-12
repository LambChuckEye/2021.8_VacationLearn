import cv2
import numpy as np


# 图片展示
def show(img):
    cv2.imshow('1', img)
    # 0：任意键退出
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ======================================
# 1.高斯金字塔
# 向下采样方法（缩小）：
#       1. 将图像与高斯核做卷积
#       2. 将所有偶数行和列去除
# 向上采样方法（放大）：
#       1. 行和列扩大两倍，新增的行和列以0填充
#       2. 使用高斯核做卷积

img = cv2.imread("data\AM.png")
show(img)
print(img.shape)

# 上采样，放大
up = cv2.pyrUp(img)
show(up)
print(up.shape)

# 下采样，缩小
down = cv2.pyrDown(img)
show(down)
print(down.shape)

# ======================================
# 2.拉普拉斯金字塔
# 每层进行：原始图像 - 先down后up后的图像
down = cv2.pyrDown(img)
down_up = cv2.pyrUp(down)
l_1 = img - down_up
show(l_1)
