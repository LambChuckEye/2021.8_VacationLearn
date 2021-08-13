import cv2
import numpy as np


# 图片展示
def show(img):
    cv2.imshow('1', img)
    # 0：任意键退出
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Canny边缘检测
# 1) 使用高斯滤波器，以平滑图像，滤除噪声。
#
# 2) 计算图像中每个像素点的梯度强度和方向。
#           使用sobel算子
#           梯度强度：G= sqt(Gx^2 + Gy^2)
#           梯度方向：θ= arctan(Gy / Gx)
# 3) 应用非极大值（Non-Maximum Suppression）抑制，以消除边缘检测带来的杂散响应。
#           线性插值法：当前点和梯度方向上，与四周像素围成的正方形的两个交点相比较，若该点为最大则视该点为边缘
#           简化线性插值法：将梯度方向近似为该像素点四周的八个方向之一，梯度方向与边缘垂直
# 4) 应用双阈值（Double-Threshold）检测来确定真实的和潜在的边缘。
#           筛选边缘结果，对已经标记为边缘的结果进行处理：
#               梯度值 > maxVal:           处理为边界
#               minVal < 梯度值 < maxVal:  当该点连接已有边界时，保留，否则舍弃
#               梯度值 < minVal:           舍弃
# 5) 通过抑制孤立的弱边缘最终完成边缘检测。
img = cv2.imread("data\lena.jpg", cv2.IMREAD_GRAYSCALE)

# 两个参数为minVal,maxVal
v1 = cv2.Canny(img, 80, 150)
v2 = cv2.Canny(img, 50, 100)

res = np.hstack((v1, v2))
show(res)

img = cv2.imread("data\car.png", cv2.IMREAD_GRAYSCALE)
v1 = cv2.Canny(img, 120, 250)
v2 = cv2.Canny(img, 50, 100)
res = np.hstack((v1, v2))
show(res)
