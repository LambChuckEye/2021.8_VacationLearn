import cv2  # opencv读取的格式是BGR
import numpy as np
import matplotlib.pyplot as plt  # Matplotlib是RGB


# 图片展示
def show(img):
    cv2.imshow('1', img)
    # 0：任意键退出
    cv2.waitKey(0)
    cv2.destroyAllWindows()


img = cv2.imread('data\cat.jpg')
# 转换图像色彩
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ===========================================================
# 图像阈值
# ret, dst = cv2.threshold(src, thresh, maxval, type)
#    src：      输入图，只能输入单通道图像，通常来说为灰度图
#    dst：      输出图
#    thresh：   阈值
#    maxval：   当像素值超过了阈值（或者小于阈值，根据type来决定），所赋予的值
#    type：     二值化操作的类型，包含以下5种类型
#           cv2.THRESH_BINARY 超过阈值部分取maxval（最大值），否则取0
#           cv2.THRESH_BINARY_INV THRESH_BINARY的反转
#           cv2.THRESH_TRUNC 大于阈值部分设为阈值，小于部分不变
#           cv2.THRESH_TOZERO 大于阈值部分不改变，小于部分设为0
#           cv2.THRESH_TOZERO_INV THRESH_TOZERO的反转

ret, thresh1 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
ret, thresh2 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY_INV)
ret, thresh3 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_TRUNC)
ret, thresh4 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_TOZERO)
ret, thresh5 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_TOZERO_INV)

titles = ['Original Image', 'BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV']
images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]

for i in range(6):
    plt.subplot(2, 3, i + 1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()

# ====================================
# 图像平滑
img = cv2.imread('data\lenaNoise.png')

# 均值滤波
# 简单的平均卷积操作，对像素点的3x3区域求平均值赋予该点
blur = cv2.blur(img, (3, 3))

# 方框滤波
# 基本和均值一样，区别是可以选择归一化。不进行归一化指，3x3区域不求平均值，越界值取255
box = cv2.boxFilter(img, -1, (3, 3), normalize=False)

# 高斯滤波
# 高斯模糊的卷积核里的数值是满足高斯分布，相当于更重视中间的
aussian = cv2.GaussianBlur(img, (5, 5), 1)

# 中值滤波
# 相当于用卷积中的中值代替该像素点
median = cv2.medianBlur(img, 5)  # 中值滤波

# 拼接图像
res = np.hstack((img, blur, box, aussian, median))
show(res)
