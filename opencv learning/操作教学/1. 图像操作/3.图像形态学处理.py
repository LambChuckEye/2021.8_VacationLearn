import cv2
import numpy as np
import matplotlib.pyplot as plt


# 图片展示
def show(img):
    cv2.imshow('1', img)
    # 0：任意键退出
    cv2.waitKey(0)
    cv2.destroyAllWindows()


img = cv2.imread('data\dige.png')

# =========================================================
# 腐蚀操作
# 一般用于二值图像（只有两个颜色）,腐蚀掉图像中的边缘像素点
# 构造腐蚀参考核
kernel = np.ones((3, 3), np.uint8)
erosion = cv2.erode(img, kernel, iterations=1)
show(erosion)
# 若该点卷积范围内存在其他颜色像素点，则将该点腐蚀掉

# 迭代效果
pie = cv2.imread('data\pie.png')
kernel = np.ones((30, 30), np.uint8)
erosion_1 = cv2.erode(pie, kernel, iterations=1)
erosion_2 = cv2.erode(pie, kernel, iterations=2)
erosion_3 = cv2.erode(pie, kernel, iterations=3)
res = np.hstack((erosion_1, erosion_2, erosion_3))
show(res)

# ==============================
# 膨胀操作
# 反向的腐蚀操作
kernel = np.ones((3, 3), np.uint8)
dilate = cv2.dilate(img, kernel, iterations=1)
show(dilate)

# 迭代效果
kernel = np.ones((30, 30), np.uint8)
dilate_1 = cv2.dilate(pie, kernel, iterations=1)
dilate_2 = cv2.dilate(pie, kernel, iterations=2)
dilate_3 = cv2.dilate(pie, kernel, iterations=3)
res = np.hstack((dilate_1, dilate_2, dilate_3))
show(res)

# ========================================
img = cv2.imread('data\\1.png')
# 开运算
# 先腐蚀，在膨胀： 可以去掉图片毛刺
kernel = np.ones((5, 5), np.uint8)
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

# 闭运算
# 先膨胀，再腐蚀： 可以填缝
kernel = np.ones((5, 5), np.uint8)
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

show(np.hstack((img, opening, closing)))

# ===============================================
# 梯度运算
# 膨胀 - 腐蚀: 描边
gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)

# ================================================
# 礼帽运算
# 原始输入-开运算结果: 只留下毛刺
tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
# 黑帽运算
# 闭运算-原始输入： 只留下缝隙
blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)

show(np.hstack((img, opening, closing, gradient, tophat, blackhat)))
