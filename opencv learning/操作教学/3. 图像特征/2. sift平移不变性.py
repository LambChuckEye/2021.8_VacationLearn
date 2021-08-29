import cv2
import numpy as np

img = cv2.imread('data/test_1.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 实例化sift算法
sift = cv2.xfeatures2d.SIFT_create()
# 传入图像，获取关键点
kp = sift.detect(gray, None)
# 绘制关键点
img = cv2.drawKeypoints(gray, kp, img)

cv2.imshow('drawKeypoints', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 计算特征，返回关键点，该关键点的特征
kp, des = sift.compute(gray, kp)
