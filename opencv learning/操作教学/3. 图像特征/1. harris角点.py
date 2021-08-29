import cv2
import numpy as np

img = cv2.imread('data/chessboard.jpg')
print('img.shape:', img.shape)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)
# cv2.cornerHarris()
#       img： 数据类型为 ﬂoat32 的入图像
#       blockSize： 角点检测中指定区域的大小
#       ksize： Sobel求导中使用的窗口大小
#       k： 取值参数为 [0,04,0.06]
dst = cv2.cornerHarris(gray, 2, 3, 0.04)
print('dst.shape:', dst.shape)
cv2.imshow('dst', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 输出可能角点
img[dst > 0.01 * dst.max()] = [0, 0, 255]
cv2.imshow('dst', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
