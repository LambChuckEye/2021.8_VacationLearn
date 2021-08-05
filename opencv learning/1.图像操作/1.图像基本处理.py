import cv2  # opencv读取的格式是BGR
import matplotlib.pyplot as plt
import numpy as np

# 图片读取
img = cv2.imread('cat.jpg')


# 图片展示
def cv_show(name, img):
    cv2.imshow(name, img)
    # 0：任意键退出
    cv2.waitKey(0)
    cv2.destroyAllWindows()


cv_show('1', img)

# 读取图片为灰度图像
img = cv2.imread('cat.jpg', cv2.IMREAD_GRAYSCALE)
cv_show('1', img)

# 保存图像
cv2.imwrite('mycat.png', img)

# 读取视频
vc = cv2.VideoCapture('test.mp4')