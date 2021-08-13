import cv2
import numpy as np


# 图片展示
def show(img):
    cv2.imshow('1', img)
    # 0：任意键退出
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ====================================================
# Sobel算子：
# dst = cv2.Sobel(src, ddepth, dx, dy, ksize)
#   ddepth:       图像的深度,一般使用-1表示自动填充，
#                   使用cv2.CV_64F时，可以存在负数。
#   dx和dy:       分别表示水平和竖直方向
#   ksize:        Sobel算子的大小
# --------------------------------------------
#  | -1  0  +1 |
#  | -2  0  +2 |  * A   ==>  水平梯度
#  | -1  0  +1 |
# ---------------------------------------------
#  | -1  -2  -1 |
#  |  0   0   0 |  * A  ==>  垂直梯度
#  | +1  +2  +1 |

img = cv2.imread('data\lena.jpg', cv2.IMREAD_GRAYSCALE)

# 只算水平梯度
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
# 可以看到负梯度被截断为0，所以需要对梯度取绝对值
sobelx = cv2.convertScaleAbs(sobelx)

# 取竖直梯度，并求绝对值
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
sobely = cv2.convertScaleAbs(sobely)

# 最后求和，获得完整梯度
sobelxy = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)

# 直接使用cv2.Sobel(img, cv2.CV_64F, 1, 1, ksize=3)进行计算的效果，不如分别计算后相加的效果好
sobelxy1 = cv2.Sobel(img, cv2.CV_64F, 1, 1, ksize=3)
sobelxy1 = cv2.convertScaleAbs(sobelxy1)
show(np.hstack((sobelxy, sobelxy1)))

# ==================================
# Scharr算子：
# dst = cv2.Scharr(src, ddepth, dx, dy, ksize)
# --------------------------------------------
#  | -3   0  +3  |
#  | -10  0  +10 |  * A   ==>  水平梯度
#  | -3   0  +3  |
# ---------------------------------------------
#  | -3  -10  -3 |
#  |  0   0    0 |  * A  ==>  垂直梯度
#  | +3  +10  +3 |
scharrxy = cv2.addWeighted(cv2.convertScaleAbs(cv2.Scharr(img, cv2.CV_64F, 1, 0)),
                           0.5,
                           cv2.convertScaleAbs(cv2.Scharr(img, cv2.CV_64F, 0, 1)),
                           0.5,
                           0)
show(scharrxy)

# ==================================
# laplacian算子：拉普拉斯
#  | 0   1   0 |
#  | 1  -4   1 |  * A
#  | 0   1   0 |
# |左-x| + |右-x| + |上-x| + |下-x|

img = cv2.imread('data\lena.jpg', cv2.IMREAD_GRAYSCALE)
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
sobelx = cv2.convertScaleAbs(sobelx)
sobely = cv2.convertScaleAbs(sobely)
sobelxy = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)

scharrx = cv2.Scharr(img, cv2.CV_64F, 1, 0)
scharry = cv2.Scharr(img, cv2.CV_64F, 0, 1)
scharrx = cv2.convertScaleAbs(scharrx)
scharry = cv2.convertScaleAbs(scharry)
scharrxy = cv2.addWeighted(scharrx, 0.5, scharry, 0.5, 0)

# 无需分别求xy，一步即可
laplacian = cv2.Laplacian(img, cv2.CV_64F)
laplacian = cv2.convertScaleAbs(laplacian)

res = np.hstack((sobelxy, scharrxy, laplacian))
show(res)
