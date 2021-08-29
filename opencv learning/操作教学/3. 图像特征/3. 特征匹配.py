import cv2
import numpy as np
import matplotlib.pyplot as plt


def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


img1 = cv2.imread('data/box.png', 0)
img2 = cv2.imread('data/box_in_scene.png', 0)

# Brute-Force蛮力匹配, 直接比较特征点的特征向量
# 构造sift
sift = cv2.xfeatures2d.SIFT_create()
# 检测关键点并计算特征向量
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# 蛮力匹配 crossCheck:只有相互之间均为对方的最短距离时才匹配
bf = cv2.BFMatcher(crossCheck=True)

# 进行匹配（1对1匹配）
matches = bf.match(des1, des2)
# 按照距离（相似度）进行排序
matches = sorted(matches, key=lambda x: x.distance)
# 绘制匹配
img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=2)

cv_show('img3', img3)

# k对最佳匹配
bf = cv2.BFMatcher()
# 一个点与k个点进行对应
matches = bf.knnMatch(des1, des2, k=2)
# 筛选结果
good = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good.append([m])
# 绘制匹配
img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)
cv_show('img3',img3)
plt.figure()
plt.imshow(img3)
plt.show()