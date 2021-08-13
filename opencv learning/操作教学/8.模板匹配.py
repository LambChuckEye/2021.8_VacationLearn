import cv2
import numpy as np

# 图片展示
from matplotlib import pyplot as plt


def show(img):
    cv2.imshow('1', img)
    # 0：任意键退出
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 模板匹配:
#       在图像中寻找和模板最相似的结果
# 模板匹配和卷积原理很像，模板在原图像上从原点开始滑动，计算模板与（图像被模板覆盖的地方）的差别程度
# 这个差别程度的计算方法在opencv里有6种，然后将每次计算的结果放入一个矩阵里，作为结果输出。
# 假如原图形是AxB大小，而模板是axb大小，则输出结果的矩阵是(A-a+1)x(B-b+1)
img = cv2.imread('data/lena.jpg', 0)
template = cv2.imread('data/face.jpg', 0)
# 获取模板大小，方便之后画图
h, w = template.shape[:2]

# cv2.matchTemplate(img, template, method)
# method，匹配方法：
#   TM_SQDIFF：计算平方不同，计算出来的值越小，越相关
#   TM_CCORR：计算相关性，计算出来的值越大，越相关
#   TM_CCOEFF：计算相关系数，计算出来的值越大，越相关
#   带归一化的方法更好一些
#   TM_SQDIFF_NORMED：计算归一化平方不同，计算出来的值越接近0，越相关
#   TM_CCORR_NORMED：计算归一化相关性，计算出来的值越接近1，越相关
#   TM_CCOEFF_NORMED：计算归一化相关系数，计算出来的值越接近1，越相关
# 返回的是图像中各个位置的相关性
res = cv2.matchTemplate(img, template, cv2.TM_SQDIFF)
print(img.shape, template.shape, res.shape)

# 获取相关性结果中的最大值和最小值，以及其位置，根据该位置可以确定匹配结果位置
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

# 对于不同方法进行模版匹配测试
methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
           'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

for meth in methods:
    img2 = img.copy()

    # 匹配方法的真值，eval执行字符串，即获取cv2.TM_xxx
    method = eval(meth)
    print(method)
    res = cv2.matchTemplate(img, template, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    # 不同方法的相关性评价方式不同
    # 如果是平方差匹配TM_SQDIFF或归一化平方差匹配TM_SQDIFF_NORMED，取最小值
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    # 画矩形
    cv2.rectangle(img2, top_left, bottom_right, 255, 2)

    plt.subplot(121), plt.imshow(res, cmap='gray')
    plt.xticks([]), plt.yticks([])  # 隐藏坐标轴
    plt.subplot(122), plt.imshow(img2, cmap='gray')
    plt.xticks([]), plt.yticks([])
    plt.suptitle(meth)
    plt.show()

# =============================================
# 多个目标的模板匹配
img_rgb = cv2.imread('data/mario.jpg')
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
template = cv2.imread('data/mario_coin.jpg', 0)
h, w = template.shape[:2]

res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
threshold = 0.8
# 取匹配程度大于80%的坐标
loc = np.where(res >= threshold)
# loc为 y,x 排列,需要进行反向遍历
for pt in zip(*loc[::-1]):  # *号表示可选参数
    bottom_right = (pt[0] + w, pt[1] + h)
    cv2.rectangle(img_rgb, pt, bottom_right, (0, 0, 255), 2)

cv2.imshow('img_rgb', img_rgb)
cv2.waitKey(0)
