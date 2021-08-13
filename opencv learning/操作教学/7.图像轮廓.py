import cv2
import numpy as np


# 图片展示
def show(img):
    cv2.imshow('1', img)
    # 0：任意键退出
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 轮廓检测：
#   必须要在二值图像上进行处理。
# cv2.findContours(img,mode,method)
#   mode:轮廓检索模式
#       RETR_EXTERNAL ：只检索最外面的轮廓；
#       RETR_LIST：检索所有的轮廓，并将其保存到一条链表当中；
#       RETR_CCOMP：检索所有的轮廓，并将他们组织为两层：顶层是各部分的外部边界，第二层是空洞的边界;
#       RETR_TREE：检索所有的轮廓，并重构嵌套轮廓的整个层次;（主要使用这个）
#
#   method:轮廓逼近方法
#       CHAIN_APPROX_NONE：正常画出轮廓。以Freeman链码的方式输出轮廓，所有其他方法输出多边形（顶点的序列）。
#       CHAIN_APPROX_SIMPLE：精简轮廓，只保留终点。压缩水平的、垂直的和斜的部分，也就是，函数只保留他们的终点部分。

img = cv2.imread('data\contours.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 二值处理，图片中非黑即白
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
show(thresh)

# 1.检测轮廓
#   binary：原图
#   contours：轮廓信息
#   hierarchy：层级
binary, contours, hierarchy = cv2.findContours(thresh,
                                               cv2.RETR_TREE,
                                               cv2.CHAIN_APPROX_NONE)

# 2.绘制轮廓
# cv2.drawContours() 会在绘制的同时对原图像进行改变,要使用备份进行绘制
draw_img = img.copy()
# (原图, 轮廓, 第几个轮廓(-1表示全部), 线条颜色(BGR), 线条宽度)
cv2.drawContours(draw_img, contours, -1, (0, 0, 255), 2)
show(draw_img)

# 3.轮廓特征计算
# 取出一个轮廓
cnt = contours[0]
# 计算面积
cv2.contourArea(cnt)
# 计算周长
cv2.arcLength(cnt, True)

# 4.轮廓近似
# 获取更加平滑的轮廓形状，将曲线近似为直线
# 求两点间的曲线距离两点间直线的最大长度，若小于阈值，则可以将该曲线近似为直线
img = cv2.imread('data\contours2.png')
# 灰度图二值处理
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
# 轮廓检测
binary, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
cnt = contours[0]
# 轮廓绘制
draw_img = img.copy()
res = cv2.drawContours(draw_img, [cnt], -1, (0, 0, 255), 2)
show(res)

# 轮廓近似
# 设置阈值为周长的0.15倍
epsilon = 0.05 * cv2.arcLength(cnt, True)
# 近似函数，传入曲线转换为直线的阈值
approx = cv2.approxPolyDP(cnt, epsilon, True)

draw_img = img.copy()
res = cv2.drawContours(draw_img, [approx], -1, (0, 0, 255), 2)
show(res)

# =================================================================
# 5.绘制外接图形（将被检测目标圈起来）
# 绘制边界矩形：
img = cv2.imread('data\contours.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
binary, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
cnt = contours[3]

# 获取该轮廓的外接矩形
x, y, w, h = cv2.boundingRect(cnt)
# 绘制矩形
img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
show(img)

# 绘制外接圆
cnt = contours[0]
# 获取外接圆
(x, y), radius = cv2.minEnclosingCircle(cnt)
center = (int(x), int(y))
radius = int(radius)
# 绘制外接圆
img = cv2.circle(img, center, radius, (0, 255, 0), 2)
show(img)
