# 导入工具包
import numpy as np
import argparse
import cv2


# 画图
import pytesseract


def show(img):
    cv2.imshow('name', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 更改图像尺寸
def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        # 缩放比例
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized


# 左上，右上，右下，左下顺序排序目标点
def order_points(pts):
    # 一共4个坐标点
    rect = np.zeros((4, 2), dtype="float32")

    # 按顺序找到对应坐标0123分别是 左上，右上，右下，左下
    # 计算左上，右下
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # 计算右上和左下
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


# 透视变换
def four_point_transform(image, pts):
    # 获取输入坐标点
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # 勾股定理计算输入的w和h值
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    # 选取其中较大的值作为变换后的w
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # 变换后对应坐标位置，长宽 -1 增加容错
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # 获得变换矩阵
    M = cv2.getPerspectiveTransform(rect, dst)
    # 执行变换
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # 返回变换后结果
    return warped


# 图片预处理：
image = cv2.imread('images/page.jpg')
# 计算图像的缩放比例(原图高 / 缩放后的高)
ratio = image.shape[0] / 500.0
orig = image.copy()

# 缩放图片
image = resize(orig, height=500)

gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
# 高斯模糊去噪点
gray = cv2.GaussianBlur(gray, (5, 5), 0)
# 边缘检测
edged = cv2.Canny(gray, 75, 200)

# 展示预处理结果
print("STEP 1: 边缘检测")
cv2.imshow("Image", image)
cv2.imshow("Edged", edged)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 轮廓检测
cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[1]
# 取前五个最大面积轮廓
cnts = sorted(cnts, key=lambda x: cv2.contourArea(x), reverse=True)[:5]

# 筛选轮廓
for c in cnts:
    # 轮廓周长
    peri = cv2.arcLength(c, True)
    # 做近似多边形， 0.02 * peri 为近似的最大阈值，即近似结果和原始结果的最大距离
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    # 若近似多边形有四个端点，则为矩形，则为所求
    if len(approx) == 4:
        screenCnt = approx
        break

# 绘制
cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
show(image)

# 透视变换，输入原始图像与透视目标矩形的四个端点，记得恢复原图比例
warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
# 二值处理
warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
ref = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

# ocr 提取字符
text = pytesseract.image_to_string(ref)
print(text)
show(ref)
