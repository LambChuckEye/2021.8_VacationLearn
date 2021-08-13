import numpy as np
import cv2


# 边界排列
def sort_contours(cnts, method="left-to-right"):
    reverse = False
    i = 0

    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True

    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]  # 用一个最小的矩形，把找到的形状包起来x,y,h,w
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))

    return cnts, boundingBoxes


# 画图
def show(img):
    cv2.imshow('name', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


img = cv2.imread('images/ocr_a_reference.png')
# 灰度
ref = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 二值
threshold, ref = cv2.threshold(ref, 10, 255, cv2.THRESH_BINARY_INV)
# 轮廓检测
ref_, refCnts, hierarchy = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img, refCnts, -1, (0, 0, 255), 3)
# 轮廓排序
refCnts = sort_contours(refCnts, method="left-to-right")[0]

# 数字和模板字典
digits = {}

for i, c in enumerate(refCnts):
    x, y, w, h = cv2.boundingRect(c)
    roi = ref[y:y + h, x:x + w]
    roi = cv2.resize(roi, (54, 85))
    digits[i] = roi

# 初始化卷积核
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (27, 9))

image = cv2.imread('images/credit_card_01.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 礼帽操作
tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)
threshold, tophat = cv2.threshold(tophat, 50, 255, cv2.THRESH_BINARY)

# 闭操作
close = cv2.morphologyEx(tophat, cv2.MORPH_CLOSE, rectKernel)

# 轮廓检测
_, cnts, _ = cv2.findContours(close.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cur_image = image.copy()
cv2.drawContours(cur_image, cnts, -1, (0, 0, 255), 3)
show(cur_image)

# 获取数字区域
locs = []
for i, c in enumerate(cnts):
    x, y, w, h = cv2.boundingRect(c)
    ar = w / float(h)
    if 3.21 < ar < 3.55:
        locs.append((x, y, w, h))

locs = sorted(locs, key=lambda x: x[0])

output = []
for i, (x, y, w, h) in enumerate(locs):
    groupOutput = []
    # 截取区域
    group = gray[y - 5:y + h + 5, x - 5:x + w + 5]
    # 分割数字
    group = cv2.threshold(group, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    _, digitsCnts, _ = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    digitsCnts = sort_contours(digitsCnts, method='left-to-right')[0]
    # 匹配数值
    for c in digitsCnts:
        x1, y1, w1, h1 = cv2.boundingRect(c)
        roi = group[y1:y1 + h1, x1:x1 + w1]
        roi = cv2.resize(roi, (54, 85))
        scores = []
        for (digit, digitROI) in digits.items():
            # 模板匹配，最大相关，取最大值
            result = cv2.matchTemplate(roi, digitROI, cv2.TM_CCOEFF)
            _, score, _, _ = cv2.minMaxLoc(result)
            scores.append(score)
        groupOutput.append(str(np.argmax(scores)))
    cv2.rectangle(image, (x - 5, y - 5), (x + w + 5, y + h + 5), (0, 0, 255), 1)
    cv2.putText(image, "".join(groupOutput), (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
    output.extend(groupOutput)

print("Credit Card #: {}".format("".join(output)))
show(image)