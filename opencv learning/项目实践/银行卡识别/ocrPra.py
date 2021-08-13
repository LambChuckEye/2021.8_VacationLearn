import numpy as np
import cv2


def show(img):
    cv2.imshow('xx', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def sort_conts(conts):
    boxes = [cv2.boundingRect(c) for c in conts]
    (conts, boxes) = zip(*sorted(zip(conts, boxes), key=lambda x: x[1][0]))
    return conts, boxes


# 模板轮廓检测
img = cv2.imread('images/ocr_a_reference.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, ref = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY_INV)
_, refCnts, _ = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 轮廓排序
refCnts, _ = sort_conts(refCnts)

# 模板切割
digits = {}  # num:roi
for i, c in enumerate(refCnts):
    x, y, w, h = cv2.boundingRect(c)
    roi = ref[y:y + h, x:x + w]
    roi = cv2.resize(roi, (53, 85))
    digits[i] = roi

# ========================================================================
# 导入信用卡
image = cv2.imread('images/credit_card_03.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 礼帽操作
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (27, 9))
tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)

# 二值图像
_, tophat = cv2.threshold(tophat, 50, 255, cv2.THRESH_BINARY)

# 闭操作
close = cv2.morphologyEx(tophat, cv2.MORPH_CLOSE, rectKernel)

# 轮廓检测
_, cnts, _ = cv2.findContours(close.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

cur_image = image.copy()
cv2.drawContours(cur_image, cnts, -1, (0, 0, 255), 2)

# 轮廓筛选
locs = []
for i, c in enumerate(cnts):
    x, y, w, h = cv2.boundingRect(c)
    ar = w / float(h)
    if 3.3 < ar <= 3.5:
        locs.append((x, y, w, h))

locs = sorted(locs, key=lambda x: x[0])

# 分割数字并匹配
output = []
for i, (x, y, w, h) in enumerate(locs):
    groupOutput = []
    # 截取区域
    group = gray[y - 5:y + h + 5, x - 5:x + w + 5]
    # 分割数字
    group = cv2.threshold(group, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    _, digitscnts, _ = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    digitscnts = sort_conts(digitscnts)[0]
    # 对每一数字匹配数值
    for c in digitscnts:
        # 截取数字
        x1, y1, w1, h1 = cv2.boundingRect(c)
        roi = group[y1:y1 + h1, x1:x1 + w1]
        roi = cv2.resize(roi, (53, 85))

        # 模板匹配
        scores = []
        for (digit, digitROI) in digits.items():
            result = cv2.matchTemplate(roi, digitROI, cv2.TM_CCOEFF)
            _, score, _, _ = cv2.minMaxLoc(result)
            scores.append(score)
        groupOutput.append(str(np.argmax(scores)))
    cv2.rectangle(image, (x - 5, y - 5), (x + w + 5, y + h + 5), (0, 255, 0), 2)
    cv2.putText(image, "".join(groupOutput), (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
    output.extend(groupOutput)
print(output)
show(image)
