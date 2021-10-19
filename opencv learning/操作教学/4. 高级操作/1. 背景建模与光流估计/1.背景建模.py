import numpy as np
import cv2

# 经典的测试视频
cap = cv2.VideoCapture('data/test.avi')
# 形态学操作需要使用
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
# 创建混合高斯模型用于背景建模
fgbg = cv2.createBackgroundSubtractorMOG2()

while (True):
    ret, frame = cap.read()
    # 应用高斯混合背景建模
    fgmask = fgbg.apply(frame)
    # 形态学开运算去噪点
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    # 寻找视频中的轮廓
    im, contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)
    index = 0
    for c in contours:
        # 计算各轮廓的周长
        perimeter = cv2.arcLength(c, True)
        # 获取周长较大的目标
        if perimeter > 188:
            # 获取外接矩形
            x, y, w, h = cv2.boundingRect(c)
            # 画出这个矩形
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            index = index + 1
    cv2.putText(frame, "Available: %d persons" % index, (30, 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2)
    cv2.imshow('frame', frame)
    cv2.imshow('fgmask', fgmask)
    k = cv2.waitKey(150) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
