import time

import keyboard
import numpy as np
import cv2
import pyautogui
import ctypes
import win32api
import win32con
import win32gui
import pydirectinput

CUT_SIZE = 650


# 画图
def show(img):
    cv2.imshow('name', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def pre(img):
    h, w = img.shape[:2]
    # 裁剪
    img = img[int(h / 2 - CUT_SIZE):int(h / 2 + CUT_SIZE), int(w / 2 - CUT_SIZE):int(w / 2 + CUT_SIZE)]
    return img


def checkCircle(img):
    # 边缘检测
    edged = cv2.Canny(img, 75, 200)
    # 轮廓检测
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
    if cnts is None:
        return None
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if np.abs(w - h) < 50 and cv2.contourArea(c) > 5000:
            # 返回第一个坐标
            return int(x + (w / 2)), int(y + (h / 2))

    return None


def autoCheck(x, y, h, w):
    x = int(x + w / 2 - CUT_SIZE)
    y = int(y + h / 2 - CUT_SIZE)

    currentX, currentY = pydirectinput.position()
    pydirectinput.moveRel(x - currentX, y - currentY, relative=True)

    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
    time.sleep(0.0001)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)


def catchScreen():
    return np.array(pyautogui.screenshot())


# def move(x, y):
#     pydirectinput.moveRel(x, y, relative=True)  # Move the mouse to the x, y coordinates 100, 150.
#     # pydirectinput.moveTo(x, y, relative=True)  # Move the mouse to the x, y coordinates 100, 150.
#     pydirectinput.click()


def drawCircle(img, x, y):
    cv2.circle(img, (x, y), 10, (0, 255, 0), 10)
    show(img)


if __name__ == '__main__':

    while True:
        keyboard.wait('9')
        while True:
            start = time.time()
            img = catchScreen()
            temp = checkCircle(pre(img))
            # temp = checkCircle(img)
            if temp is None:
                break
            else:
                x, y = temp
            h, w = img.shape[:2]
            autoCheck(x, y, h, w)
            print(time.time() - start)
            # drawCircle(img, x, y)
