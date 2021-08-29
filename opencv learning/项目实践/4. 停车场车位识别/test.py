from PIL import ImageGrab
import numpy as np
import cv2
import keyboard
import time
import pyautogui as pag


while True:
    time.sleep(1)
    x1, y1 = pag.position()
    print(x1, y1)

#
# while True:
#     keyboard.wait('p')
#     im = ImageGrab.grab()  # 获得当前屏幕
#     imm = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)  # 转为opencv的BGR格式
#     cv2.imshow('imm', imm)#显示
#     if cv2.waitKey(1) & 0xFF == ord('q'):  # q键推出
#         break
# cv2.destroyAllWindows()