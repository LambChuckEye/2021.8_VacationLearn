import cv2  # opencv读取的格式是BGR
import matplotlib.pyplot as plt
import numpy as np

# 图片读取
img = cv2.imread('data/cat.jpg')


# 图片展示
def cv_show(img):
    cv2.imshow('1', img)
    # 0：任意键退出
    cv2.waitKey(0)
    cv2.destroyAllWindows()


cv_show(img)

# 读取图片为灰度图像
img = cv2.imread('data/cat.jpg', cv2.IMREAD_GRAYSCALE)
cv_show(img)
# 保存图像
cv2.imwrite('data/mycat.png', img)

# ============================================================================
# 读取视频
vc = cv2.VideoCapture('data/test.mp4')

# 检查是否打开正确
if vc.isOpened():
    # 逐帧读取视频
    oepn, frame = vc.read()
else:
    open = False

# 显示视频
while open:
    ret, frame = vc.read()
    if frame is None:
        break
    if ret == True:
        cv2.imshow('result', frame)
        if cv2.waitKey(10) & 0xFF == 27:
            break
vc.release()
cv2.destroyAllWindows()

# =====================================
# 图像操作

# 图像截取
img = cv2.imread('data/cat.jpg')
cat = img[0:50, 0:200]
cv_show(cat)

# 颜色通道提取
b, g, r = cv2.split(img)
# 颜色通道合并
img = cv2.merge((r, g, b))
cv_show(img)
# 修改颜色通道
# 只保留R
cur_img = img.copy()
cur_img[:, :, 0] = 0
cur_img[:, :, 1] = 0
cv_show(cur_img)

# 边界填充
top_size, bottom_size, left_size, right_size = (50, 50, 50, 50)
# BORDER_REPLICATE:复制法，直接复制最边缘的像素
replicate = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, borderType=cv2.BORDER_REPLICATE)
# BORDER_REFLECT:反射法，以边缘为轴做反射  fedcba|abcdefg|gfedcb
reflect = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, cv2.BORDER_REFLECT)
# BORDER_REFLECT_101:反射法，去掉边缘的重复像素 fedcb|abcdefg|fedcb
reflect101 = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, cv2.BORDER_REFLECT_101)
# BORDER_WRAP:外包装法 将对边部分平移过来 cdefg|abcdefg|abcde
wrap = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, cv2.BORDER_WRAP)
# BORDER_CONSTANT:填充常数值
constant = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, cv2.BORDER_CONSTANT, value=0)

import matplotlib.pyplot as plt

plt.subplot(231), plt.imshow(img, 'gray'), plt.title('ORIGINAL')
plt.subplot(232), plt.imshow(replicate, 'gray'), plt.title('REPLICATE')
plt.subplot(233), plt.imshow(reflect, 'gray'), plt.title('REFLECT')
plt.subplot(234), plt.imshow(reflect101, 'gray'), plt.title('REFLECT_101')
plt.subplot(235), plt.imshow(wrap, 'gray'), plt.title('WRAP')
plt.subplot(236), plt.imshow(constant, 'gray'), plt.title('CONSTANT')

plt.show()

# 图像叠加
img_cat = cv2.imread('data/cat.jpg')
img_cat2 = img_cat + 10
# 超过255的部分进行取余
cv_show(img_cat + img_cat2)
# 超过255部分保留255
cv_show(cv2.add(img_cat, img_cat2))

# 图像融合
img_cat = cv2.imread('data/cat.jpg')
img_dog = cv2.imread('data/dog.jpg')
# 改变图像像素，cat。shape：（414,500）
img_dog = cv2.resize(img_dog, (500, 414))
# 权重叠加
res = cv2.addWeighted(img_cat, 0.4, img_dog, 0.6, 0)
cv_show(res)

# 图像拉伸
res = cv2.resize(img, (0, 0), fx=3, fy=4)
cv_show(res)