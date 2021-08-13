import cv2
import numpy as np
import matplotlib.pyplot as plt


def cv_show(img, name):
    cv2.imshow(name, img)
    cv2.waitKey()
    cv2.destroyAllWindows()


# 1. 生成直方图，统计0-255范围内，每个像素值在图像中出现的次数
# cv2.calcHist(images,channels,mask,histSize,ranges)
#       images: 原图像图像格式为 uint8 或float32。当传入函数时应 用中括号 [] 括来例如[img]
#       channels: 指定需要统计的通道，应用中括号括起来。
#            如果入图像是灰度图它的值就是 [0]如果是彩色图像 的传入的参数可以是 [0][1][2] 它们分别对应着 BGR。
#       mask: 掩模图像，指定直方图的统计区域。
#            统整幅图像的直方图就把它为 None。但是如 果你想统图像某一分的直方图的你就制作一个掩模图像并 使用它。
#       histSize:BIN 的数目，即直方图的数目，比如10个像素一个直方。也应用中括号括来
#       ranges: 像素值范围常为 [0256]


img = cv2.imread('data/cat.jpg')
color = ('b', 'g', 'r')
# 对于每个通道都做直方图
for i, col in enumerate(color):
    # 绘制直方图
    histr = cv2.calcHist([img], [i], None, [256], [0, 256])
    plt.plot(histr, color=col)
    plt.xlim([0, 256])
plt.show()

# =========================================
# 2. mask操作
# 创建mast，无需通道
mask = np.zeros(img.shape[:2], np.uint8)

# 指定显象范围
mask[100:300, 100:400] = 255

# 使用mask遮罩图像
masked_img = cv2.bitwise_and(img, img, mask=mask)  # 与操作

# 使用mask遮罩直方图
hist_full = cv2.calcHist([img], [0], None, [256], [0, 256])
hist_mask = cv2.calcHist([img], [0], mask, [256], [0, 256])

plt.subplot(221), plt.imshow(img, 'gray')
plt.subplot(222), plt.imshow(mask, 'gray')
plt.subplot(223), plt.imshow(masked_img, 'gray')
plt.subplot(224), plt.plot(hist_full), plt.plot(hist_mask)
plt.xlim([0, 256])
plt.show()

# =========================================
# 3. 直方图均衡化
img = cv2.imread('data/clahe.jpg', 0)  # 0表示灰度图 #clahe
plt.hist(img.ravel(), 256)
plt.show()

# 均衡化
equ = cv2.equalizeHist(img)
plt.hist(equ.ravel(), 256)
plt.show()

res = np.hstack((img, equ))
cv_show(res, 'res')

# 自适应均衡化，分区域进行均衡化，效果更好
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
res_clahe = clahe.apply(img)

res = np.hstack((img, equ, res_clahe))
cv_show(res, 'res')
