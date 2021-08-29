import cv2
import numpy as np
import matplotlib.pyplot as plt


def cv_show(img, name):
    cv2.imshow(name, img)
    cv2.waitKey()
    cv2.destroyAllWindows()


# 1. 傅里叶变换
# 高频区域：边界
# 低频区域：非边界
# 低通滤波器：只保留低频，图片模糊
# 高通滤波器：只保留高频，图像细节增强

# opencv中主要就是cv2.dft()和cv2.idft()，输入图像需要先转换成np.float32 格式。
# 得到的结果中频率为0的部分会在左上角，通常要转换到中心位置，可以通过shift变换来实现。
# cv2.dft()返回的结果是双通道的（实部，虚部），通常还需要转换成图像格式才能展示（0,255）。

img = cv2.imread('data/lena.jpg', 0)

# 转换格式
img_float32 = np.float32(img)
# 傅里叶变换
dft = cv2.dft(img_float32, flags=cv2.DFT_COMPLEX_OUTPUT)
# 低频值转换到中间，高频转换到四周
dft_shift = np.fft.fftshift(dft)

# 转换成灰度图能表示的形式
res = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])
# 将数值映射到0-255之间
magnitude_spectrum = 20 * np.log(res)

plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()

# 2. 手写滤波器
# 计算图像的中心位置
rows, cols = img.shape
crow, ccol = int(rows / 2), int(cols / 2)  # 中心位置

# 低通滤波
# 使用掩码进行滤波，即掩码过后只保留中心区域，去除四周区域
# mask = np.zeros((rows, cols, 2), np.uint8)
# mask[crow - 30:crow + 30, ccol - 30:ccol + 30] = 1

# 高通滤波掩码
mask = np.ones((rows, cols, 2), np.uint8)
mask[crow-30:crow+30, ccol-30:ccol+30] = 0

# 使用掩码
fshift = dft_shift * mask
# 恢复低频至左上角
f_ishift = np.fft.ifftshift(fshift)
# 反傅里叶变换，恢复为图片
img_back = cv2.idft(f_ishift)
# 转换成灰度图能表示的形式
img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(img_back, cmap='gray')
plt.title('Result'), plt.xticks([]), plt.yticks([])

plt.show()


