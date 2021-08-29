from Stitcher import Stitcher
import cv2

# 读取拼接图片
imageA = cv2.imread("image/l2.jpg")
imageB = cv2.imread("image/r2.jpg")

# 把图片拼接成全景图
stitcher = Stitcher()
(result, vis) = stitcher.stitch([imageA, imageB], showMatches=True)


def show(name, img):
    cv2.namedWindow(name, 0)
    cv2.imshow(name, img)


# 显示所有图片
show("Image A", imageA)
show("Image B", imageB)
show("Keypoint Matches", vis)
show("Result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
