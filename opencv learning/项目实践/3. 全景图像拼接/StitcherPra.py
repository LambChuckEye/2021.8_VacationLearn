import numpy as np
import cv2


def cv_show(name, img):
    cv2.namedWindow(name, 0)
    cv2.imshow(name, img)


# 暴力匹配并输出变换矩阵
def matchKeypoints(kpsA, kpsB, fearuresA, featuresB, ratio, rthreshold):
    # 构造暴力匹配器
    matcher = cv2.BFMatcher()
    # 对sift描述符执行匹配
    knnMatches = matcher.knnMatch(fearuresA, featuresB, 2)

    # 对匹配结果进行筛选
    matches = []
    for m in knnMatches:
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            # query -> A, train -> B
            matches.append((m[0].queryIdx, m[0].trainIdx))

    # 使用RANSAC方法计算变换矩阵
    if len(matches) > 4:
        ptsA = np.float32([kpsA[i] for (i, _) in matches])
        ptsB = np.float32([kpsB[i] for (_, i) in matches])
        (H, status) = cv2.findHomography(ptsB, ptsA, cv2.RANSAC, rthreshold)
        return (matches, H, status)
    return None


# 计算关键点及其描述符
def detectAndDescribe(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 生成sift构造器
    des = cv2.xfeatures2d.SIFT_create()
    # 计算关键点并计算其描述符
    (kps, features) = des.detectAndCompute(image, None)
    kps = np.float32([kp.pt for kp in kps])
    return (kps, features)

# 绘制匹配点连线
def drawMatches(imageA, imageB, kpsA, kpsB, matches, status):
    (hA, wA) = imageA.shape[0:2]
    (hB, wB) = imageB.shape[0:2]
    # 构造底图
    vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
    vis[0:hA, 0:wA] = imageA
    vis[0:hB, wA:] = imageB

    # 画线
    for ((queryIdx, trainIdx), s) in zip(matches, status):
        if s == 1:
            ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
            # 右图点需要加上左图的宽度
            ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
            # 画线
            cv2.line(vis, ptA, ptB, (0, 255, 0), 1)
    return vis

# 拼接主方法
def stitch(images, ratio=0.75, rthreshold=4.0):
    (imageA, imageB) = images
    # 获得关键点和描述符
    (kpsA, featuresA) = detectAndDescribe(imageA)
    (kpsB, featuresB) = detectAndDescribe(imageB)

    # 获得变换矩阵
    M = matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio, rthreshold)

    if M is None:
        return None

    (matches, H, status) = M
    # 空间变换
    result = cv2.warpPerspective(imageB, H, (imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
    # 填充左图
    result[0:imageA.shape[0], 0:imageA.shape[1]] = imageA
    # 画线
    vis = drawMatches(imageA, imageB, kpsA, kpsB, matches, status)

    return result, vis


if __name__ == '__main__':
    imageA = cv2.imread("image/l2.jpg")
    imageB = cv2.imread("image/r2.jpg")
    result, vis = stitch([imageA, imageB])
    cv_show('imageA', imageA)
    cv_show('imageB', imageB)
    cv_show('vis', vis)
    cv_show('result', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
