__author__ = 'kate'

import cv2
import numpy as np

KEY_POINT_COUNT = 2000
ROTATE_ANGEL = 45
SCALE = 2
FLANN_INDEX_KDTREE = 0
DISTANCE = 5


def rotateImage(beforeRotate, angel, scale):
    rows, cols = beforeRotate.shape
    transform_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angel, 1. / scale)
    afterRotate = cv2.warpAffine(beforeRotate, transform_matrix, (cols, rows))
    return afterRotate

def drawMatches(img1, kp1, img2, kp2, matches):
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    out = np.zeros((max([rows1, rows2]), cols1 + cols2, 3), np.uint8)
    out[:rows1, :cols1, :] = np.dstack([img1, img1, img1])
    out[:rows2, cols1:cols1 + cols2, :] = np.dstack([img2, img2, img2])
    for match in matches:
        (x1, y1) = kp1[match[0].queryIdx].pt
        (x2, y2) = kp2[match[0].trainIdx].pt

        cv2.circle(out, (int(x1), int(y1)), 4, (255, 0, 0), 1)
        cv2.circle(out, (int(x2) + cols1, int(y2)), 4, (255, 0, 0), 1)

        cv2.line(out, (int(x1), int(y1)), (int(x2) + cols1, int(y2)), (255, 0, 0), 1)
    return out


def getTransformMatrix(centerX, centerY, angel, scale):
    return cv2.getRotationMatrix2D((centerX, centerY), angel, 1. / scale)


inImg = cv2.imread('mandril.bmp', cv2.CV_LOAD_IMAGE_GRAYSCALE)
transformedImage = rotateImage(inImg, ROTATE_ANGEL, 2)

sift = cv2.SIFT(KEY_POINT_COUNT)

keyPointsInitial, desInitial = sift.detectAndCompute(inImg, None)
keyPointsTransformed, desTransformed = sift.detectAndCompute(transformedImage, None)


# flann
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)  # or pass empty dictionary

flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(desInitial, desTransformed, k=2)
findPointsCount = len(matches)

# Need to draw only good matches, so create a mask
matchesMask = [[0, 0] for i in xrange(len(matches))]
for i, (m, n) in enumerate(matches):
    if m.distance < 0.7 * n.distance:
        matchesMask[i] = 1

matches = [m for (i, m) in enumerate(matches) if matchesMask[i] == 1]


col, row = inImg.shape
transformedMat = getTransformMatrix(col / 2, row / 2, ROTATE_ANGEL, SCALE)

# precision
sum = 0
for match in matches:
    x = np.append(keyPointsInitial[match[0].queryIdx].pt, 1)
    transformed_point = np.array([np.dot(x, np.array(transformedMat[0, :])), np.dot(x, np.array(transformedMat[1, :]))])
    distance = np.linalg.norm(transformed_point - keyPointsTransformed[match[0].trainIdx].pt)

    if (distance < DISTANCE):
        sum += 1


print ("precision = ", sum * 100.0 / findPointsCount)

# write images
result_image = drawMatches(inImg, keyPointsInitial, transformedImage, keyPointsTransformed, matches)
initialWithPoints = cv2.drawKeypoints(inImg, keyPointsInitial)
transformedWithPoints = cv2.drawKeypoints(transformedImage, keyPointsTransformed)
cv2.imwrite('initialWithPoints.bmp', initialWithPoints)
cv2.imwrite('transformedWithPoints.bmp', transformedWithPoints)
cv2.imwrite("matches.bmp", result_image)


