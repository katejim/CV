__author__ = 'kate'

import numpy as np
import cv2


def showImg(name, mat):
    cv2.imshow(name, mat)
    key = cv2.waitKey(0)
    if key == 27:  # wait for ESC key to exit
        cv2.destroyWindow(name)


def saveImg(name, mat):
    cv2.imwrite(name, mat)


def saveBinImg(binImg):
    saveImg("rezBin.bmp", binImg)


def saveCounturedImg(inImg, binImg):
    h, w = binImg.shape
    mask = np.zeros((h + 2, w + 2), np.uint8)
    x, y = 0, 0
    while y < h:
        x = 0
        while x < w:
            if (binImg[y][x] == 255):

                retval, rect = cv2.floodFill(binImg, mask, (x, y), (0, 255, 255))
                if (retval != 0):
                        cv2.rectangle(inImg, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (255, 0, 0), 2)
                        x =rect[0] + rect[2] + 1
                else:
                    x += 1
            else:
                x += 1
        y += 1
    showImg("Contured image", inImg)
    saveBinImg(inImg)
    print("done")


def createBinaryImg(inImg):
    blurImg = cv2.GaussianBlur(inImg, (21, 21), 0)
    grayImg = cv2.cvtColor(blurImg, cv2.COLOR_BGR2GRAY)

    laplacian = cv2.Laplacian(grayImg, cv2.CV_32F, None, ksize=17, scale=1)
    threshed_image = cv2.threshold(laplacian, 0, 255, cv2.THRESH_BINARY)[1]
    threshed_image = cv2.convertScaleAbs(threshed_image)
    opening = cv2.morphologyEx(threshed_image, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    saveBinImg(opening)
    return opening


def readImg(name):
    return cv2.imread(name, 1)


inImg = readImg("text.bmp")
binImg = createBinaryImg(inImg)
saveCounturedImg(inImg, binImg)
