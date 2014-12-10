__author__ = 'kate'
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


def createBinaryImg(inImg):
    blurImg = cv2.GaussianBlur(inImg, (39, 21), 0)
    laplacian = cv2.Laplacian(blurImg, cv2.CV_32F, None, ksize=21, scale=1)
    thresh = cv2.threshold(laplacian, 0, 255, cv2.THRESH_BINARY)[1]
    saveBinImg(thresh)
    return thresh

def readImg(name):
    return cv2.imread(name, 1)


inImg = readImg("text.bmp")
binImg = createBinaryImg(inImg)
showImg("RESULT", binImg)


