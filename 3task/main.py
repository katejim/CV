__author__ = 'kate'
import numpy as np
import cv2


inImg = cv2.imread("mandril.bmp", cv2.CV_LOAD_IMAGE_GRAYSCALE)


def furieTransform(size):
    dftImage = cv2.dft(np.float32(inImg))
    dftShiftImage = np.fft.fftshift(dftImage)
    rows, cols = inImg.shape
    centralRow, centralCol = rows / 2, cols / 2
    mask = np.ones((rows, cols), np.uint8)
    mask[centralRow - size:centralRow + size, centralCol - size:centralCol + size] = 0
    fshift = dftShiftImage * mask
    fIShift = np.fft.ifftshift(fshift)
    imgResult = cv2.idft(fIShift)
    cv2.imwrite("out_furie.bmp", imgResult)


def laplacian():
    laplacian = cv2.Laplacian(cv2.bitwise_not(inImg, inImg), cv2.CV_32F, ksize=7)
    cv2.imwrite("out_laplacian.bmp", laplacian)


furieTransform(30)
laplacian()
print("Transforms write")

