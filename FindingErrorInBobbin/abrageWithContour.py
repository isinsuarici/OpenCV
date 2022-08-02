import cv2 as cv
import math

img = cv.imread("../input_bobins/pure/9.png")
cv.imshow("Original image", img)

gray = cv.cvtColor(img, cv.COLOR_RGBA2GRAY)
img_forth = cv.medianBlur(gray, 3)  # noiseu azaltmak için
thresh_adaptive_gaussian = cv.adaptiveThreshold(img_forth, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 29, 2)
cv.imshow('thresh_adaptive_gaussian.png', thresh_adaptive_gaussian)
contours, hierarchy = cv.findContours(thresh_adaptive_gaussian, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

cv.drawContours(img, contours, -1, (0, 255, 0), 3)
cv.imshow('Contours', img)
cv.waitKey(0)
