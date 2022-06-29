import cv2 as cv


""" Color changing - Grayscale ****************************************************************"""

img = cv.imread('../input_pictures/input_rende.png')
gray2 = cv.cvtColor(img, cv.COLOR_RGBA2GRAY)  # color to color
cv.imwrite('../deneme.png', gray2)
# veya görseli alırken parametrede 0 vererek direkt grayscale yapabiliriz:

gray = cv.imread("../input_pictures/input_rende.png", 0)
cv.imwrite('gray.png', gray)