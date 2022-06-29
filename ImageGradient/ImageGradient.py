import cv2 as cv
import numpy as np


"""  IMAGE GRADIENTS **************************************************************** """
# 1.Sobel and Scharr Derivatives
# SOBEL = gaussian smoothing + differentitation
# bu nedenle gürültüye karşı dayanıklı.
# türev yönü dikey veya yatay olarak belirlenebilir.
# sırasıyla yorder,xorder
# ksize= çekirdeğin boyutu

img = cv.imread("../input_pictures/input_sudoku.png", 0)
sobelx = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=5)
sobely = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=5)

cv.imwrite("sobelx.png", sobelx)
cv.imwrite("sobely.png", sobely)

# 2.Laplacian Derivaties
laplace = cv.Laplacian(img, cv.CV_64F)
cv.imwrite("laplace.png", laplace)

# ayrıntılı olarak yeniden bak