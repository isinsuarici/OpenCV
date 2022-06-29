import cv2 as cv
import numpy as np

img = cv.imread('../input_pictures/input_rende.png')
height, width = img.shape[:2]

""" TRANSLATION **************************************************************** """
M = np.float32([[1, 0, 50], [0, 1, 150]])
dist = cv.warpAffine(img, M, (width, height))  # parametreler = image, kaydırma miktarı, image boyutu
cv.imshow("img",dist)
cv.waitKey(0)
cv.destroyAllWindows()

""" ROTATION **************************************************************** """
M = cv.getRotationMatrix2D(((width - 1) / 2.0, (width - 1) / 2.0), 90, 1)
# getRotationMatrix2D parametreleri = döndürme merkezi, döndürülecek açı ( pozitifse saat yönünün tersinde ), scale
dist_rotate = cv.warpAffine(img, M, (height, width))
cv.imshow("img",dist_rotate)
cv.waitKey(0)
cv.destroyAllWindows()