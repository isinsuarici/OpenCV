import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread("../input_pictures/input_person.png")
mask = np.zeros(img.shape[:2], np.uint8)

bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)

rect = (140, 10, 110, 150)  # x,y,w,h
cv.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_RECT)

mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

img = img * mask2[:, :, np.newaxis]
plt.imshow(img), plt.show()
cv.imwrite("result.png", img)
