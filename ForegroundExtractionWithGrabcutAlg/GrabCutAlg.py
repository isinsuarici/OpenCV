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


# import cv2 as cv
# import numpy as np
# from matplotlib import pyplot as plt
#
# img = cv.imread("../input_bobins/G0032623/bobin_1.png")
# print('Original Dimensions : ', img.shape)
# if img.shape[0] > 3000 or img.shape[1] > 3000:
#     scale_percent = 10
# elif img.shape[0] > 1500 or img.shape[1] > 1500:
#     scale_percent = 20
# else:
#     scale_percent = 60
# width = int(img.shape[1] * scale_percent / 100)
# height = int(img.shape[0] * scale_percent / 100)
# dim = (width, height)
#
# img = cv.resize(img, dim, interpolation=cv.INTER_AREA)
#
# mask = np.zeros(img.shape[:2], np.uint8)
#
# bgdModel = np.zeros((1, 65), np.float64)
# fgdModel = np.zeros((1, 65), np.float64)
# w, h, b = img.shape[:]
# rect = (0, 0, w, h)  # x,y,w,h
# cv.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_RECT)
#
# mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
#
# img = img * mask2[:, :, np.newaxis]
# plt.imshow(img), plt.show()
# cv.imwrite("result.png", img)
