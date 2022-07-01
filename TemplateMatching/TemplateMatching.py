import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# Bir görüntüdeki nesneleri bulmak için kullanılır.
# Önemli metodları:  cv.matchTemplate(), cv.minMaxLoc()
# Daha büyük bir görüntüde bir şablon görüntüsünün konumunu bulmak için kullanılır.
# cv.matchTemplate() ile şablon, inputun üzerinde kaydırılır ve karşılaştırılır.
# (CNN gibi bu kısım)
# her piksel, o pikselin komşularının template ile ne kadar eşleştiğini gösteren
# bir grayscale görüntü döndürür


img = cv.imread('../input_pictures/input_chineese.png', 0)
img2 = img.copy()
temp = cv.imread("../input_pictures/input_chineeseperson.png", 0)
w, h = temp.shape[::-1]

methods = ["cv.TM_CCORR", "cv.TM_CCORR_NORMED",
           "cv.TM_CCOEFF", "cv.TM_CCOEFF_NORMED",
           "cv.TM_SQDIFF", "cv.TM_SQDIFF_NORMED"]

for m in methods:
    img = img2.copy()
    m = eval(m)

    ret = cv.matchTemplate(img, temp, m)
    min_val, max_mal, min_loc, max_loc = cv.minMaxLoc(ret)

    if m in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv.rectangle(img, top_left, bottom_right, 255, 2)
    # rectangle params=> input img, vertex, opposite vertex,
    # color, thickness(negatif değerler, doldurulmuş dikdörtgen çizer),
    # line type, shift

    plt.subplot(121), plt.imshow(ret, cmap='gray')
    plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(img, cmap='gray')
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    plt.suptitle(m)
    plt.show()
