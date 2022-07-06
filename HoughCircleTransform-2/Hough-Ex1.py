import cv2 as cv
import numpy as np
import argparse

# konsoldan çalıştırmak için:
# python Hough-Ex1.py -i ../input_pictures/input_money2.png

# minDist çok küçük olursa daire olmayan yerlerde bile daire algılayabiliriz.
# minDist çok büyük olursa bazı daireleri algılayamayabiliriz

# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True, help="Path of image")
# args = vars(ap.parse_args())
#
# img = cv.imread(args["image"])
img = cv.imread("../input_pictures/input_money2.png")
img2 = img.copy()  # bu işlemi yaptığım yere göre outputun size'ı(kB) değişiyor.

img = cv.GaussianBlur(img, (7, 7), 1.5)


img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

circles = cv.HoughCircles(img, cv.HOUGH_GRADIENT, 1.3, 30, param1=60, param2=70)

# param2 küçükse yanlış daireler de algılıyor
circles = np.uint16(np.around(circles))

for c in circles[0, :]:
    cv.circle(img2, (c[0], c[1]), c[2], (0, 255, 0), 2)
    cv.circle(img2, (c[0], c[1]), 1, (0, 0, 255), 3)

cv.imshow("img", img2)
# cv.imshow("img", np.hstack([img, img2]))
cv.waitKey(0)
cv.imwrite("houghcircle-ex1.png", img2)
