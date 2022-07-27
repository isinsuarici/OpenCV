import cv2 as cv
import numpy as np

img = cv.imread("../input_bobins/with/1.png")
# sadece 2.görselde algılamıyor. HCT yok tresh ve morph işlemler.
print('Original Dimensions : ', img.shape)
if img.shape[0] > 3000 or img.shape[1] > 3000:
    scale_percent = 10
elif img.shape[0] > 1500 or img.shape[1] > 1500:
    scale_percent = 20
else:
    scale_percent = 60
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)

img = cv.resize(img, dim, interpolation=cv.INTER_AREA)

print('Resized Dimensions : ', img.shape)

cv.imshow("Resized image", img)
cv.waitKey(0)
cv.destroyAllWindows()

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
blur = cv.GaussianBlur(gray, (5, 5), 0)
ret3, thresh_otsu_blur = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
cv.imwrite('thresh_otsu_blur.png', thresh_otsu_blur)

kernel = np.ones((5, 5), np.uint8)  # belki 3,3
erosion = cv.erode(thresh_otsu_blur, kernel, iterations=3)
cv.imwrite('erosion.png', erosion)
kernel2 = np.ones((1, 1), np.uint8)
closing = cv.morphologyEx(erosion, cv.MORPH_CLOSE, kernel2)
cv.imwrite('closing.png', closing)
gradient = cv.morphologyEx(closing, cv.MORPH_GRADIENT, kernel)
cv.imwrite('gradient.png', gradient)

cv.imshow("closing image", gradient)
cv.waitKey(0)
cv.destroyAllWindows()
