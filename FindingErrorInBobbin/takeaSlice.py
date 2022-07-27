import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread("../input_bobins/with/1.png")

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


# img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
# cv.imshow(" image", img)
# cv.waitKey(0)

h, w, b = img.shape[:]
cropped_image = img[0:int(w / 2), 0:int(h / 2)]
cv.imshow("Resized image", cropped_image)
cv.waitKey(0)
cv.destroyAllWindows()
cv.imwrite('crop.png', cropped_image)

gray = cv.imread("../HoughCircleTransform-2/crop.png", 0)
img_blur = cv.medianBlur(gray, 3)
cv.imwrite('img_blur.png', img_blur)
x, thresh_binary = cv.threshold(img_blur, 90, 255, cv.THRESH_BINARY)
cv.imwrite('thresh_binary.png', thresh_binary)

kernel = np.ones((5, 5), np.uint8)
gradient = cv.morphologyEx(thresh_binary, cv.MORPH_GRADIENT, kernel)
cv.imwrite('dilation.png', gradient)
cv.imshow("dilation image", gradient)
cv.waitKey(0)
cv.destroyAllWindows()