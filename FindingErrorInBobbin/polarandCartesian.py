import cv2
import numpy as np

source = cv2.imread('../input_bobins/with/1.png', 1)

scale_percent = 20
width = int(source.shape[1] * scale_percent / 100)
height = int(source.shape[0] * scale_percent / 100)
dim = (width, height)

img = cv2.resize(source, dim, interpolation=cv2.INTER_AREA)

img = img.astype(np.float32)

value = np.sqrt(((img.shape[0] / 2.0) ** 2.0) + ((img.shape[1] / 2.0) ** 2.0))

polar_image = cv2.linearPolar(img, (img.shape[0] / 2, img.shape[1] / 2), value, cv2.WARP_FILL_OUTLIERS)

polar_image = polar_image.astype(np.uint8)
cv2.imshow("Polar Image", polar_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("sonuc.png", polar_image)


gray = cv2.cvtColor(polar_image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (9, 9), 0)
retVal, thresh_otsu_blur = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
cv2.imwrite('thresh_otsu_blur.png', thresh_otsu_blur)

kernel = np.ones((3, 3), np.uint8)
erosion = cv2.erode(thresh_otsu_blur, kernel, iterations=3)
cv2.imwrite('erosion.png', erosion)
kernel2 = np.ones((1, 1), np.uint8)
closing = cv2.morphologyEx(erosion, cv2.MORPH_CLOSE, kernel2)
cv2.imwrite('closing2.png', closing)
