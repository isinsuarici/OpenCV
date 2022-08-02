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
cv.imwrite('cropped_image.png', cropped_image)

# img_blur = cv.medianBlur(gray, 5)
# cv.imwrite('img_blur.png', img_blur)
# x, thresh_binary = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)
# cv.imwrite('thresh_binary.png', thresh_binary)

ksize = 9  # Use size that makes sense to the image and fetaure size. Large may not be good.
# On the synthetic image it is clear how ksize affects imgae (try 5 and 50)
sigma = 5  # Large sigma on small features will fully miss the features.
theta = 1 * np.pi / 4  # /4 shows horizontal 3/4 shows other horizontal. Try other contributions
lamda = 1 * np.pi / 4  # 1/4 works best for angled.
gamma = 0.5  # Value of 1 defines spherical. Calue close to 0 has high aspect ratio
# Value of 1, spherical may not be ideal as it picks up features from other regions.
phi = 0  # Phase offset. I leave it to 0. (For hidden pic use 0.8)

kernel = cv.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, phi, ktype=cv.CV_32F)

plt.imshow(kernel)

img = cv.imread('../input_bobins/IMG_4693/thresh_otsu.png')  # USe ksize:15, s:5, q:pi/2, l:pi/4, g:0.9, phi:0.8
plt.imshow(img, cmap='gray')

img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
fimg = cv.filter2D(img, cv.CV_8UC3, kernel)

kernel_resized = cv.resize(kernel, (400, 400))  # Resize image

plt.imshow(kernel_resized)
plt.imshow(fimg, cmap='gray')
cv.imshow('Kernel', kernel_resized)
cv.imshow('Original Img.', img)
cv.imshow('Filtered', fimg)
cv.waitKey()
cv.destroyAllWindows()
