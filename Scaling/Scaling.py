import cv2 as cv

""" SCALING **************************************************************** """
img = cv.imread('../input_pictures/input_rende.png')

print(img.shape)
img_resize = cv.resize(img, None, 2, 2, cv.INTER_CUBIC)  # INTER_CUBIC => zoom
cv.imwrite('img_resize.png', img_resize)
print(img_resize.shape)

print(img.shape)
height, width = img.shape[:2]
dec = cv.resize(img, (int(50 * width / 100), int(50 * height / 100)), interpolation=cv.INTER_AREA)
# interpolasyon parametre olarak verilmezse default INTER_LINEAR kullanıyor.
# görüntü büyütülürken arada boş pikseller kalıyor o kısımlara ne konulacağına interpolasyon ile karar veriliyor.
cv.imwrite('img_resize2.png', dec)
print(dec.shape)

cv.imshow("img",dec)
cv.waitKey(0)
cv.destroyAllWindows()

change_height = cv.resize(img, (width, int(50 * height / 100)), interpolation=cv.INTER_AREA)
cv.imshow("img", change_height)
cv.waitKey(0)
cv.destroyAllWindows()

change_weight = cv.resize(img, (int(50 * width / 100), height), interpolation=cv.INTER_AREA)
cv.imshow("img", change_weight)
cv.waitKey(0)
cv.destroyAllWindows()

change_withSpecs = cv.resize(img, (100, 100), interpolation=cv.INTER_AREA)
cv.imshow("img", change_withSpecs)
cv.waitKey(0)
cv.destroyAllWindows()
