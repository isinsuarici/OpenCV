import cv2 as cv

img = cv.imread("../input_pictures/input_altigen.png")
cv.imshow("img", img)
cv.waitKey(0)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow("gray", gray)
cv.waitKey(0)
# bounding rectangle
# griye dönüştürdükten sonra threshold uygulama sebebimiz renklerin yoğunluğunu arttırmak
retVal, binary = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)
contours, hierarchy = cv.findContours(binary, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
x, y, w, h = cv.boundingRect(contours[0])
cv.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 3)
cv.imshow("result", img)
cv.waitKey(0)
cv.destroyAllWindows()
cv.imwrite("border.png", img)

# multiple mask                                                  !!!

img_2 = cv.imread("../input_pictures/input_ev.png", 0)
cv.imshow("gray img", img_2)
retVal2, th = cv.threshold(img_2, 127, 255, 0)
contour, hier = cv.findContours(th, 1, 2)
for cnt in contour:
    x, y, w, h = cv.boundingRect(cnt)
    img_2 = cv.rectangle(img_2, (x, y), (x+w, y+h), (0, 255, 0), 2)
cv.imshow("final img", img_2)
cv.waitKey()



