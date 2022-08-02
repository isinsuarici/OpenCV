import cv2 as cv

"""  CANNY EDGE DETECTION **************************************************************** """
# Öncelikle noise reduction yapılır çünkü canny edge detection, gürültüye karşı duyarlıdır.
# Bu yüzden ilk adımda gauss filtresi ile gürültü kaldırılır.
# Ardından görüntünün intensity gradienti bulunur.
# smoothened image, sobel kerneliyle hem yatay hem dikeyde filtrelenir.
# non-maximum suppression yapılır ardından hysteresis thresholding ?
# opencv tüm bu işlemlerin cv.Canny fonksiyonu ile yapılabilmesini sağlar.

gray = cv.imread("../input_pictures/input_agac.png", 0)
edges = cv.Canny(gray, 100, 200)  # 2.arg=minVal, 3.arg=maxVal
cv.imshow('edges', edges)
cv.waitKey(0)
