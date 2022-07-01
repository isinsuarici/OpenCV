import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# çizgileri belirlemek için kullanıyoruz. Feature extraction yöntemi.
# rho ve theta değerleri var
# rho = doğrunun orijinden uzaklığı
# theta = açı

# teori:
# koordinat noktasında bi çizgi olsun.
# bu çizgi üzerinden iki nokta alalım.
# (bu uzaydaki her nokta, hough uzayında bir çizgiye karşılık gelir.)
# bu iki çizginin hough uzayında ifade ettiği çizgiler mutlaka bir noktada kesişir!
# eğer beş nokta almış olsaydık,
# houghda beş tane bir noktada kesişen doğru olacaktı.

img = cv.imread("../input_pictures/input_rende.png")
gray = cv.cvtColor(img, cv.COLOR_RGBA2GRAY)
canny = cv.Canny(gray, 40, 160)

lines = cv.HoughLines(canny, 1, np.pi / 180, 190, )
# HoughLines params=>
# input img
# rho = (0,0)'dan yani sol üstten uzaklık
# theta = açı (radyan cinsinden)
# threshold = örn thres 140'sa sadece thres>140 olan linelar dönecek.

for line in lines:
    rho, theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b)) # ??
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))
    cv.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
cv.imwrite('houghlinetf.png', img)

# bunu kullanmak yerine biraz kayıp ile probabilistic versiyonu kullanabiliriz.
