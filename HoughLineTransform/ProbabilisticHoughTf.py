import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# normal hough tf'de çok fazla işlem olduğu için prob hough tf kullanmayı
# tercih etmek isteyebiliriz.
# Diğerinin optimize edilmiş versiyonu.
# Tüm noktaları dikkate almıyor.
# Bunun yerine, yalnızca çizgi tespit için yeterli olan
# rastgele bir nokta alt kümesi alır.
# EŞİĞİ DÜŞÜRMEMİZ GEREK.

# Ekstra iki parametre var:
# minLineLength: min çizgi uzunluğu. Bundan kısa olan çizgiler dikkate alınmayacak.
# maxLineGap: çizgi elemanları arasında izin verilen max boşluk.

img = cv.imread("../input_pictures/input_sudoku.png")
gray = cv.cvtColor(img, cv.COLOR_RGBA2GRAY)
edges = cv.Canny(gray, 50, 140)
lines = cv.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)

cv.imwrite("probabilisticHoughLine.png", img)
