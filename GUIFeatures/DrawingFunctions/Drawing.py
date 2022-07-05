import numpy as np
import cv2 as cv


def draw_circle(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDBLCLK:  # çift tıklamada çizer
        cv.circle(img, (x, y), 100, (200, 110, 200), -1)


img = np.zeros((515, 1000, 3), np.uint8)  # pencere boyutları
cv.namedWindow('img')
cv.setMouseCallback('img', draw_circle)
while True:
    cv.imshow('img', img)
    if cv.waitKey(20) & 0xFF == 27:  # esc ye basınca çıkış
        break
cv.destroyAllWindows()
