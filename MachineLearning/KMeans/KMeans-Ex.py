import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

X = np.random.randint(30, 80, (50, 2))
Y = np.random.randint(50, 90, (50, 2))
Z = np.vstack((X, Y))

Z = np.float32(Z)

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
ret, label, center = cv.kmeans(Z, 2, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)

A = Z[label.ravel() == 0]
B = Z[label.ravel() == 1]

plt.scatter(A[:, 0], A[:, 1])
plt.scatter(B[:, 0], B[:, 1], c='r')
plt.scatter(center[:, 0], center[:, 1], s=100, c='b', marker='o')
plt.xlabel('Height'), plt.ylabel('Weight')
plt.show()
