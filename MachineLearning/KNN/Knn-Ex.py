import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

trainData = np.random.randint(0, 100, (100, 2)).astype(np.float32)
responses = np.random.randint(0, 2, (100, 1)).astype(np.float32)

red = trainData[responses.ravel() == 0]
plt.scatter(red[:, 0], red[:, 1], 80, 'r', 'x')

blue = trainData[responses.ravel() == 1]
plt.scatter(blue[:, 0], blue[:, 1], 80, 'b', 'o')


new = np.random.randint(0, 100, (1, 2)).astype(np.float32)
print("value of new: " + str(new))

plt.scatter(new[:, 0], new[:, 1], 80, 'g', 's')
knn = cv.ml.KNearest_create()
knn.train(trainData, cv.ml.ROW_SAMPLE, responses)
ret, results, neighbours, dist = knn.findNearest(new, 5)
print("result label:" + str(results))
print("label of neighbours: "+str(neighbours))
print("distance to neighbours: "+str(dist))
plt.show()
