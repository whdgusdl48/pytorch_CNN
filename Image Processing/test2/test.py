import numpy as np
import cv2
import matplotlib.pyplot as plt

src = cv2.imread('Image Processing/test2/fruits_div3.jpg', cv2.IMREAD_GRAYSCALE)
hist = cv2.calcHist([src],[0],None,[256],[0,256])
hist = hist.flatten()

hist2 = cv2.equalizeHist(src,hist)
binX = np.arange(len(hist2))
plt.title('my histogram')
plt.bar(binX, hist2, width=0.5, color='g')
plt.show()

cv2.imshow('1',hist2)
cv2.waitKey()
cv2.destroyAllWindows()