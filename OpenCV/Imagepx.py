import numpy as np
import cv2

img = cv2.imread('OpenCV\image\cheon.jpg')
b, g, r = cv2.split(img)

subimg = img[300:400 , 350:700]
cv2.imshow('sub',subimg)
cv2.imshow('cheon woo heeb', b)
cv2.imshow('cheon woo heeg', g)
cv2.imshow('cheon woo heer', r)

merge_img = cv2.merge((b,g,r))
cv2.imshow('merge',merge_img)
cv2.waitKey(0)
cv2.destroyAllWindows()