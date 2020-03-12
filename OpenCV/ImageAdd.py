import numpy as np
import cv2

def addImage(img1,img2):
    img1 = cv2.imread(img1)
    img2 = cv2.imread(img2)

    subimg1 = img1[300:700, 200:600]
    subimg2 = img2[100:500, 100:500]

    cv2.imshow('img1',subimg1)
    cv2.imshow('img2',subimg2)

    add_img1 = subimg1 + subimg2
    add_img2 = cv2.add(subimg1,subimg2)

    cv2.imshow('add1',add_img1)
    cv2.imshow('add2',add_img2)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


addImage('OpenCV\image\cheon.jpg','OpenCV\image\jungbong.jpg')    