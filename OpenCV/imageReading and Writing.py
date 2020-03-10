#-*- coding:utf-8 -*-
import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt

def showImage():
    imgfile = 'OpenCV\image\cheon.jpg' 
    # 이미지 경로
    img = cv2.imread(imgfile,cv2.IMREAD_COLOR)
    # 이미지 읽어오기 cv2.IMREAD_COLOR 컬러이미지 cv2.IMREAD_GRAYSCALE 흑백이미지 cv2.IMREAD_UNCHAGNED 알파채널을 포함하여 이미지 그대로 로드
    cv2.namedWindow('Cheon woo hee', cv2.WINDOW_NORMAL)
    # 이미지가 보여질 제목 설정 윈도우 크기로
    cv2.imshow('Cheon woo hee', img)
    # 이미지 이름 설정하고 2번째 인자는 이미지를 받아온다.
    cv2.waitKey(0)
    # 키보드 입력을 기다리는 함수
    cv2.destroyAllWindows()
    # 생성한 모든 윈도우를 제거

showImage() 

def showImage2():
    imgfile = 'OpenCV\image\cheon.jpg'
    img = cv2.imread(imgfile, cv2.IMREAD_COLOR)
    cv2.imshow('cheon', img)

    k = cv2.waitKey(0) & 0xFF

    if k == 27:
        cv2.destroyAllWindows()
    elif k == ord('c'):
        cv2.imwrite('OpenCV\image\cheon_copy.jpg', img)
        cv2.destroyAllWindows()
    #c키를 입력하면 복사본 생성 imwrite함수를 이용하여 생성될 카피본 이미지이름    

def showImageWithmat():

    imgfile = 'OpenCV\image\cheon.jpg'
    img = cv2.imread(imgfile, cv2.IMREAD_GRAYSCALE)

    plt.imshow(img, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.title('Cheon woo hee')
    plt.show()

showImageWithmat()    