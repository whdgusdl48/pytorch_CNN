import numpy as np
import cv2
from random import shuffle
import math

# def drawing():
#     img = np.zeros((512,512,3) , np.uint8)

#     cv2.line(img, (0,0), (511,511), (255, 0, 0), 5)
#     #라인을 그리는 함수 0,0 위치에서 511,511 위치 좌표까지 파란색으로 5두께만큼 그린다.
#     cv2.rectangle(img, (384, 0), (510,128) , (0,255,0), 3)
#     #사각형을 그리는 함수 384,0 좌측상단 꼭짓점 위치에서 510,128 우측 하단 꼭짓점으로 녹색으로 두께 3만큼 그린다.
#     cv2.circle(img, (447,63), 63, (0,0,255), -1)
#     #원을 그리는 함수 원의중심 좌표를 설정하고 반지름을 설정하고 빨간색으로 칠한다 -1이면 색을 안에 채운다.
#     cv2.ellipse(img, (256,256), (100,50), 0,0,180,(255,0,0), -1)
#     #반원을 그리는 함수 좌표를 설정하고 100,50 : 장축과 단축의 길이 그리고 기울기 각도와 호를 그리는 각도 호를 그리는 끝각도를 설정한다.
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     cv2.putText(img, 'OpenCV', (10, 500), font, 4, (255,255,255) , 2)
#     cv2.imshow('drawing', img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

b = [i for i in range(256)]
g = [i for i in range(256)]
r = [i for i in range(256)]



def onMouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        shuffle(b), shuffle(g), shuffle(r)
        cv2.circle(param, (x,y), 50, (b[0],g[0],r[0]), -1)

def mouseBrush():
    img = np.zeros((512,512,3) , np.uint8)  
    cv2.namedWindow('paint')
    cv2.setMouseCallback('paint', onMouse, param=img)

    while True:
        cv2.imshow('paint', img)
        k = cv2.waitKey(1) & 0xFF

        if k == 27:
            break

    cv2.destroyAllWindows()

mode, drawing = True, False
ix, iy = -1, -1
B = [i for i in range(256)]
G = [i for i in range(256)]
R = [i for i in range(256)]

def onMouse2(event, x, y, flags, param):
    global ix, iy, drawing, mode, B,G,R

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        shuffle(B), shuffle(R), shuffle(G)

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            if mode:
                cv2.rectangle(param, (ix,iy), (x,y),  (B[0],G[0],R[0]), -1)
            else:
                r = (ix - x) ** 2 + (iy - y) ** 2
                r = int(math.sqrt(r))
                cv2.circle(param, (ix,iy), r, (B[0],G[0],R[0]), -1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if mode:
            cv2.rectangle(param, (ix,iy), (x,y), (B[0],G[0],R[0]), -1)
        else:
            r = (ix - x) ** 2 + (iy - y) ** 2
            r = int(math.sqrt(r))
            cv2.circle(param, (ix,iy), r, (B[0],G[0],R[0]), -1)   

def mouseBrush2():
    global mode
    img = np.zeros((512,512,3) , np.uint8)  
    cv2.namedWindow('paint')
    cv2.setMouseCallback('paint', onMouse2, param=img)

    while True:
        cv2.imshow('paint', img)
        k = cv2.waitKey(1) & 0xFF

        if k == 27:
            break
        elif k == ord('m'):
            mode = not mode

    cv2.destroyAllWindows()

mouseBrush2()    