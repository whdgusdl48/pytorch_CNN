import numpy as np
import cv2

def showVideo():
    try:
        print('카메라를 구동합니다.')
        cap = cv2.VideoCapture('C:\\Users\종현\PycharmProjects\Data\mycam.avi')
        # 비디오 캡처를 위해 객체생성한다. 인자로는 장치 인덱스또는 비디오 파일 이름
    except:
        print('카메라 구동 실패')
        return

    cap.set(3, 480)
    cap.set(4, 320)

    while True:
        ret, frame = cap.read()
        #프레임 읽기 읽었으면 ret은 True가 된다. 

        if not ret:
            print('비디오 읽기 오류')
            break       
        
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        #프레임을 흑백으로 변환
        cv2.imshow('video', gray)

        k = cv2.waitKey(0) & 0xFF

        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()  

showVideo()

def writeVideo():
    try:
        print('카메라를 구동합니다.')
        cap =cv2.VideoCapture(0)

    except:
        print('카메라 구동 실패')
        return

    fps = 20.0
    width = int(cap.get(3))
    height = int(cap.get(4))
    fcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')

    out = cv2.VideoWriter('mycam.avi', fcc, fps, (width,height))
    print('녹화를 시작합니다.')

    while True:
        ret, frame = cap.read()

        if not ret:
            print('비디오 읽기 오류')
            break
        cv2.imshow('video', frame)
        out.write(frame)

        k = cv2.waitKey(0) & 0xFF

        if k == 27:
            print('녹화를 종료합니다.')
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()        
