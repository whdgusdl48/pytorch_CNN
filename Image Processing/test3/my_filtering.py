import cv2
import numpy as np
import decimal
def my_filtering(src, ftype, fsize):
    (h, w) = src.shape
    dst = np.zeros((h, w),dtype=float)

    if ftype == 'average':
        print('average filtering')
        ###################################################
        # TODO                                            #
        # mask 완성                                       #
        ###################################################
        mask = np.ones((fsize))
        mask = mask / (mask.shape[0] * mask.shape[1])
        #mask 확인
        print(mask)

    elif ftype == 'sharpening':
        print('sharpening filtering')
        ##################################################
        # TODO                                           #
        # mask 완성                                      #
        ##################################################
        mask = np.zeros((fsize))
        filter1 = np.ones((fsize))
        filter1 = filter1 / (filter1.shape[0] * filter1.shape[1])
        mask[int(mask.shape[0]/2)][int(mask.shape[1]/2)] = 2
        mask = mask - filter1
        #mask 확인
        print(mask)

    #########################################################
    # TODO                                                  #
    # dst 완성                                              #
    # dst : filtering 결과 image                            #
    #########################################################
    max = int(mask.shape[0]/2)
    min = int(mask.shape[1]/2)
    for row in range(src.shape[0]):
        for col in range(src.shape[1]):
            if row < max or col < min:
                dst[row][col] = 0
            elif row >= h - max or col >= w - min:
                dst[row][col] = 0
            else:
                sum = 0
                for i in range(mask.shape[0]):
                    for j in range(mask.shape[1]):
                        sum += src[row - i][col - j] * mask[i][j]
                if sum > 255:
                    sum = 255
                if sum < 0:
                    sum = 0
                dst[row][col] = sum
    dst = (dst).astype(np.uint8)
    print(dst)
    return dst

if __name__ == '__main__':
    src = cv2.imread('Lena.png', cv2.IMREAD_GRAYSCALE)
    # 3x3 filter
    # dst_average = my_filtering(src, 'average', (3,3))
    # dst_sharpening = my_filtering(src, 'sharpening', (3,3))

    #원하는 크기로 설정
    # dst_average = my_filtering(src, 'average', (7,7))
    # dst_sharpening = my_filtering(src, 'sharpening', (7,7))

    # 11x13 filter
    dst_average = my_filtering(src, 'average', (11,13))
    dst_sharpening = my_filtering(src, 'sharpening', (11,13))

    # cv2.imshow('original', src)
    cv2.imshow('average filter', dst_average)
    cv2.imshow('sharpening filter', dst_sharpening)
    cv2.waitKey()
    cv2.destroyAllWindows()
