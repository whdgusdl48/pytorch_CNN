import cv2
import numpy as np

def my_bilinear(src, dst_shape):
    #########################
    # TODO                  #
    # my_bilinear 완성      #
    #########################
    rate = dst_shape[0]/src.shape[0]
    dst = np.zeros(dst_shape,dtype=np.uint8)
    print(src.shape)
    for y in range(dst_shape[1]):
        for x in range(dst_shape[0]):
            px = int(x / rate)
            py = int(y / rate)

            fx1 = float(x/rate) - px
            fx2 = 1 - fx1
            fy1 = float(y/rate) - py
            fy2 = 1 - fy1

            v4 = fx1 * fy1
            v2 = fx1 * fy2
            v3 = fx2 * fy1
            v1 = fx2 * fy2

            if px < src.shape[0]-1 and py < src.shape[1]-1 :
                result = (v4 * src[py+1][px+1]) + (v2 * src[py][px+1]) + (v3 * src[py + 1][px]) + (v1 * src[py][px])


            dst[y][x] = result

    dst = (dst+0.5).astype(np.uint8)
    return dst

if __name__ == '__main__':
    src = cv2.imread('Lena.png', cv2.IMREAD_GRAYSCALE)
    # dst = cv2.resize(src, dsize=(128, 128), interpolation=cv2.INTER_LINEAR)
    # dst2 = cv2.resize(src, dsize=(512, 512), interpolation=cv2.INTER_LINEAR)
    #
    # cv2.imshow("src", src)
    # cv2.imshow("dst", dst)
    # cv2.imshow("dst2", dst2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    #이미지 크기 ??x??로 변경
    my_dst_mini = my_bilinear(src, (128,128))
    #이미지 크기 512x512로 변경(Lena.png 이미지의 shape는 (512, 512))
    my_dst = my_bilinear(my_dst_mini, (512,512))
    #
    cv2.imshow('original', src)
    cv2.imshow('my bilinear mini', my_dst_mini)
    cv2.imshow('my bilinear', my_dst)

    cv2.waitKey()
    cv2.destroyAllWindows()
