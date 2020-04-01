import numpy as np
import cv2
import matplotlib.pyplot as plt

def my_calcHist(src):
    h, w = src.shape[:2]
    # 높이와 너비를 각각 따온다 shape 
    hist = np.zeros((256,),dtype=np.int)
    # 히스토그램형태를 세우기 위해 1차원배열로 선언해준다.
    for i in range(h):
        for j in range(w):
            intensity = src[i,j]
            hist[intensity] += 1
    # 이중반복문을 사용하여 중첩된 부분의 이미지 픽셀값을 계속해서 더한다.  
    return hist

def my_normalize_hist(hist, pixel_num):
    # 히스토그램 배열형태의 넘파이 나누기 연산을 통해 픽셀 넘버를 받아와서 나눠준다.
    return hist / (pixel_num)


def my_PDF2CDF(pdf):
    pdf = np.array(pdf)
    # 넘파이 배열로 변환해준다 매개변수를
    cdf = np.zeros((len(pdf),))
    # cdf라는 넘파이 배열을 만들어준다.
    cdf = np.cumsum(pdf)   
    # 넘파이 라이브러리 누적함수인 cumsum을 통해 누적 히스토그램을 만들어준다.
    return cdf


def my_denormalize(normalized, gray_level):
    denormalized = np.zeros((len(normalized),))
    # denormalized 넘파이 배열을 선언해준다.
    denormalized = normalized * gray_level
    # 넘파이 곱하기 연산을 통해 매개변수 2번째 입력값을 곱해주고 반환해준다.
    return denormalized


def my_calcHist_equalization(denormalized, hist):
    hist_eq = np.zeros((256,),dtype=np.float32)
    # 가짜 배열 하나를 선언해준다.
    index, count = np.unique(denormalized,return_counts=True)
    # 넘파이 카운트와 인덱스를 생성해주는 unique함수를 이용하여 count 1개인거 0개인거 다수인거를 구별한다.
    for i in range(len(count)):
        if count[i] == 1:
            hist_eq[i] = hist[i]
        else :
            break
    # 첫번째 반복문은 카운트 배열을 통해 카운트가 만약 1개이면 히스토그램에 값을 넣는다.  
    for i in range(len(hist)-1):
        for j in range(i+1,len(hist)):
            if hist_eq[i] == 0 :
                hist_eq[i] += hist[j]
    # 두번째 반복문은 카운트가 2이상인 경우에만 히스토그램값을 중첩시켜서 계산한다. 
    min = 0
    max = 81
    # 최대 최소는 히스토그램 인덱스 기준으로 255이하를 나눈다.
    hist_equal = np.zeros((256,),dtype=np.float)
    # 반환할 넘파이 배열을 선언해준다.
    for i in range(min,max+1):
        j = int((255-0)/(max - min) * (i-min) + 0)
        hist_equal[j] = int(hist_eq[i])
    # 새로 배열된 히스토그램을 stretch한다.   
    return hist_equal


def my_equal_img(src, output_gray_level):
    (h , w) = src.shape
    dst = np.zeros((h,w),dtype=np.uint8)
    # 새로 이미지를 받을 배열을 선언한다.
    output_gray_level = np.ma.filled(output_gray_level,0).astype('uint8')
    # 이미지가 0으로 된부분을 다시 int8형태로 채운다.
    dst = output_gray_level[src]
    # 반환값에 1차원배열에 이미지를 덧붙인다.
    return dst

#input_image의  equalization된 histogram & image 를 return
def my_hist_equal(src):
    (h, w) = src.shape
    max_gray_level = 255
    histogram = my_calcHist(src)
    normalized_histogram = my_normalize_hist(histogram, h * w)
    normalized_output = my_PDF2CDF(normalized_histogram)
    denormalized_output = my_denormalize(normalized_output, max_gray_level)
    output_gray_level = denormalized_output.astype(int)
    hist_equal = my_calcHist_equalization(output_gray_level, histogram)

    ### dst : equalization 결과 image
    dst = my_equal_img(src, output_gray_level)

    return dst, hist_equal

if __name__ == '__main__':
    src = cv2.imread('Image Processing/test2/fruits_div3.jpg', cv2.IMREAD_GRAYSCALE)
    hist = my_calcHist(src)
    dst, hist_equal = my_hist_equal(src)

    cv2.imshow('original', src)
    binX = np.arange(len(hist))
    plt.title('my histogram')
    plt.bar(binX, hist, width=0.5, color='g')
    plt.show()

    cv2.imshow('equalizetion after image', dst)
    binX = np.arange(len(hist_equal))
    plt.title('my histogram equalization')
    plt.bar(binX, hist_equal, width=0.5, color='g')
    plt.show()

    cv2.waitKey()
    cv2.destroyAllWindows()

