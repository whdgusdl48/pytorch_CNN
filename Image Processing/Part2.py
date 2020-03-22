import imageio
import numpy as np
import warnings
import matplotlib.pyplot as plt
import matplotlib.cbook as cb
from scipy.signal import convolve2d
from skimage import color
from skimage import exposure

warnings.filterwarnings("ignore",category=cb.mplDeprecation)

image = imageio.imread('Image Processing\image\cheon.jpg')

plt.figure(figsize=(5,5))
plt.imshow(image)
plt.axis('off')
plt.show()

#image negative

pixv = 255 - image
plt.figure(figsize=(5,5))
plt.imshow(pixv)
plt.axis('off')
plt.show()

#Log transformation

# s=c∗log(r+1)

def gray(image):
    return np.dot(image[...,:3],[0.299,0.587,0.114])

gray = gray(image)

def log_transform(max):
    return (255/np.log(1+max)) * np.log(1+gray)

max = np.max(gray)
log = log_transform(max)
plt.figure(figsize = (5,5))
plt.imshow(log, cmap = plt.get_cmap(name = 'gray'))
plt.axis('off')
plt.show()

#Gamma Correction

# Vo=V1Gi V는 0~255

gamma = 2.2 # Gamma < 1 ~ Dark  ;  Gamma > 1 ~ Bright

gamma_correction = ((image/255) ** (1/gamma)) 
plt.figure(figsize = (5,5))
plt.imshow(gamma_correction)
plt.axis('off')
plt.show()

# def Convolution(image, kernel):
#     conv_bucket = []
#     for d in range(image.ndim):
#         conv_channel = convolve2d(image[:,:,d], kernel, 
#                                mode="same", boundary="symm")
#         conv_bucket.append(conv_channel)
#     return np.stack(conv_bucket, axis=2).astype("uint8")


# kernel_sizes = [9,15,30,60]
# fig, axs = plt.subplots(nrows = 1, ncols = len(kernel_sizes), figsize=(15,15));

# pic = imageio.imread('Image Processing\image\cheon.jpg')

# for k, ax in zip(kernel_sizes, axs):
#     kernel = np.ones((k,k))
#     kernel /= np.sum(kernel)
#     ax.imshow(Convolution(pic, kernel));
#     ax.set_title("Convolved By Kernel: {}".format(k));
#     ax.set_axis_off()

# plt.show()

#edge kernel

img = color.rgb2gray(image)

kernel = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])

edges = convolve2d(img,kernel, mode='valid')

edges_equalized = exposure.equalize_adapthist(edges/np.max(np.abs(edges)),clip_limit=0.03)

plt.figure(figsize = (5,5))
plt.imshow(edges_equalized, cmap='gray')    
plt.axis('off');
plt.show()