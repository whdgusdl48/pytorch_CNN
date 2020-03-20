import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as image
import imageio
def pprint(pic):
    print('Type of the image : ' , type(pic))
    print()
    print('Shape of the image : {}'.format(pic.shape))
    print('Image Hight {}'.format(pic.shape[0]))
    print('Image Width {}'.format(pic.shape[1]))
    print('Dimension of Image {}'.format(pic.ndim))
    print('Image size {}'.format(pic.size))
    print('Maximum RGB value in this image {}'.format(pic.max()))
    print('Minimum RGB value in this image {}'.format(pic.min()))

# image = image.imread('Image Processing\image\cheon.jpg')
# pprint(image)
# print('Value of only R channel {}'.format(image[ 100, 50, 0]))
# print('Value of only G channel {}'.format(image[ 100, 50, 1]))
# print('Value of only B channel {}'.format(image[ 100, 50, 2]))
# plt.figure(figsize = (15,15))

# plt.imshow(image)
# plt.show()

# plt.title('Red Channel')
# plt.imshow(image[:,:,0])
# plt.show()

# plt.title('Green Channel')
# plt.imshow(image[:,:,1])
# plt.show()

# plt.title('Blue Channel')
# plt.imshow(image[:,:,2])
# plt.show()

# image2 = imageio.imread('Image Processing\image\cheon.jpg')
# image2[50:150,:,0] = 255
# image2[200:250,:,1] = 255
# image2[300:330,:,2] = 255
# plt.imshow(image2)
# plt.show()

image3 = imageio.imread('Image Processing\image\cheon.jpg')


for c in zip(range(3)):

    splitimg = np.zeros(image3.shape, dtype='uint8')

    splitimg[:,:,c] = image3[:,:,c]

    plt.imshow(splitimg)
    plt.show()

image4 = imageio.imread('Image Processing\image\cheon.jpg')

gray = lambda rgb : np.dot(rgb[...,:3],[0.299,0.587,0.114])
# 내적 이용
gray = gray(image4)

plt.figure(figsize=(10,10))
plt.imshow(gray, cmap = plt.get_cmap(name = "gray"))
plt.show()

pic = imageio.imread('C:\\Users\종현\PycharmProjects\Data\Image Processing\image\special.JPG')
plt.figure(figsize=(10,10))
plt.imshow(pic)
plt.show()

low_pixel = pic < 20
# set value randomly range from 25 to 225 - these value also randomly choosen
pic[low_pixel] = np.random.randint(25,225)

# display the image
plt.figure( figsize = (10,10))
plt.imshow(pic)
plt.show()

# masking

pic2 = imageio.imread('C:\\Users\종현\PycharmProjects\Data\Image Processing\image\special.JPG')
total_row , total_col , layers = pic.shape
x , y = np.ogrid[:total_row , :total_col]
cen_x , cen_y = total_row/2 , total_col/2
distance_from_the_center = np.sqrt((x-cen_x)**2 + (y-cen_y)**2)
radius = (total_row/2)
circular_pic = distance_from_the_center > radius
pic[circular_pic] = 0
plt.figure(figsize = (10,10))
plt.imshow(pic) 
plt.show()   