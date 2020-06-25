import numpy as np
import matplotlib.pyplot as plt
from skimage import color
from skimage import io

from scipy import ndimage
from scipy.signal import convolve2d



def compare_imgs(orig, modified):
    fig = plt.figure(figsize=(10,6))

    ax1 = fig.add_subplot(1,2,1)
    ax1.imshow(orig, cmap='gray')
    ax1.set_xlabel('Original')

    ax2 = fig.add_subplot(1,2,2)
    ax2.imshow(modified, cmap='gray')
    ax2.set_xlabel('Modified')

def image_utility(im):
    im_ = im.astype(np.float)
    
    # check if grayscale, it not, pack values to use
    if len(im.shape) > 2:
        width, height, c = im.shape
        if c > 1:
            img = 0.2126 * im_[:,:,0] + 0.7152 * im_[:,:,1] + 0.0722 * im[:,:,2]
    else:
        img = im_
    
    return img




def normalize(im):
    im *= 255.0 / np.max(im)
    return im

def average(im):
    img = image_utility(im)

    kernel = np.ones((5,5), dtype=np.float)

    img_ = convolve2d(img, kernel,
                      mode='same',
                      boundary='symm',
                      fillvalue=0)

    img_ *= normalize(img_)

    return img


def sobel(im, orientation='both'):
    img = image_utility(im)

    vert = np.array([[-1,-2,0,2,1],
                     [-4,-8,0,8,4],
                     [-6,-12,0,12,6],
                     [-4,-8,0,8,4],
                     [-1,-2,0,2,1],])

    horiz = np.array([[1,4,6,4,1],
                      [2,8,12,8,2],
                      [0,0,0,0,0],
                      [-2,-8,-12,-8,-2],
                      [-1,-4,-6,-4,-1]])
    if orientation == 'horiz':
        img_ = convolve2d(img, horiz, mode='same', boundary='symm', fillvalue=0)
    elif orientation == 'vert':
        img_ = convolve2d(img, vert, mode='same', boundary='symm', fillvalue=0)
    else:
        img_h = convolve2d(img, horiz, mode='same', boundary='symm', fillvalue=0)
        img_v = convolve2d(img, vert, mode='same', boundary='symm', fillvalue=0)

        img_ = np.sqrt(np.square(img_h) + np.square(img_v))

    img_ = normalize(img_)
    
    img_ = np.clip(img_, 0, 255)

    return img_