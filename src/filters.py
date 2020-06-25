import numpy as np
import matplotlib.pyplot as plt
from skimage import color
from skimage import io

from scipy import ndimage
from scipy.signal import convolve2d



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


