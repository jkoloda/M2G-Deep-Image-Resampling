"""Image (pre)processing utilities."""

import numpy as np
import cv2
import random


def extract_random_block(img, blk_size):
    """Extract random block from the image.

    Returns
    -------
    block : ndarray
        Image block with size blk_size x blk_size

    """
    [rows, cols] = img.shape
    r_start = np.random.randint(0, rows - blk_size)
    c_start = np.random.randint(0, cols - blk_size)
    return img[r_start:r_start+blk_size, c_start:c_start+blk_size]


def add_border(img, border, fill=0):
    """Add border around image.

    Parameters
    ----------
    border : int
        Border with (in pixels).

    fill : float
        Pixel value the border should be filled with.

    Returns
    -------
    img : ndarray
        Image with the added border.

    """
    if border == 0:
        return img

    # Add border
    [rows, cols] = img.shape
    container = np.ones((rows+2*border, cols+2*border))*fill
    container[border:-border, border:-border] = img
    return container


def add_random_sign(x):
    """Add random sign to a given variable.

    Paramaters
    ----------
    x : float or ndarray
        Variable whose sign is to be randomly flipped.

    Returns
    -------
    x : float or ndarray
        Input variable with the sign randomly flipped.

    """
    sign = random.choice([+1, -1])
    return sign * x
