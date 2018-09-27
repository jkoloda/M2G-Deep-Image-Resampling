"""Image (pre)processing utilities."""

import numpy as np
import random
import math
from scipy.interpolate import griddata


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


def set_border(img, border, fill=0):
    """Set border around image to value.

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

    # Set border
    img = img[border:-border, border:-border]
    img = add_border(img, border=border, fill=fill)
    return img


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


def build_transformation_matrix(rotation=0, zoom=0, translation=(0, 0)):
    """Build transformation matrix according to specifications.

    Parameters
    ----------
    rotation : int
        Clockwise rotation agnle in degrees.

    zoom : float
        Zoom ratio. A value of 0.15 means a zoom-in of 15%. A value of -0.15
        means a zoom-out of 15%.

    translation : tuple
        Tuple of (x,y) values that are to be used for translation.

    Returns
    -------
    T : ndarray
        A 2x2 transformation matrix.

    t : ndarray
        A 2x1 translation vector.

    """
    angle = math.radians(rotation)
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle), np.cos(angle)]])

    Z = np.array([[1+zoom, 0], [0, 1+zoom]])
    t = np.reshape(np.asarray(translation), (2, 1))
    return np.matmul(R, Z), t


def transform_image(imgs, T, t=None):
    """Apply affine trnaform to image.

    Parameters
    ----------
    imgs : list
        List of nadarrays (images) that are to be transformed.

    T : ndarray
        A 2x2 transformation matrix.

    t : ndarray
        A 2x1 translation vector.

    Returns
    -------
    trans_img : ndarray
        Transformed image.

    mesh_r : ndarray
        Rows of the resulting floating mesh.

    mesh_c : ndarray
        Columns of the resulting floating mesh.

    """
    # Designed for even dimensions only
    [rows, cols] = imgs[0].shape
    assert (rows % 2 == 0 and rows % 2 == 0)

    # Images must have the same dimensions
    for img in imgs:
        [rows_temp, cols_temp] = img.shape
        assert (rows == rows_temp and cols == cols_temp)

    if t is None:
        t = np.reshape(np.asarray([0, 0]), (2, 1))

    # Compute floating mesh
    reference = [rows/2+0.5, cols/2+0.5]
    mesh_r = np.zeros((rows, cols))
    mesh_c = np.zeros((rows, cols))
    for r in range(0, rows):
        for c in range(0, cols):
            position = np.reshape(
                        np.array((r - reference[1], c - reference[0])), (2, 1))
            temp = np.matmul(T, position) + t
            mesh_r[r, c] = temp[1] + reference[0]
            mesh_c[r, c] = temp[0] + reference[1]

    # Estimate transformed image
    grid_x, grid_y = np.mgrid[0:cols:1, 0:rows:1]
    points = np.transpose(np.array([mesh_r.ravel(), mesh_c.ravel()]))
    points = points.astype(np.float32)
    xi = (grid_x, grid_y)
    out = [griddata(points, (img.ravel()).astype(np.float32),
                    xi, method='nearest') for img in imgs]

    return out, mesh_r, mesh_c
