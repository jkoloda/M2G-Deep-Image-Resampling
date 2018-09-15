"""Implements custom generator for resampling purposes.


"""

import os
import cv2
import numpy as np
from keras.preprocessing.image import Iterator


class ResamplingGenerator(Iterator):
    """Custom block generator for resampling."""

    def __init__(self, folder, blk_size, border):
        """Construct generator object.

        The object will contain a list of loaded images and their
        corresponding filenames.
        """
        self.blk_size = blk_size
        self.border = border

        filenames = os.listdir(folder)
        filenames = [os.path.join(folder, f) for f in filenames]
        self.filenames = filenames
        self.num_images = len(filenames)

        # Load images and convert to gray scale
        self.images = []
        for f in filenames:
            img = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2GRAY)
            img = img.astype(np.float32)
            self.images.append(img)

        [self.rows, self.cols] = img.shape

    def next(self):
        """Implement next function."""
        # Randomly select images to fill up the batch
        indices = np.random.randint(0, self.num_images, self.batch_size)
        yield self.extract_random_block(self.images[indices[0]])

    def extract_random_block(self, img):
        """Extract random block from the image.

        Returns
        -------
        block : ndarray
            Image block with size (blk_size + 2*border) x (blk_size + 2*border)

        """
        # Select block
        r_start = np.random.randint(0, self.rows - self.blk_size)
        c_start = np.random.randint(0, self.cols - self.blk_size)
        blk = img[r_start:r_start+self.blk_size, c_start:c_start+self.blk_size]

        # Add border
        container = np.zeros((self.rows+2*self.border,
                              self.cols+2*self.border))
        container[self.border:-self.border, self.border:-self.border] = blk
        return container


g = ResamplingGenerator('dataset/KODAK', blk_size=32, border=0)
blk = g.next()
print blk
print blk.shape
