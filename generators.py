"""Implements custom generator for resampling purposes.

The genrator extracts a random block for a given set of images and applies
on it a random transform.

The generator should return the following items:
1) Selected block
2) Image (or mask?) after transformation.
2) Transformed mesh (rows), i.e., pixel grid (positions) after transformation
3) Transformed mesh (cols), i.e., pixel grid (positions) after transformation
4) Transformed grid (rows)
5) Transformed grid (cols)

"""

import os
import numpy as np
import cv2
from utils import extract_random_block, add_border, add_random_sign, set_border


class ResamplingGenerator():
    """Custom block generator for resampling."""

    def __init__(self, folder, batch_size, blk_size, border, max_rotation=45):
        """Construct generator object.

        The object will contain a list of loaded images and their
        corresponding filenames.
        """
        self.blk_size = blk_size
        self.border = border
        self.batch_size = batch_size

        self.max_rotation = max_rotation

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

        # Build grid
        self.c_grid, self.r_grid = np.meshgrid(
                                    np.arange(0, self.blk_size+2*self.border),
                                    np.arange(0, self.blk_size+2*self.border))
        self.r_grid = (self.r_grid).astype(np.float32)
        self.c_grid = (self.c_grid).astype(np.float32)

        # Build mask
        self.mask = np.ones((self.blk_size, self.blk_size))
        self.mask = add_border(self.mask, border=self.border, fill=0)

    def flow_from_generator(self):
        """Create iterator that can be called with next()."""
        # Randomly select images to fill up the batch
        indices = np.random.randint(0, self.num_images, self.batch_size)
        input_batch = []
        masks_batch = []
        for index in indices:
            T = self.get_random_transform()

            # Input batch
            blk = self.get_block(self.images[index])
            r_mesh = cv2.warpAffine(self.r_grid,
                                    T,
                                    (self.blk_size+2*self.border,
                                     self.blk_size+2*self.border))
            c_mesh = cv2.warpAffine(self.c_grid,
                                    T,
                                    (self.blk_size+2*self.border,
                                     self.blk_size+2*self.border))

            # Input batch normalization
            blk = (blk/255.0 - 0.5)*2.0
            blk = blk * self.mask
            r_mesh = (r_mesh - self.r_grid)/self.blk_size
            c_mesh = (c_mesh - self.c_grid)/self.blk_size

            r_mesh = r_mesh*self.mask
            c_mesh = c_mesh*self.mask

            input_batch.append(np.stack((blk, r_mesh, c_mesh), axis=-1))

            # Mask batch
            transformed_mask = cv2.warpAffine(self.mask,
                                              T,
                                              (self.blk_size+2*self.border,
                                               self.blk_size+2*self.border))
            transformed_mask[transformed_mask <= 0] = 0
            transformed_mask[transformed_mask > 0] = 1
            masks_batch.append(transformed_mask)
        yield np.asarray(input_batch), np.asarray(masks_batch)

    def get_block(self, img):
        """Extract random block from the image.

        Returns
        -------
        block : ndarray
            Image block with size (blk_size + 2*border) x (blk_size + 2*border)

        """
        # Select block
        blk = extract_random_block(img, self.blk_size)
        blk = add_border(blk, border=self.border)
        return blk

    def get_random_transform(self):
        """Get random transform matrix (rotation, translation or zoom)."""
        transform = np.random.randint(0, 2)
        transform = 0

        # Rotation
        if transform == 0:
            angle = add_random_sign(np.random.randint(0, self.max_rotation))
            T = cv2.getRotationMatrix2D(
                ((self.blk_size+2*self.border)/2,
                 (self.blk_size+2*self.border)/2), angle, 1)
        elif transform == 1:
            pass
        return T
