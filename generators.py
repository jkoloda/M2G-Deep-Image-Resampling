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
from numpy.linalg import inv
import cv2
from utils import (
    add_border,
    add_random_sign,
    build_transformation_matrix,
    extract_random_block,
    transform_image,
)


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
        forward_r_mesh_batch = []
        forward_c_mesh_batch = []
        backward_r_mesh_batch = []
        backward_c_mesh_batch = []
        forward_blk_batch = []
        forward_mask_batch = []

        while True:
            for index in indices:
                T = self.get_random_transform()

                # Input batch
                blk = self.get_block(self.images[index])
                transformed, r_mesh, c_mesh = transform_image(
                                                        [blk, self.mask], T)

                # Input batch normalization
                blk = (blk/255.0 - 0.5)*2.0
                blk = blk * self.mask

                transformed_blk = transformed[0]
                transformed_mask = (transformed[1] > 0)
                transformed_blk = (transformed_blk/255.0 - 0.5)*2.0
                transformed_blk = transformed_blk * transformed_mask

                r_mesh = (r_mesh - self.r_grid)/self.blk_size
                c_mesh = (c_mesh - self.c_grid)/self.blk_size
                r_mesh = r_mesh*self.mask
                c_mesh = c_mesh*self.mask

                # 1) Build input batch (ground truth)
                # and forward transform meshes
                input_batch.append(blk)
                forward_r_mesh_batch.append(r_mesh)
                forward_c_mesh_batch.append(c_mesh)

                # 2) Transformed data
                forward_blk_batch.append(transformed_blk)
                forward_mask_batch.append(transformed_mask)

                # 3) Backward transform meshes
                _, r_mesh, c_mesh = transform_image([self.mask], inv(T))
                r_mesh = (r_mesh - self.r_grid)/self.blk_size
                c_mesh = (c_mesh - self.c_grid)/self.blk_size
                r_mesh = r_mesh*transformed_mask
                c_mesh = c_mesh*transformed_mask

                backward_r_mesh_batch.append(r_mesh)
                backward_c_mesh_batch.append(c_mesh)

            yield [np.asarray(input_batch),
                   np.asarray(forward_r_mesh_batch),
                   np.asarray(forward_c_mesh_batch),
                   np.asarray(forward_blk_batch),
                   np.asarray(forward_mask_batch),
                   np.asarray(backward_r_mesh_batch),
                   np.asarray(backward_c_mesh_batch)]

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
        # Rotation
        angle = add_random_sign(np.random.randint(0, self.max_rotation))
        T, t = build_transformation_matrix(rotation=angle)
        return T
