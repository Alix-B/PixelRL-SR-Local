# SPDX-License-Identifier: AGPL-3.0-or-later

import numpy as np
from common import unsharp, box_blur, laplacian, edge_enhance_more, sharpen
import cv2


class State:
    def __init__(self, size, move_range):
        self.image = np.zeros(size, dtype=np.float32)
        self.move_range = move_range
        self.tensor = None

    def reset(self, x, n):
        """
        Function was originally used at the start of an episode to add noise to image.

        Although effectively replaced by directly implementing processing into mini_batch_loader.py,
        it was retained to allow flexibility for future works.
        """

        self.image = x + n
        size = self.image.shape
        prev_state = np.zeros((size[0], 64, size[2], size[3]), dtype=np.float32)
        self.tensor = np.concatenate((self.image, prev_state), axis=1)

    def set(self, x):
        self.image = x
        self.tensor[:, :self.image.shape[1], :, :] = self.image

    def step(self, act, inner_state):
        """
        Function is used for creating the next image during each episode step based on the pixel-wise agent decisions.

        Individual action images are calculated only if said action is used in the step.

        Function returns a convolution of the input image and the action space.
        """

        neutral = (self.move_range - 1) / 2

        move = act.astype(np.float32)
        move = (move - neutral) / 255

        moved_image = self.image + move[:, np.newaxis, :, :]

        unsharp1 = np.zeros(self.image.shape, self.image.dtype)
        laplace1 = np.zeros(self.image.shape, self.image.dtype)
        gaussian = np.zeros(self.image.shape, self.image.dtype)
        blur1 = np.zeros(self.image.shape, self.image.dtype)
        edge2 = np.zeros(self.image.shape, self.image.dtype)
        sharpen1 = np.zeros(self.image.shape, self.image.dtype)

        batch_size, channels, height, width = self.image.shape

        for i in range(0, batch_size):
            if np.sum(act[i] == self.move_range) > 0:
                unsharp1[i, 0] = unsharp(self.image[i, 0])
            if np.sum(act[i] == self.move_range + 1) > 0:
                laplace1[i, 0] = laplacian(self.image[i, 0])
            if np.sum(act[i] == self.move_range + 2) > 0:
                gaussian[i, 0] = cv2.GaussianBlur(self.image[i, 0], ksize=(5, 5), sigmaX=0.5)
            if np.sum(act[i] == self.move_range + 3) > 0:
                blur1[i, 0] = box_blur(self.image[i, 0], (5, 5))
            if np.sum(act[i] == self.move_range + 4) > 0:
                edge2[i, 0] = edge_enhance_more(self.image[i, 0])
            if np.sum(act[i] == self.move_range + 5) > 0:
                sharpen1[i, 0] = sharpen(self.image[i, 0])

        self.image = moved_image
        self.image = np.where(act[:, np.newaxis, :, :] == self.move_range, unsharp1, self.image)
        self.image = np.where(act[:, np.newaxis, :, :] == self.move_range + 1, laplace1, self.image)
        self.image = np.where(act[:, np.newaxis, :, :] == self.move_range + 2, gaussian, self.image)
        self.image = np.where(act[:, np.newaxis, :, :] == self.move_range + 3, blur1, self.image)
        self.image = np.where(act[:, np.newaxis, :, :] == self.move_range + 4, edge2, self.image)
        self.image = np.where(act[:, np.newaxis, :, :] == self.move_range + 5, sharpen1, self.image)

        self.tensor[:, :self.image.shape[1], :, :] = self.image
        self.tensor[:, -64:, :, :] = inner_state
