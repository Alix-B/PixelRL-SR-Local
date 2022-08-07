# SPDX-License-Identifier: AGPL-3.0-or-later

import os
import numpy as np
import cv2
from common import low_res


class MiniBatchLoader(object):

    def __init__(self, train_path, test_path, image_dir_path, crop_size):

        # load data paths
        self.training_path_infos = self.read_paths(train_path, image_dir_path)
        self.testing_path_infos = self.read_paths(test_path, image_dir_path)

        self.crop_size = crop_size

    # test ok
    @staticmethod
    def path_label_generator(txt_path, src_path):
        for line in open(txt_path):
            line = line.strip()
            src_full_path = os.path.join(src_path, line)
            if os.path.isfile(src_full_path):
                yield src_full_path

    # test ok
    @staticmethod
    def count_paths(path):
        c = 0
        for _ in open(path):
            c += 1
        return c

    # test ok
    @staticmethod
    def read_paths(txt_path, src_path):
        cs = []
        for pair in MiniBatchLoader.path_label_generator(txt_path, src_path):
            cs.append(pair)
        return cs

    def load_training_data(self, indices):
        return self.load_data(self.training_path_infos, indices, augment=True)

    def load_testing_data(self, indices):
        # print("LOADING TESTING DATA", self.testing_path_infos)
        return self.load_data(self.testing_path_infos, indices)

    # test ok
    def load_data(self, path_infos, indices, augment=False):
        mini_batch_size = len(indices)
        in_channels = 1
        desired_indices = []
        # print(indices)

        if augment:
            raw_xs = np.zeros((mini_batch_size, in_channels, self.crop_size, self.crop_size)).astype(np.float32)
            processed_xs = np.zeros((mini_batch_size, in_channels, self.crop_size, self.crop_size)).astype(np.float32)

            for i, index in enumerate(indices):
                # print("PATH INFOS", path_infos, index)
                path = path_infos[index]
                # print("LOAD PATH", path)
                img = cv2.imread(path, 0)

                if img is None:
                    raise RuntimeError("invalid image: {i}".format(i=path))
                h, w = img.shape

                if np.random.rand() > 0.5:
                    img = np.fliplr(img)

                if np.random.rand() > 0.5:
                    angle = 10 * np.random.rand()
                    if np.random.rand() > 0.5:
                        angle *= -1
                    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
                    img = cv2.warpAffine(img, M, (w, h))

                processed_image = low_res(img)
                rand_range_h = h - self.crop_size
                rand_range_w = w - self.crop_size
                x_offset = np.random.randint(rand_range_w)
                y_offset = np.random.randint(rand_range_h)

                final_processed_image = processed_image[y_offset:y_offset + self.crop_size,
                                                        x_offset:x_offset + self.crop_size]

                final_image = img[y_offset:y_offset + self.crop_size,
                                  x_offset:x_offset + self.crop_size]

                processed_xs[i, 0, :, :] = (final_processed_image / 255).astype(np.float32)
                raw_xs[i, 0, :, :] = (final_image / 255).astype(np.float32)
                # print("LOADED IMAGES", raw_xs.shape, processed_xs.shape)

        elif mini_batch_size == 1:
            """ As currently implemented this is only true when validating / testing rather than training. """
            img = None

            # Save indices of desired images for exporting purposes later
            for i, index in enumerate(indices):
                # print("PATH INFOS", path_infos, index)
                path = path_infos[index]
                # print("PATH", path)
                path = str(path)

                img = cv2.imread(path, 0)

                if img is None:
                    raise RuntimeError("invalid image: {i}".format(i=path))

                # file names of desired images
                desired_images = ["2018_square", "8068_square"]

                image_id = path.split('/')
                # print(image_id[-1][:-4])

                if image_id[-1][:-4] in desired_images:
                    desired_indices.append([image_id, index])

            h, w = img.shape
            raw_xs = np.zeros((mini_batch_size, in_channels, h, w)).astype(np.float32)
            processed_xs = np.zeros((mini_batch_size, in_channels, h, w)).astype(np.float32)
            processed_image = low_res(img)
            processed_xs[0, 0, :, :] = (processed_image / 255).astype(np.float32)
            raw_xs[0, 0, :, :] = (img / 255).astype(np.float32)
            # print("LOADED TEST IMAGES", raw_xs.shape, processed_xs.shape)

        else:
            raise RuntimeError("mini batch size must be 1 when testing")

        return raw_xs, processed_xs, desired_indices
