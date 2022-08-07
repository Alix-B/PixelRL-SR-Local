# SPDX-License-Identifier: Unlicense

from PIL import Image, ImageFilter
import scipy
from scipy import ndimage
import numpy as np
import cv2
import torch
import torchvision.io


def draw_action_map(actions, colour_table):
    """ Returns a tensor action map with corresponding RGB values given actions and a corresponding colour map. """
    h = actions[0].shape[0]
    w = actions[0].shape[1]

    action_map = torch.zeros((3, h, w), dtype=torch.uint8)

    for i in range(0, h):
        for j in range(0, w):
            action_map[:, i, j] = colour_table[actions[0][i][j]]

    return action_map


def unsharp(images):
    blurred_f = ndimage.gaussian_filter(images, 1)
    filter_blurred_f = ndimage.gaussian_filter(blurred_f, 2)
    alpha = 3

    return np.clip(blurred_f + alpha * (blurred_f - filter_blurred_f), 0.0, 1.0)


def laplacian(images):
    kernel = np.ones((3, 3)) * (-1)
    kernel[1, 1] = 8
    Lap = scipy.ndimage.filters.convolve(images, kernel)
    Laps = Lap / 0.5
    sharp_im = images + Laps

    return np.clip(sharp_im, 0, 1)


def mae(image1, image2):
    img1 = np.asarray(image1, dtype=np.int16)
    img2 = np.asarray(image2, dtype=np.int16)
    abs_diff = np.abs(np.subtract(img1, img2))
    abs_diff_d = np.asarray(abs_diff, dtype=np.uint8)

    return np.mean(abs_diff_d)


def low_res(image):
    local_averaged = cv2.blur(image, (4, 4))
    image_down = cv2.resize(local_averaged, (64, 64), interpolation=cv2.INTER_CUBIC)
    image_up = cv2.resize(image_down, (256, 256), interpolation=cv2.INTER_CUBIC)

    return image_up


def low_res_bilinear(image):

    local_averaged = cv2.blur(image, (4, 4))
    image_down = cv2.resize(local_averaged, (64, 64), interpolation=cv2.INTER_LINEAR_EXACT)
    image_up = cv2.resize(image_down, (256, 256), interpolation=cv2.INTER_LINEAR_EXACT)

    return image_up


def low_res_nearest(image):

    local_averaged = cv2.blur(image, (4, 4))
    image_down = cv2.resize(local_averaged, (64, 64), interpolation=cv2.INTER_NEAREST)
    image_up = cv2.resize(image_down, (256, 256), interpolation=cv2.INTER_NEAREST)

    return image_up


def write_image(filepath, src):
    torchvision.io.write_png(src, filepath)


def box_blur(images, kernel):
    bb = cv2.boxFilter(images, ddepth=-1, ksize=kernel)
    return bb


def edge_enhance_more(images):
    image = Image.fromarray((images * 255).astype(np.uint8))
    image_edge_more = image.filter(ImageFilter.EDGE_ENHANCE_MORE)
    edge_more_im = np.asarray(image_edge_more) / 255

    return edge_more_im


def sharpen(images):
    image = Image.fromarray((images * 255).astype(np.uint8))
    image_sharp = image.filter(ImageFilter.SHARPEN)
    sharp_im = np.asarray(image_sharp) / 255

    return sharp_im
