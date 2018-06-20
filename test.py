#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Patrick Wieschollek <mail@patwie.com>

from build import blur_library
import cv2
import psf
import numpy as np


img = cv2.imread('input.jpg')
img = img.astype(np.float32) / 255.

psf = psf.PSF(kernel_size=17)
k = next(psf.sample()).astype(np.float32)


def single_image_blur(img, k, gpu_id=0):
    """Apply PSF to images on the GPU usin cuDNN

    Args:
        img (nd.array.float32): image in shape [H,W,C]
        k (nd.array.float32): kernel in shape [H,W]
    """

    output = img.astype(np.float32).transpose((2, 0, 1)).copy()
    k = k.astype(np.float32)
    blur_library.blur(output, k, gpu_id)
    return output.transpose((1, 2, 0))


def batch_image_blur(img, k, gpu_id=0):
    """Apply PSF to images on the GPU usin cuDNN

    Args:
        img (nd.array.float32): image in shape [B,H,W,C]
        k (nd.array.float32): kernel in shape [B,H,W]
    """

    output = img.astype(np.float32).transpose((0, 3, 1, 2)).copy()
    print output.shape
    k = k.astype(np.float32)

    blur_library.blur_batch(output, k, gpu_id)
    return output.transpose((0, 2, 3, 1))


cv2.imwrite('input.jpg', img * 255)
cv2.imwrite('output.jpg', single_image_blur(img, k, 0) * 255)

img = img[None, ...].copy()
k = k[None, ...].copy()
cv2.imwrite('output2.jpg', batch_image_blur(img, k, 0)[0] * 255)
