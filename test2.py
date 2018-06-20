#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Patrick Wieschollek <mail@patwie.com>

import blur_library
import cv2
import numpy as np
import sys
sys.path.append('../../deblurring')
import psf


img = cv2.imread('/home/wieschol/git/github.com/patwie/diss/figures/deblurring/41070615952_ff2e03b22e_o.jpg-crop-t859-l4311-b1154-r4607.png')
img = img.astype(np.float32)[:, :, :] / 255.
img = img.transpose((2, 0, 1)).copy()

psf = psf.PSF(kernel_size=17)
k = next(psf.sample()).astype(np.float32)

print img.shape
print k.shape

cv2.imwrite('input.jpg', img.transpose((1, 2, 0)) * 255)
blur_library.blur(img, k, 0)
cv2.imwrite('output.jpg', img.transpose((1, 2, 0)) * 255)
