#!/usr/bin/env python3

# SCC0251 - Image Processing and Analysis (2023.1)
# Assignment A1 - Enhancement and Superresolution
# Frederico Bulhões de Souza Ribeiro
# NUSP - 11208440

import numpy as np
import imageio.v3 as iio
from enum import Enum
from typing import Dict, Tuple, Callable

import matplotlib.pyplot as plt

MAX_VAL = 255

### input auxiliary functions
def read_parameters() -> Tuple[Dict[str, str], Dict[str, float]]:
    # returns dictionary mapping inputs to values
    params = {k : input().strip() for k in ['input', 'expected', 'filter_idx']}

    filter_param_map = [
        ["r"],
        ["r"],
        ["r1", "r2"],
        [],
        ["s1", "s2"],
        ["d0", "n"],
        ["d0", "n"],
        ["d0", "d1", "n1", "n2"],
        ["d0", "d1", "n1", "n2"],
    ]

    filter_params = {
        filter_parameter : float(input().strip()) for filter_parameter
        in filter_param_map[int(params["filter_idx"])]
    }
    
    return params, filter_params

### float -> uint8 conversion function
def to_greyscale(image : np.ndarray, normalize : bool = False) -> np.ndarray:
    # converts the image to greyscale, considering img_min and img_max as the range of the 
    # pixels of the image
    # if normalize is true, considers the smallest value of the image as 0 and the max as 255
    img_min, img_max = (image.min(), image.max()) if normalize else (0, 1)

    raw_normalized = (image - img_min) / (img_max - img_min)
    raw = raw_normalized * MAX_VAL

    return raw.astype(np.uint8)

# print image
def print_img(img : np.ndarray, title = "", normalize=False):
    img = to_greyscale(img, normalize)
    plt.imshow(img, vmin=0, vmax=255)
    plt.title(title)
    plt.show()

# root mean square error
def rmse(img1 : np.ndarray, img2 : np.ndarray):
    # returns the root mean square error of the difference of two images
    return np.sqrt(np.sum((img1.astype(np.float32) - img2.astype(np.float32))**2) / np.size(img1))

# fft + fftshift tranformation
def to_frequency_domain(img : np.ndarray) -> np.ndarray:
    return np.fft.fftshift(np.fft.fft2(img))

# inverse of previous transformation
def to_spatial_domain(img : np.ndarray) -> np.ndarray:
    return np.real(np.fft.ifft2(np.fft.ifftshift(img)))

### filters

class PassType(Enum):
    LOW_PASS = 0
    HIGH_PASS = 1
    BAND_PASS = 2
    BAND_REJECT = 3

# generate mask (N x M) by calling func on the pair of indexes (x, y)
def mask_from_func(N : int, M : int, func : Callable):
    mask = np.zeros((N, M), dtype=np.float32)

    for i in range(N):
        for j in range(M):
            mask[i, j] = func(i, j)

    return mask

# ideal filter
def ideal_filter(img : np.ndarray, pass_type : PassType, *, r = None, r1 = None, r2 = None):
    N, M = img.shape

    dist = lambda x, y: np.sqrt((x - N//2)**2 + (y - M//2)**2)
    match pass_type:
        case pass_type.LOW_PASS:
            f = mask_from_func(N, M, lambda x, y: 1.0 if dist(x, y) <= r else 0.0)
        case pass_type.HIGH_PASS:
            f = mask_from_func(N, M, lambda x, y: 0.0 if dist(x, y) <= r else 1.0)
        case pass_type.BAND_PASS:
            f = mask_from_func(N, M, lambda x, y: 1.0 if r2 <= dist(x, y) <= r1 else 0.0)

    return to_spatial_domain(to_frequency_domain(img) * f)

def main():
    inputs, filter_params = read_parameters()

    input = iio.imread(inputs['input'])
    input_norm = input.astype(np.float32) / MAX_VAL

    expected = iio.imread(inputs['expected'])

    filters_map = [
        ideal_filter, ideal_filter, ideal_filter,
        # laplacian_filter,
        # gaussian_filter,
        # butterworth_filter, butterworth_filter, butterworth_filter, butterworth_filter
    ]

    filter_pass_type_map = [
        PassType.LOW_PASS, PassType.HIGH_PASS, PassType.BAND_PASS,
        PassType.LOW_PASS,
        PassType.LOW_PASS,
        PassType.LOW_PASS, PassType.HIGH_PASS, PassType.BAND_REJECT, PassType.BAND_PASS
    ]
    
    filter_idx = int(inputs['filter_idx'])

    filter_func = filters_map[filter_idx]
    filter_pass_type = filter_pass_type_map[filter_idx]

    output_norm = filter_func(input_norm, filter_pass_type, **filter_params)
    output = to_greyscale(output_norm, normalize=True)

    # print_img(input, title="input")
    # print_img(output, title="output")
    # print_img(expected, title="expected")

    print(rmse(output, expected))

if __name__ == '__main__':
    main()