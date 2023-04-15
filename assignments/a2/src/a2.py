#!/usr/bin/env python3

# SCC0251 - FloatImage Processing and Analysis (2023.1)
# Assignment A2 - Fourier Transform
# Frederico Bulhões de Souza Ribeiro
# NUSP - 11208440

import numpy as np
import numpy.typing as npt
import imageio.v3 as iio
from enum import Enum
from typing import Dict, Tuple, Callable, List, Optional
from typing import TypeAlias

import matplotlib.pyplot as plt # type:ignore

MAX_VAL = 255

ImageDType : TypeAlias = np.float64
ImageRawDType : TypeAlias = np.uint8
RawImage : TypeAlias = npt.NDArray[ImageRawDType]
FloatImage : TypeAlias = npt.NDArray[ImageDType]
FreqDomImage : TypeAlias = npt.NDArray[np.complex128]
FilterFunc : TypeAlias = Callable[..., FloatImage]
          
### input auxiliary functions
def read_inputs() -> Tuple[Dict[str, str], Dict[str, float]]:
    # returns dictionary mapping inputs to values
    params = {k : input().strip() for k in ['input', 'expected', 'filter_idx']}

    filter_param_map : List[List[str]] = [
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
def to_greyscale(image : FloatImage, normalize : bool = False) -> RawImage:
    # converts the image to greyscale, considering img_min and img_max as the range of the 
    # pixels of the image
    # if normalize is true, considers the smallest value of the image as 0 and the max as 255
    img_min, img_max = (image.min(), image.max()) if normalize else (0, 1)

    raw_normalized : FloatImage = (image - img_min) / (img_max - img_min)
    raw = raw_normalized * MAX_VAL

    return raw.astype(np.uint8)

# print image
def print_imgs(imgs : List[RawImage], titles : Optional[List[str]] = None, normalize : bool = False) -> None:
    fig, axs = plt.subplots(1, len(imgs))

    if titles is None:
        titles = [f"image {i}" for i in range(1, len(imgs)+1)]

    for img, title, p in zip(imgs, titles, axs):
        p.imshow(img, cmap='gray', vmin=0, vmax=MAX_VAL)
        p.set_title(title)
        
    plt.show()

# root mean square error
def rmse(img1 : FloatImage, img2 : FloatImage) -> float:
    # returns the root mean square error of the difference of two images
    return float(np.sqrt(np.sum((img1.astype(np.float64) - img2.astype(np.float64))**2) / np.size(img1)))

# fft + fftshift tranformation
def to_frequency_domain(img : FloatImage) -> FreqDomImage:
    return np.fft.fftshift(np.fft.fft2(img))

# inverse of previous transformation
def to_spatial_domain(img : FreqDomImage) -> FloatImage:
    return np.real(np.fft.ifft2(np.fft.ifftshift(img)))

### filters

class PassType(Enum):
    LOW_PASS = 0
    HIGH_PASS = 1
    BAND_PASS = 2
    BAND_REJECT = 3

# generate mask (N x M) by calling func on the pair of indexes (x, y)
def mask_from_func(N : int, M : int, func : Callable[[FloatImage, FloatImage], FloatImage]) -> FloatImage:
    xs, ys = np.mgrid[0:N,0:M].astype(np.float64)
    
    mask = func(xs, ys)

    return mask.astype(ImageDType)

### filters

def ideal_filter(img : FloatImage, pass_type : PassType, *, \
                 r : Optional[float] = None, r1 : Optional[float] = None, \
                 r2 : Optional[float] = None) -> FloatImage:
    N, M = img.shape

    dist = lambda x, y: np.sqrt((x - N//2)**2 + (y - M//2)**2)

    f : FloatImage
    match pass_type:
        case pass_type.LOW_PASS:
            f = mask_from_func(N, M, lambda x, y: np.zeros((N, M)) + (dist(x, y) <= r))
        case pass_type.HIGH_PASS:
            f = mask_from_func(N, M, lambda x, y: np.ones((N, M)) - (dist(x, y) <= r))
        case pass_type.BAND_PASS:
            f = mask_from_func(N, M, lambda x, y: np.zeros((N, M)) + (r2 < dist(x, y))*(dist(x, y) <= r1))
        case pass_type.BAND_REJECT:
            f = mask_from_func(N, M, lambda x, y: np.ones((N, M)) - (r2 < dist(x, y))*(dist(x, y) <= r1))

    return to_spatial_domain(to_frequency_domain(img) * f)

def laplacian_filter(img : FloatImage, pass_type : PassType) -> FloatImage:
    N, M = img.shape

    dist2 = lambda x, y: (x - N//2)**2 + (y - M//2)**2

    f : FloatImage
    match pass_type:
        case pass_type.LOW_PASS:
            f = mask_from_func(N, M, lambda x, y: +4.0 * np.pi**2 * dist2(x, y))
        case pass_type.HIGH_PASS:
            f = mask_from_func(N, M, lambda x, y: -4.0 * np.pi**2 * dist2(x, y))

    return to_spatial_domain(to_frequency_domain(img) * f)

def gaussian_filter(img : FloatImage, pass_type : PassType, *, \
                    s1 : float, s2 : float) -> FloatImage:
    N, M = img.shape

    e = lambda x, y: (x - N//2)**2 / (2.0*s1**2) + (y - M//2)**2 / (2.0*s2**2)

    f : FloatImage
    match pass_type:
        case pass_type.LOW_PASS:
            f = mask_from_func(N, M, lambda x, y: np.exp(-e(x, y)))
        case pass_type.HIGH_PASS:
            f = mask_from_func(N, M, lambda x, y: -np.exp(-e(x, y)))

    return to_spatial_domain(to_frequency_domain(img) * f)

def butterworth_filter(img : FloatImage, pass_type : PassType, *, \
                       n : Optional[float] = None, n1 : Optional[float] = None, \
                       n2 : Optional[float] = None, d0 : Optional[float]= None, \
                       d1 : Optional[float] = None) -> FloatImage:
    N, M = img.shape

    dist2 = lambda x, y: (x - N//2)**2 + (y - M//2)**2

    f : FloatImage
    match pass_type:
        case pass_type.LOW_PASS:
            f = mask_from_func(N, M, lambda x, y: 1 / (1 + (dist2(x, y) / (d0 * d0)) ** n))
        case pass_type.HIGH_PASS:
            f = mask_from_func(N, M, lambda x, y: 1 - 1 / (1 + (dist2(x, y) / (d0 * d0)) ** n))
        case pass_type.BAND_PASS:
            f1 = mask_from_func(N, M, lambda x, y: 1 / (1 + (dist2(x, y) / (d0 * d0)) ** n1))
            f2 = mask_from_func(N, M, lambda x, y: 1 / (1 + (dist2(x, y) / (d1 * d1)) ** n2))
            f = f1 - f2
        case pass_type.BAND_REJECT:
            f1 = mask_from_func(N, M, lambda x, y: 1 - 1 / (1 + (dist2(x, y) / (d0 * d0)) ** n1))
            f2 = mask_from_func(N, M, lambda x, y: -1 / (1 + (dist2(x, y) / (d1 * d1)) ** n2))
            f = f1 - f2

    return to_spatial_domain(to_frequency_domain(img) * f)

def main() -> None:
    inputs, filter_params = read_inputs()

    input = iio.imread(inputs['input'])
    input_norm = input.astype(ImageDType) / MAX_VAL

    expected = iio.imread(inputs['expected'])

    filters_map : List[Tuple[FilterFunc, PassType]] = [
        (ideal_filter,      PassType.LOW_PASS),
        (ideal_filter,      PassType.HIGH_PASS),
        (ideal_filter,      PassType.BAND_PASS),
        (laplacian_filter,  PassType.LOW_PASS),
        (gaussian_filter,   PassType.LOW_PASS),
        (butterworth_filter,PassType.LOW_PASS),
        (butterworth_filter,PassType.HIGH_PASS),
        (butterworth_filter,PassType.BAND_REJECT),
        (butterworth_filter,PassType.BAND_PASS)
    ]

    filter_idx = int(inputs['filter_idx'])

    filter_func, filter_pass_type = filters_map[filter_idx]

    output_norm = filter_func(input_norm, filter_pass_type, **filter_params)
    output = to_greyscale(output_norm, normalize=True)

    answer = rmse(output.astype(ImageDType), expected.astype(ImageDType))

    print(f"{answer:.4f}")

if __name__ == '__main__':
    main()