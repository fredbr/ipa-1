#!/usr/bin/env python3

# Image Processing and Analysis (2023)
# Assignment #1 - enhancement and superresolution
# Frederico Bulhões de Souza Ribeiro
# NUSP - 11208440

import numpy as np
import imageio.v3 as iio
import glob
import sys

MAX_VAL = 255

### input auxiliary functions
def read_parameters():
    return {k : input() for k in ['imglow', 'imghigh', 'F']}

def glob_filenames(base_filename):
    return glob.glob(f'{base_filename}*')

### histogram functions
def cumulative_histogram(image):
    return np.bincount(image.flatten()).cumsum()

def normalized_cumulative_histogram(image):
    return cumulative_histogram(image) / np.size(image)

### float -> uint8 conversion function
def to_greyscale(image):
    img_min, img_max = image.min(), image.max()

    raw_normalized = (image - img_min) / (img_max - img_min)
    raw = raw_normalized * MAX_VAL

    return raw.astype(np.uint8)

### transformations
def superresolution(images):
    s = int(np.sqrt(len(images)))
    assert s * s == len(images)

    n, m = images[0].shape

    super_image = np.zeros((n*s, m*s))

    for i, img in enumerate(images):
        x_offset = i // s
        y_offset = i % s

        super_image[x_offset::s, y_offset::s] = img

    return super_image

def gamma_correction(image, gamma):
    image = image.astype(np.float32)

    return np.floor(MAX_VAL * (image/(MAX_VAL)) ** (1/gamma)).astype(np.uint8)

# root mean square error
def rmse(img1, img2):
    return np.sqrt(np.sum((img1 - img2)**2) / np.size(img1))

def main():
    inputs = read_parameters()
    inputs['imglow_files'] = sorted(glob_filenames(inputs['imglow']))

    imglow = [iio.imread(uri) for uri in inputs['imglow_files']]
    imghigh = iio.imread(inputs['imghigh'])

    if inputs['F'] == '3':
        inputs['gamma'] = input()

    img_samples_low = []

    match inputs['F']:
        case '1':
            for image in imglow:
                hist = normalized_cumulative_histogram(image)
                img_samples_low.append(to_greyscale(hist[image]))

        case '2':
            hist = np.zeros(MAX_VAL+1)
            for image in imglow:
                hist += normalized_cumulative_histogram(image)
            hist /= len(imglow)

            for image in imglow:
                img_samples_low.append(to_greyscale(hist[image]))

        case '3':
            gamma = float(inputs['gamma'])
            for image in imglow:
                img_samples_low.append(gamma_correction(image, gamma))

        case _:
            img_samples_low = imglow
    
    new_image = superresolution(img_samples_low)

    print(rmse(new_image, imghigh))

if __name__ == '__main__':
    main()