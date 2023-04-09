#!/usr/bin/env python3

# SCC0251 - Image Processing and Analysis (2023.1)
# Assignment A1 - Enhancement and Superresolution
# Frederico Bulhões de Souza Ribeiro
# NUSP - 11208440

import numpy as np
import imageio.v3 as iio
import glob
import sys

MAX_VAL = 255

### input auxiliary functions
def read_parameters():
    # returns dictionary mapping inputs to values
    return {k : input() for k in ['imglow', 'imghigh', 'F']}

def glob_filenames(base_filename):
    # returns all filenames matching a given prefix
    return glob.glob(f'{base_filename}*')

### histogram functions
def cumulative_histogram(image):
    # returns the cumulative sum of the histogram of the value of the pixels of the image
    return np.histogram(image.flatten(), bins=MAX_VAL+1, range=(0, MAX_VAL))[0].cumsum()

def normalized_cumulative_histogram(image):
    # returns the cumulative histogram normalized such that max(hist) = 1
    # the normalized cumulative histogram is such that for every value x
    # norm_cum_hist[x] equals the probability of a pixel being of value  
    # smaller or equal than x in the original image
    return cumulative_histogram(image) / np.size(image)

### float -> uint8 conversion function
def to_greyscale(image, normalize = False):
    # converts the image to greyscale, considering img_min and img_max as the range of the 
    # pixels of the image
    # if normalize is true, considers the smallest value of the image as 0 and the max as 255
    img_min, img_max = (image.min(), image.max()) if normalize else (0, 1)

    raw_normalized = (image - img_min) / (img_max - img_min)
    raw = raw_normalized * MAX_VAL

    return raw.astype(np.uint8)

### transformations
def superresolution(images):
    # expects an array of images, containing a square number of images to makeup the supperresolution
    s = int(np.sqrt(len(images)))
    assert s * s == len(images)

    n, m = images[0].shape

    super_image = np.zeros((n*s, m*s))

    # calculates the offset of each original image in the superresolution and then
    # sets the pixels of the final image with given offset and stride equal to the 
    # side of the square image array
    for i, img in enumerate(images):
        x_offset = i // s
        y_offset = i % s

        super_image[x_offset::s, y_offset::s] = img

    return super_image

def gamma_correction(image, gamma):
    # returns the image with an applied pixelwise gamma correction
    image = image.astype(np.float32)

    return np.floor(MAX_VAL * (image/(MAX_VAL)) ** (1/gamma)).astype(np.uint8)

# root mean square error
def rmse(img1, img2):
    # returns the root mean square error of the difference of two images
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
            # enhances each image by their respective normalized histogram
            for image in imglow:
                hist = normalized_cumulative_histogram(image)
                img_samples_low.append(to_greyscale(hist[image]))

        case '2':
            # calculates the normalized histogram for all images combined and then 
            # enhances each image with it
            hist = np.zeros(MAX_VAL+1)
            for image in imglow:
                hist += normalized_cumulative_histogram(image)
            hist /= len(imglow)

            for image in imglow:
                img_samples_low.append(to_greyscale(hist[image]))

        case '3':
            # applies gamma correction in each image
            gamma = float(inputs['gamma'])
            for image in imglow:
                img_samples_low.append(gamma_correction(image, gamma))

        case _:
            # returns original images
            img_samples_low = imglow
    
    # applies superresolution in the processed samples
    new_image = superresolution(img_samples_low)

    print(rmse(new_image, imghigh))

if __name__ == '__main__':
    main()