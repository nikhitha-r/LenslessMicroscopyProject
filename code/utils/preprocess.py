#!/usr/bin/python3

import os
import numpy as np
import cv2
from scipy import stats
import matplotlib.pyplot as plt


if __name__ == '__main__':
    BASE_DIR = 'path_to_basedir'
    
    cells = ['HeLa']
    cropX = 20
    cropY = 15
    paths = {}
    ssim_scores = []
    uqi_scores = []
    for cell in cells:
        # change the paths as per need
        paths.update(dict([(file.split('.')[0], os.path.join(BASE_DIR, cell, 'pos0/micro', file)) for file in os.listdir(
            os.path.join(BASE_DIR, cell, 'pos0/micro')
        ) if file.endswith('png')]))


    for key in paths:
        # key can be replaced with the actual file name to test for
        img = cv2.imread(paths[key], cv2.IMREAD_GRAYSCALE)
        r, c = img.shape
        
        # Crop the borders
        img = img[0:r-cropX, 0:c-cropY]

        # Create the image histogram
        img_hist, bins = np.histogram(img.flatten(), 16, density=True)
        # Determine the cumulative distribution function
        cdf = img_hist.cumsum() 
        
        # Normalise the distribution
        cdf = 255 * cdf / cdf[-1] 
        
        # Find the max value in the image
        maxval = stats.mode(img)
        maxval = maxval[0][0, 0]
        
        # Linear interpolation
        image_equalized = np.interp(img.flatten(), bins[:-1], cdf)
        
        # Reshape the image
        final_img = image_equalized.reshape(img.shape)

        plt.figure()
        # Display the original image
        f, (ax1) = plt.subplots(1, 1, sharey=True, figsize=(15,15))
        ax1.set_title('Original Image')
        ax1.imshow(img, cmap='gray')
        
        
        # Make the final image binary
        final_img[np.where(final_img >= maxval)] = 255
        final_img[np.where(final_img < maxval)] = 0
        # Apply a median blur on the image
        final_img = cv2.medianBlur(np.float32(final_img),5)
        
        # Replace pixels in ooriginal image to determine if the overlap is correct
        img[np.where(final_img == 0)] = 255
        
        f, (a1) = plt.subplots(1, 1, sharey=True, figsize=(15,15))
        a1.imshow(final_img, cmap='gray') 
        a1.set_title("Final preprocessed image")
        f, (a2) = plt.subplots(1, 1, sharey=True, figsize=(15,15))
        a2.imshow(img, cmap='gray')
        a2.set_title("Overlap of Processed and Original Image(white cells)")

        plt.show()
        # Can be removed to analyse more images
        break