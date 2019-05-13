import os
import numpy as np
import cv2

import matplotlib.pyplot as plt


class ImagePreProcessor(object):
    def __init__(self):
        self.blurSizeX=201
        self.blurSizeY=201
        self.claheClip=6.0
        self.claheTileSize=12
     
    
    def subtractBackground(self,img):
        """
        subtracts inhomogeneous background e.g. due to tilted illumination
        :param img: a grayscale image
        :return: background subtracted image
        """
        blur = cv2.GaussianBlur(img,(self.blurSizeX,self.blurSizeY),0)
        subtractedImg = cv2.subtract(img,blur)
        cv2.normalize(subtractedImg,subtractedImg, 0, 255, cv2.NORM_MINMAX)
        return subtractedImg
    
    
    def histogramEq(self,img):
        """
        performs contrast limited adaptive histogram equalization (https://docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html)
        :param img: a grayscale image
        :return: histogram equalized image
        """
        clahe = cv2.createCLAHE(clipLimit=self.claheClip, tileGridSize=(self.claheTileSize,self.claheTileSize))
        return clahe.apply(img)

    