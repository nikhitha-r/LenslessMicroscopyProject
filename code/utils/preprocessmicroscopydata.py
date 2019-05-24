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
        self.NLM1 = 5.000000
	    self.NLM2 = 7
	    self.NLM3 = 21
        self.bandpassUpper = 100
    
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


    def fastNonLocalDenoising(self,img):
        """
        :param img: grayscale image
        :return: denoised image
        """
        return cv2.fastNlMeansDenoising(cl1, self.NLM1, self.NLM2, self.NLM3)

    def highPassFilter(self,img):
        img = cl1

    rows,cols = img.shape

    nrows = cv2.getOptimalDFTSize(rows)
    ncols = cv2.getOptimalDFTSize(cols)

    right = ncols - cols
    bottom = nrows - rows
    bordertype = cv2.BORDER_CONSTANT
    nimg = cv2.copyMakeBorder(img,0,bottom,0,right,bordertype, value = 0)

    dft2= cv2.dft(np.float32(nimg),flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft2)

    crow,ccol = int(nrows/2) , int(ncols/2)
    mask = np.zeros((nrows,ncols,2),np.uint8)
    #mask[crow-10:crow+10, ccol-10:ccol+10] = 1
    #mask = 1-mask

    n1 = (1.0/np.sqrt(2.0*np.pi*self.bandpassUpper*self.bandpassUpper));
    #n2 = (1.0/np.sqrt(2.0*np.pi*bandpassLower*bandpassLower));

    y = np.linspace(-crow, crow, nrows)
    x = np.linspace(-ccol, ccol, ncols)
    xv, yv = np.meshgrid(x, y)
    g1 = n1*np.exp(-((xv*xv)+(yv*yv))  / (2*self.bandpassUpper*self.bandpassUpper))
    norm_g1 = 1-cv2.normalize(g1, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    mask = np.moveaxis([norm_g1,norm_g1], 0, -1)

    fshift = dft_shift*mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])
    return img_back

    