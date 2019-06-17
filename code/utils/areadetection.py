#steps pipeline
#grayscale bild
#foreground mask
#   variance filter
#   gradient filter
#   adaptive thresholding
#   combination using bitwise or
#morphological closing
# remove foregraound area brigdes
# gaussian blur to smooth outlines
#watershed
#visualization
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt



def varianceFilter(raw,isGaussian,ksize,gsigma,absnorm):
	raw64F = np.float64(raw)
	if isGaussian :
		mean = cv2.GaussianBlur(raw64F, (ksize,ksize), gsigma)
		meanSqr = cv2.GaussianBlur(raw64F**2, (ksize,ksize), gsigma)
		variance = cv2.absdiff(meanSqr, mean**2)
		variance = cv2.sqrt(variance)
		if absnorm <= 0 :
			varianceNormed = cv2.normalize(variance, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
			variance8Bit = np.uint8(varianceNormed)
		
		else:
			varianceNormed = variance/absnorm
			variance8Bit = np.uint8(varianceNormed)

	else:
		#KPY: this condition doesn't run -> cv2.Blur not in cv2.cv2 module ??!
		mean    = cv2.Blur(raw64F,    (ksize,ksize))
		meanSqr = cv2.Blur(raw64F**2, (ksize,ksize))
		variance = cv2.absdiff(meanSqr, mean**2)
		variance = cv2.sqrt(variance)
		if absnorm <= 0 :
			varianceNormed = cv2.normalize(variance, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
			variance8Bit = np.uint8(varianceNormed)
		
		else:
			varianceNormed = variance/absnorm
			variance8Bit = np.uint8(varianceNormed)

	return variance8Bit

def gradientFilter(raw,absnorm):
	sobelx64f = cv2.Sobel(raw,cv2.CV_64F,1,0,ksize=7) #default scaling, probably=1
	sobely64f = cv2.Sobel(raw,cv2.CV_64F,0,1,ksize=7) #KPY: why sobely64 computed in x-direction?? ->cv2.Sobel(raw,cv2.CV_64F,0,1,ksize=7) 
	abs_sobelx64f = np.absolute(sobelx64f)
	abs_sobely64f = np.absolute(sobely64f)
	abs_sobel64f = abs_sobelx64f + abs_sobely64f
	if absnorm <= 0 :
		sobelNormed = cv2.normalize(abs_sobel64f, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
		sobel8Bit = np.uint8(sobelNormed)

	else:
		sobelNormed = variance/absnorm
		sobel8Bit = np.uint8(varianceNormed)

	return sobel8Bit

def adaptiveThresh(raw,foregroundMask,removeNoise,noiseKsize,thresholdBlockSize,threshold_under,threshold_over):
	noiseElement = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(noiseKsize,noiseKsize))
	over = cv2.adaptiveThreshold(raw, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, thresholdBlockSize, -threshold_over)
	under = cv2.adaptiveThreshold(raw, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, thresholdBlockSize, threshold_under)
	filterResult = cv2.bitwise_or(over, under)
	if removeNoise :
		filterResult = cv2.morphologyEx(filterResult, cv2.MORPH_OPEN, noiseElement)
	foregroundMask = cv2.bitwise_or(foregroundMask, filterResult)
	return foregroundMask

def thresholdFiltered(filtered,foregroundMask,threshold,removeNoise,noiseKsize):
	noiseElement = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(noiseKsize,noiseKsize))
	ret, filterResult = cv2.threshold(filtered, threshold, 255, cv2.THRESH_BINARY)
	if removeNoise :
		filterResult = cv2.morphologyEx(filterResult, cv2.MORPH_OPEN, noiseElement)

	foregroundMask = cv2.bitwise_or(foregroundMask, filterResult)
	return foregroundMask


def fastFilterParticles(inputMask, minSize) :
	im2, contours, hierarchy = cv2.findContours(inputMask.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
	i = 0
	for cnt in contours:
		if(hierarchy[0][i][3] == -1) :
			if(cv2.contourArea(cnt) < minSize) :
				cv2.drawContours(inputMask, [cnt], 0, 0, -1)
		i += 1

def fastFillHoles(inputMask, maxSize, extrapolateBorders = False) :
	""" The fastFillHoles only uses a two level hierachy. This means that the foreground
	areas (white) inside a hole are regarded as "hole-area" when the area of the hole is
	calculated. The extrapolateBorders option allows filling holes that are adjacent to
	the border (i.e., the image is extented with one foregroundpixel at each border)
	"""
	if(extrapolateBorders == True) :
		replicate = cv2.copyMakeBorder(inputMask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value = 255)
		im2, contours, hierarchy = cv2.findContours(replicate, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
	else :
		im2, contours, hierarchy = cv2.findContours(inputMask.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
	i = 0
	for cnt in contours:
		if(hierarchy[0][i][3] != -1) :
			if(cv2.contourArea(cnt) < maxSize) :
				if(extrapolateBorders == True) :
					cv2.drawContours(inputMask, [cnt], 0, 255, -1, offset = (-1,-1))
				else :
					cv2.drawContours(inputMask, [cnt], 0, 255, -1)
		i += 1

def watershed(inputMask, seedShrinking, minSeedSize, separation) :
	element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (seedShrinking, seedShrinking))
	markersmask = cv2.morphologyEx(inputMask, cv2.MORPH_ERODE, element, borderType=cv2.BORDER_CONSTANT, borderValue=255)
	markers  = np.zeros(inputMask.shape, np.int32)
	im2, contours, hierarchy = cv2.findContours(markersmask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
	i = 0
	for cnt in contours:
		if(hierarchy[0][i][3] == -1) :
			if(math.fabs(cv2.contourArea(cnt)) > minSeedSize) :
				cv2.drawContours(markers, [cnt], 0, i+1, cv2.FILLED)
				i += 1
	#--------------------------------------------------------------------------
	# Do the watershed segmentation
	whatever = cv2.cvtColor(inputMask, cv2.COLOR_GRAY2RGB)
	cv2.watershed(whatever, markers)
	#--------------------------------------------------------------------------
	ret, temp_32f = cv2.threshold(np.float32(markers), 0, 255, cv2.THRESH_BINARY)
	temp_uint8 = np.uint8(temp_32f)
	crop = temp_uint8[1:(temp_uint8.shape[0] - 1), 1:(temp_uint8.shape[1] - 1)]
	rep = cv2.copyMakeBorder(crop, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value = 255)
	element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (separation, separation))
	temp2 = cv2.morphologyEx(rep, cv2.MORPH_ERODE, element, borderType=cv2.BORDER_CONSTANT, borderValue=255)
	cv2.bitwise_and(inputMask, temp2, inputMask)

def calculateGainLoss(mat0, mat1):
	"""
	:param mat0:
	:param mat1:
	"""
	gain = cv2.subtract(mat1, mat0)
	loss = cv2.subtract(mat0, mat1)
	return gain, loss

def timeactivity(images,activityStackSize,activityNormalization):
	"""
	:param images: needs a list of opencv mat-images
	"""
	raw = images[0]
	sumGain = np.zeros(raw.shape, np.int32)
	sumLoss = np.zeros(raw.shape, np.int32)
	if len(images) >= activityStackSize :
		for i in range(0, activityStackSize) :
			img0 = images[i]
			img1 = images[i+1]
			gain, loss = calculateGainLoss(img0, img1)
			sumGain += gain
			sumLoss += loss
		netGain = cv2.subtract(sumGain, sumLoss)
		netGain = netGain.clip(min=0)
		netGain = netGain/activityNormalization
		netLoss = cv2.subtract(sumLoss, sumGain)
		netLoss = netLoss.clip(min=0)
		netLoss = netLoss/activityNormalization
	#-----------------------------------------------------------
		netGain8Bit = np.uint8(netGain)
		netLoss8Bit = np.uint8(netLoss)
		empty = np.zeros(netLoss8Bit.shape, np.uint8)
		netGL = cv2.merge((netLoss8Bit,empty,netGain8Bit))
		activity = cv2.add(netGain8Bit,netLoss8Bit)
		return activity
	

def getImageActivity_Check(netLoss8Bit,netGain8Bit):
	empty = np.zeros(netLoss8Bit.shape, np.uint8)
	netGL = cv2.merge((netLoss8Bit,empty,netGain8Bit))
	return netGL


def filterForBrightfield(img):
	"""TODO 
	"""
	#int canny_blur_size = 5;
	#
	#blur(input, result, cv::Size(_blur_size, _blur_size)); <= X,Y
	#
	#int _lowThreshold = 18;
	#int canny_ratio = 3;
	#int canny_kernel_size = 3;
	#
	# Canny( result, result, _lowThreshold, _lowThreshold*canny_ratio, _kernel_size)
	#
	#The following operation is to cover the inner part, circumsized by the detected edges from the Canny filter
	#having the edges -> dilate by an amount of ca. 3 Pixels (try to close area or close contour to fill hole)
	#then erode to shrink foreground area to original size

class AreaDetection(object):
	def __init__(self,paramset):
		self.params = paramset

	def compFinalSegmentation(self,raw):
		raw = self.preBlurrImage(raw)
		varFiltered = self.doVarianceFilter(raw)
		gradFiltered = self.doGradientFilter(raw)
		foregnd = self.threshNcompForegroundMask(raw,varFiltered,gradFiltered)
		finalsegmented = self.postProcessMaskPipeline(foregnd)    
		return finalsegmented

	def compFinalSegmentationTimeActivity(self,images):
		selectedIdx = int(np.around(len(images)/2))-1
		raw = images[selectedIdx]
		raw = self.preBlurrImage(raw)
		varFiltered = self.doVarianceFilter(raw)
		gradFiltered = self.doGradientFilter(raw)
		activityFiltered = self.doTimeActivity(images)
		foregnd = self.threshNcompForegroundMaskActivity(raw,varFiltered,gradFiltered,activityFiltered)
		finalsegmented = self.postProcessMaskPipeline(foregnd)    
		return finalsegmented

	def getSelectedIdxfromImages(self,img_len):
		"""
		:param img_len: length of images list, len(images)
		"""
		return int(np.around(img_len/2))-1

	def preBlurrImage(self,raw):
		"""
		 blurring function to improve segmentation results
		:param raw: greyscaled and preprocessed image (usually background subtraction and clahe)
		"""
		return cv2.GaussianBlur(raw, (self.params.preprocblur_kernel_size,self.params.preprocblur_kernel_size), self.params.preprocblur_gaussian_sigma)

	def doVarianceFilter(self,raw):
		return varianceFilter(raw, True,
			self.params.variance_kernel_size,
			self.params.variance_gaussian_sigma,
			self.params.variance_absoluteNormalization)

	
	def doGradientFilter(self,raw):
		return gradientFilter(raw,self.params.sobel_absoluteNormalization)


	def doAdaptiveThreshRaw(self,raw,foreground):
		return adaptiveThresh(raw,foreground,self.params.removeNoise,self.params.noiseKernelSize,
			self.params.adapthresholdBlockSize, 
			self.params.adapthreshold_under, 
			self.params.adapthreshold_over)

	def doTimeActivity(self,images):
		return timeactivity(images,self.params.activityStackSize,self.params.activityNormalization)

	def threshNcompForegroundMask(self,raw,varFiltered,gradFiltered):
		foregroundMask = np.zeros(raw.shape, np.uint8)
		foregroundMask = self.doAdaptiveThreshRaw(raw,foregroundMask)
		foregroundMask = thresholdFiltered(varFiltered,foregroundMask,
			self.params.varianceThreshold, self.params.removeNoise,self.params.noiseKernelSize)
		foregroundMask = thresholdFiltered(gradFiltered,foregroundMask,
			self.params.sobelThreshold, self.params.removeNoise,self.params.noiseKernelSize)
		return foregroundMask

	def threshNcompForegroundMaskActivity(self,raw,varFiltered,gradFiltered,activityFiltered):
		foregroundMask = np.zeros(raw.shape, np.uint8)
		foregroundMask = self.doAdaptiveThreshRaw(raw,foregroundMask)
		foregroundMask = thresholdFiltered(varFiltered,foregroundMask,
			self.params.varianceThreshold, self.params.removeNoise,self.params.noiseKernelSize)
		foregroundMask = thresholdFiltered(gradFiltered,foregroundMask,
			self.params.sobelThreshold, self.params.removeNoise,self.params.noiseKernelSize)
		foregroundMask = thresholdFiltered(activityFiltered,foregroundMask,
			self.params.activityThreshold, self.params.removeNoise,self.params.noiseKernelSize)
		return foregroundMask

	def postProcessMaskPipeline(self,foregroundMask):
		inverse = 255 - foregroundMask
		#step 1
		distanceMap = cv2.distanceTransform(inverse, cv2.DIST_L2, cv2.DIST_MASK_3)
		ret, distanceMap = cv2.threshold(distanceMap, self.params.distanceConnect, 255, cv2.THRESH_BINARY)
		inverse = np.uint8(distanceMap)
		element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*self.params.distanceConnect,2*self.params.distanceConnect))
		inverse = cv2.morphologyEx(inverse, cv2.MORPH_DILATE, element, inverse, (-1,-1), 1, cv2.BORDER_CONSTANT, 0)
		foregroundMask = 255 - inverse
		fastFillHoles(foregroundMask, self.params.maxHoleSize, True)
		#step 2
		distanceMap = cv2.distanceTransform(foregroundMask, cv2.DIST_L2, cv2.DIST_MASK_3)
		ret, distanceMap = cv2.threshold(distanceMap, self.params.distanceRemove, 255, cv2.THRESH_BINARY)
		foregroundMask = np.uint8(distanceMap)
		element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*self.params.distanceRemove, 2*self.params.distanceRemove))
		foregroundMask = cv2.morphologyEx(foregroundMask, cv2.MORPH_DILATE, element, foregroundMask, (-1,-1), 1, cv2.BORDER_CONSTANT, 0)
		fastFilterParticles(foregroundMask, self.params.minParticleSize)
		#step 3
		foregroundMask = cv2.GaussianBlur(foregroundMask, (self.params.smoothingKernelSize, self.params.smoothingKernelSize), 
			self.params.smootingStrength, self.params.smootingStrength)
		ret, foregroundMask = cv2.threshold(foregroundMask, self.params.smmothingThreshold, 255, cv2.THRESH_BINARY)
		element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.params.smoothingExtension, self.params.smoothingExtension))
		foregroundMask = cv2.morphologyEx(foregroundMask, cv2.MORPH_DILATE, element, borderType=cv2.BORDER_CONSTANT, borderValue=0)
		#step 4
		watershed(foregroundMask, self.params.wsShrinkingForSeeds, self.params.wsMinSeedSize, self.params.wsSeparationDist)
		return foregroundMask

	def drawContoursImg(self,raw,mask):
		resImg = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
		#size = raw.shape[0], raw.shape[1], 3
		im2, contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
		cv2.drawContours(resImg,contours,-1,(0,0,255), 2)
		return resImg

	def drawContoursImgFilled(self,raw,mask):
		resImg = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
		#size = raw.shape[0], raw.shape[1], 3
		im2, contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
		cv2.drawContours(resImg,contours,-1,(0,0,255), 2)
		resImg[mask==255,1]=0.4*resImg[mask==255,1]
		resImg[mask==255,2]=0.4*resImg[mask==255,2]
		return resImg
