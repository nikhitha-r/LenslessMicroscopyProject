import datahandler as dh
import areaparameters as aparam
import areadetection  as adet

import cv2


if __name__=="__main__":
    parameter = aparam.AreaDetParams()
    parameter.activityThreshold = 200
    parameter.sobelThreshold = 80
    parameter.variance_gaussian_sigma = 5
    parameter.variance_kernel_size = 19
    parameter.wsMinSeedSize = 250
    parameter.maxHoleSize = 250
    parameter.minParticleSize = 400
    parameter.smoothingExtension = 10
    parameter.smmothingThreshold = 210
    parameter.adapthreshold_over = 45
    parameter.adapthreshold_under = 45
    parameter.adapthresholdBlockSize = 101
    parameter.varianceThreshold = 80
    parameter.distanceRemove = 4

    parameter.__dict__

    aAnalysis = adet.AreaDetection(parameter)

    imggray = cv2.imread('exampleImg.png',0)
    print(imggray.shape)
    img_mask = aAnalysis.compFinalSegmentation(imggray)
    imgCont = aAnalysis.drawContoursImgFilled(imggray,img_mask)
    cv2.imwrite('exampleImgSegmented.png',imgCont)
    
