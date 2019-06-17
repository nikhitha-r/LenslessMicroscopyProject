#steps pipeline:
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


class AreaDetParams(object):

    def __init__(self):
        self.preprocblur_kernel_size = 9
        self.preprocblur_gaussian_sigma = 0.5
        self.variance_kernel_size = 11
        self.variance_gaussian_sigma = 2
        self.variance_absoluteNormalization = -1
        self.varianceThreshold = 45
        self.sobel_absoluteNormalization = -1
        self.sobelThreshold = 30
        self.removeNoise = True
        self.noiseKernelSize = 3
        self.adapthresholdBlockSize = 251
        self.adapthreshold_over = 25
        self.adapthreshold_under = 35
        self.distanceConnect = 3
        self.maxHoleSize = 400
        self.distanceRemove = 5
        self.minParticleSize = 400
        self.smoothingKernelSize = 11
        self.smootingStrength = 7
        self.smmothingThreshold = 230
        self.smoothingExtension = 6
        self.wsShrinkingForSeeds = 15
        self.wsMinSeedSize = 400
        self.wsSeparationDist = 5
        self.activityStackSize = 6           # this must be an even number, as the activity is calculated symmetrically around the image I
        self.activityNormalization = 2
        self.activityThreshold = 45

