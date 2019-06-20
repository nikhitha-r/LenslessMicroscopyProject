import cv2
import numpy as np
import json
import pdb

#selPts = []
selPts_led = []
selPts_micro = []
new_clck_Pt  = []
selecting = False

def createCompImage(im1,im2):
    offsetx=im1.shape[1]
    compimg = np.zeros((np.maximum(im1.shape[0],im2.shape[0]),im1.shape[1]+im2.shape[1],3))
    compimg[0:im1.shape[0],0:im1.shape[1],:]=im1
    compimg[0:im2.shape[0],offsetx:(offsetx+im2.shape[1]),:]=im2
    return compimg

def click_select(event, x, y, flags, param):
    global selPts, selecting, new_clck_Pt

    if event ==cv2.EVENT_LBUTTONDOWN:
        new_clck_Pt = [(x, y)]
        selecting = True

    elif event == cv2.EVENT_MOUSEMOVE and selecting:
        new_clck_Pt = [(x, y)]

    elif event == cv2.EVENT_LBUTTONUP:
        if x<im1_thumb.shape[1]:
            selPts_led.append((x, y))
        else:
            selPts_micro.append((x, y))
        selecting = False

        cv2.circle(compimg,(new_clck_Pt[0][0], new_clck_Pt[0][1]), 5, (0,255,0), 2)
        cv2.imshow("image",compimg)

def convertPoints(ledpts,micropts,dscaleratio,imledwidth):
    if not (len(ledpts) == len(micropts)):
        raise Exception('selected points on led image {} not equal to selected points on microscopy image {}'.format(len(ledpts),len(micropts)))
    data = []
    for idx,led in enumerate(ledpts):
        newpt={}
        newpt_led = {}
        newpt_led['x'] = int(led[0]*dscaleratio)
        newpt_led['y'] = int(led[1]*dscaleratio)
        newpt_micro = {}
        newpt_micro['x'] = int(micropts[idx][0]*dscaleratio)-imledwidth
        newpt_micro['y'] = int(micropts[idx][1]*dscaleratio)
        newpt['id'] = idx
        newpt['led'] = newpt_led
        newpt['micro'] = newpt_micro
        data.append(newpt)

    return data


def compAffineTrafo(pts_led,pts_micro,dscaleratio,imledwidth):

    pts_led_scaled = []
    pts_micro_scaled = []
    #pdb.set_trace()
    for idx,led in enumerate(pts_led):
        pts_led_scaled.append((int(led[0]*dscaleratio),int(led[1]*dscaleratio)))
        pts_micro_scaled.append((int(pts_micro[idx][0]*dscaleratio)-imledwidth,int(pts_micro[idx][1]*dscaleratio)))
    
    M = cv2.getAffineTransform(np.float32(np.array(pts_micro_scaled)),np.float32(np.array(pts_led_scaled)))
    return M

def compHomographicTrafo(pts_led,pts_micro,dscaleratio,imledwidth):

    pts_led_scaled = []
    pts_micro_scaled = []
    #pdb.set_trace()
    for idx,led in enumerate(pts_led):
        pts_led_scaled.append((int(led[0]*dscaleratio),int(led[1]*dscaleratio)))
        pts_micro_scaled.append((int(pts_micro[idx][0]*dscaleratio)-imledwidth,int(pts_micro[idx][1]*dscaleratio)))
    
    h, status = cv2.findHomography(np.float32(np.array(pts_micro_scaled)),np.float32(np.array(pts_led_scaled)))
    return h, status

def storeDataJson(data,filename):
    with open(filename, 'w') as fp:
        json.dump(data, fp)

impath  =['cmos00000.png','holo00000.png']# ['00000049.png','micro_ch10000.png']#
thumbratio = 2.0
im1 = cv2.imread(impath[0],1)
im2 = cv2.imread(impath[1],1)
#im2 = cv2.flip(im2, 0) #brauchts nicht bilder sind aber schon ordentlich in y verschoben!
im1_thumb = cv2.resize(im1,(int(im1.shape[1]/thumbratio),int(im1.shape[0]/thumbratio)))
im2_thumb = cv2.resize(im2,(int(im2.shape[1]/thumbratio),int(im2.shape[0]/thumbratio)))

compimg = createCompImage(im1_thumb,im2_thumb)
compimg=compimg/255

storedImg = compimg.copy()

cv2.namedWindow("image")
cv2.setMouseCallback("image",click_select)
#cv2.imshow('image',compimg/255)
#cv2.waitKey(0)
while True:
    if not selecting:
        cv2.imshow('image', compimg)
    elif selecting and new_clck_Pt:
        compimg_cpy = compimg.copy()
        cv2.circle(compimg_cpy,(new_clck_Pt[0][0], new_clck_Pt[0][1]), 5, (255,0,0), 2)
        cv2.imshow('image', compimg_cpy)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("r"):
        compimg = clone.copy()

    elif key == ord("c"):
        break

if len(selPts_led)>0 and len(selPts_micro)>0:
    ptsdata = convertPoints(selPts_led,selPts_micro,thumbratio,im1.shape[1])
    print(ptsdata)
    storeDataJson(ptsdata,'coregpts.json')
    #trafo = compAffineTrafo(selPts_led,selPts_micro,thumbratio,im1.shape[1])
    #print(trafo)
    #aligned_micro = cv2.warpAffine(im2, trafo, (im1.shape[1], im1.shape[0]))
    h,status = compHomographicTrafo(selPts_led,selPts_micro,thumbratio,im1.shape[1])
    print(h)
    aligned_micro = cv2.warpPerspective(im2, h, (im1.shape[1], im1.shape[0]))
    cv2.imwrite('aligned_micro.png',aligned_micro)
    #roi = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
    #cv2.imshow("ROI",roi)
    #cv2.waitKey(0)

cv2.destroyAllWindows()