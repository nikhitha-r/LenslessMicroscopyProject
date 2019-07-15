import os
import numpy as np
import cv2

if __name__ == '__main__':
    BASE_DIR_PRED_MASK = 'path to model prediciton'
    FINAL_MASKS = 'path to save the final predictions'

    pred_masks = []
    pred_masks.extend(os.path.join(BASE_DIR_PRED_MASK, file) for file in sorted(os.listdir(BASE_DIR_PRED_MASK)) if file.endswith('png'))

    for i in range(len(pred_masks)):
        img_pred_mask = cv2.imread(pred_masks[i], cv2.IMREAD_GRAYSCALE)

        blur = cv2.GaussianBlur(img_pred_mask, (15, 15), 0)

        smooth = cv2.addWeighted(blur, 1.5, img_pred_mask, -0.5, 0)

        intensity_max = np.amax(smooth)
        threshold = int(0.5*intensity_max)
        ret, img_thres = cv2.threshold(smooth, threshold, intensity_max, cv2.THRESH_BINARY)
        img_thres[np.where(img_thres >= 127)] = 255
        img_thres[np.where(img_thres < 127)] = 0

        filename = pred_masks[i].split('/')[-1]
        cv2.imwrite((FINAL_MASKS + filename), img_thres)
