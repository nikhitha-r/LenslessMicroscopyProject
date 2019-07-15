import os
import numpy as np
import cv2
from copy import deepcopy
from scipy import stats
from skimage import measure

if __name__ == '__main__':
    # Masks  generated from Matlab code
    BASE_DIR_ORIG = 'path to bf images'
    BASE_DIR_MASK = 'path to masks from Matlab code'
    BASE_DIR_FINAL_MASK = 'path to save the final masks'

    masks, orig = [], []
    masks.extend(os.path.join(BASE_DIR_MASK, file) for file in sorted(os.listdir(BASE_DIR_MASK)) if file.endswith('png'))
    orig.extend(os.path.join(BASE_DIR_ORIG, file) for file in sorted(os.listdir(BASE_DIR_ORIG)) if file.endswith('png'))

    for i in range(0, len(masks)):
        # FIRST STAGE
        img_mask = cv2.imread(masks[i], cv2.IMREAD_GRAYSCALE)
        img_mask[np.where(img_mask > 0)] = 255
        # We cropped the right and bottom borders since we saw a contrast boundary
        # and those were detected as cells from Matlab code
        img_mask = img_mask[:-20, :-20]

        # dilation with (2,2) sphere
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        img_dilation = cv2.dilate(img_mask, kernel, iterations=1)
        img_dilation = cv2.morphologyEx(img_dilation, cv2.MORPH_CLOSE, kernel)

        # get the external contours and fill them up
        contours_dilation, hierarchy = cv2.findContours(img_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        first_stage_mask = deepcopy(img_dilation)
        cv2.drawContours(first_stage_mask, contours_dilation, -1, 255, -1)
        first_stage_mask[np.where(first_stage_mask < 255)] = 0

        # identify the small holes in the mask and fill them up
        negate_mask = deepcopy(first_stage_mask)
        negate_mask = 255 - negate_mask
        contours_hole, hierarchy = cv2.findContours(negate_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        areas_hole = [[cv2.contourArea(contour)] for contour in contours_hole]
        contours_area = zip(areas_hole, contours_hole)
        contours_area = sorted(contours_area, key=lambda x: x[0], reverse=False)
        small_hole = []
        for (area, contour) in contours_area:
            if area[0] <= 500:
                small_hole.append(contour)

        cv2.drawContours(first_stage_mask, small_hole, -1, 255, -1)
        first_stage_mask = cv2.GaussianBlur(first_stage_mask, (3, 3), 0)
        first_stage_mask[np.where(first_stage_mask < 255)] = 0

        # SECOND STAGE
        # Final mask from this stage didn't look good on crowded cases.
        # We ignored the contrast differed region as interference pattern
        # Following is done to fill holes

        # Get the mode of the hole region using the original bright field image
        img_orig = cv2.imread(orig[i], cv2.IMREAD_GRAYSCALE)

        hole_idx = np.where(first_stage_mask == 0)
        cell_idx = np.where(first_stage_mask == 255)

        # img_edit will be used to precisely find the hole region
        img_edit = deepcopy(img_orig)

        img_edit[cell_idx] = 0
        holes = img_edit[hole_idx]
        holes_mode = stats.mode(holes).mode[0]

        # Fill mode +5, -5 as potential holes in img_edit
        for id in range(holes_mode - 5, holes_mode + 6):
            img_edit[np.where(img_edit == id)] = 255

        img_edit[np.where(img_edit < 250)] = 0
        img_edit[np.where(img_edit >= 250)] = 255

        # erode and dilate to fill tiny holes
        img_edit = cv2.erode(img_edit, None, iterations=2)
        img_edit = cv2.dilate(img_edit, None, iterations=2)

        # Identify connected components and consider only the ones with >500 px as holes
        labels = measure.label(img_edit, neighbors=8, background=0)
        mask = np.zeros(img_edit.shape, dtype="uint8")

        # loop over the unique components
        for label in np.unique(labels):
            # if this is the background label, ignore it
            if label == 0:
                continue

            # otherwise, construct the label mask and count the
            # number of pixels
            labelMask = np.zeros(img_edit.shape, dtype="uint8")
            labelMask[labels == label] = 255
            numPixels = cv2.countNonZero(labelMask)

            # if the number of pixels in the component is sufficiently
            # large, then add it to our mask of "large blobs"
            if numPixels > 500:
                mask = cv2.add(mask, labelMask)

        # mask from above has white px for the hole region
        # so negate it to get the cell covered region
        mask = 255 - mask

        # Now identify the connected components >500 px
        # to ensure some small darks spots which are not actually cells are neglected
        labels_final_mask = measure.label(mask, neighbors=8, background=0)
        final_mask = np.zeros(mask.shape, dtype="uint8")

        # loop over the unique components
        for label in np.unique(labels_final_mask):
            # if this is the background label, ignore it
            if label == 0:
                continue

            # otherwise, construct the label mask and count the
            # number of pixels
            labelMask = np.zeros(final_mask.shape, dtype="uint8")
            labelMask[labels_final_mask == label] = 255
            numPixels = cv2.countNonZero(labelMask)

            # if the number of pixels in the component is sufficiently
            # large, then add it to our mask of "large blobs"
            if numPixels > 500:
                final_mask = cv2.add(final_mask, labelMask)

        final_mask = cv2.erode(final_mask, None, iterations=3)

        # Save the final mask
        filename = masks[i].split('/')[-1]
        cv2.imwrite((BASE_DIR_FINAL_MASK + filename), final_mask)

