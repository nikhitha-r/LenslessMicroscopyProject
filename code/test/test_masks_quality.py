import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

if __name__ == '__main__':
    BASE_DIR_MASK_MED_5 = 'path_to_masks_medianfilter_5'
    BASE_DIR_MASK_MED_3 = 'path_to_masks_medianfilter_3'
    BASE_DIR_MASK_GAU_2 = 'path_to_masks_gaussian_2'
    BASE_DIR_ORIG = 'path_to_original_image'

    masks_med_5, masks_med_3, masks_gau_2, orig = [], [], [], []
    masks_med_5.extend(os.path.join(BASE_DIR_MASK_MED_5, file) for file in sorted(os.listdir(BASE_DIR_MASK_MED_5)) if file.endswith('png'))
    masks_med_3.extend(os.path.join(BASE_DIR_MASK_MED_3, file) for file in sorted(os.listdir(BASE_DIR_MASK_MED_3)) if file.endswith('png'))
    masks_gau_2.extend(os.path.join(BASE_DIR_MASK_GAU_2, file) for file in sorted(os.listdir(BASE_DIR_MASK_GAU_2)) if file.endswith('png'))
    orig.extend(os.path.join(BASE_DIR_ORIG, file) for file in sorted(os.listdir(BASE_DIR_ORIG)) if file.endswith('png'))

    video_gau = 'gau2VSmed5.avi'
    video_med = 'med5VSmed3.avi'
    height, width = 500, 1500

    video1 = cv2.VideoWriter(video_gau, 0, 3, (width, height))
    video2 = cv2.VideoWriter(video_med, 0, 3, (width, height))

    area_med_5, area_med_3, area_gau_2 = [], [], []
    for i in range(0, len(orig)):

        img_orig = cv2.imread(orig[i])
        img_orig = img_orig[:-20, :-20, :]
        img_orig = cv2.resize(img_orig, (500, 500))
        cv2.putText(img_orig, "Original Image", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2, cv2.LINE_AA)

        img_med_5 = cv2.imread(masks_med_5[i])
        img_med_5 = img_med_5[:-20, :-20, :]
        area = np.count_nonzero(img_med_5) / (img_med_5.shape[0]*img_med_5.shape[1])
        area_med_5.append(area)
        img_med_5 = cv2.resize(img_med_5, (500, 500))
        cv2.putText(img_med_5, "Median 5*5 filter", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        img_med_3 = cv2.imread(masks_med_3[i])
        img_med_3 = img_med_3[:-20, :-20, :]
        area = np.count_nonzero(img_med_3) / (img_med_3.shape[0] * img_med_3.shape[1])
        area_med_3.append(area)
        img_med_3 = cv2.resize(img_med_3, (500, 500))
        cv2.putText(img_med_3, "Median 3*3 filter", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        img_gau_2 = cv2.imread(masks_gau_2[i])
        img_gau_2 = img_gau_2[:-20, :-20, :]
        area = np.count_nonzero(img_gau_2) / (img_gau_2.shape[0] * img_gau_2.shape[1])
        area_gau_2.append(area)
        img_gau_2 = cv2.resize(img_gau_2, (500, 500))
        cv2.putText(img_gau_2, "Gaussian with sigma 2", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        img1 = np.concatenate((img_gau_2, img_orig, img_med_5), axis=1)
        img2 = np.concatenate((img_med_5, img_orig, img_med_3), axis=1)
        video1.write(img1)
        video2.write(img2)

    f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(15, 15))
    ax1.plot(area_gau_2)
    ax1.set_title('Gaussian Filter')
    ax1.set_xlabel('Number of images')
    ax1.set_ylabel('Area covered')
    ax2.plot(area_med_5)
    ax2.set_title('Median 5*5 Filter')
    ax2.set_xlabel('Number of images')
    ax2.set_ylabel('Area covered')
    ax3.plot(area_med_3)
    ax3.set_title('Median 3*3 Filter')
    ax3.set_xlabel('Number of images')
    ax3.set_ylabel('Area covered')
    plt.show()

    cv2.destroyAllWindows()
    video1.release()
    video2.release()


