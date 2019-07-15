#!/usr/bin/python3

import argparse
import pathlib
import os
import cv2
import numpy as np
import tensorflow as tf
from model.unet import UNet
from model.linknet import LinkNet
from data.pipeline import input_pipeline
from utils.loss import *
import constants

def predict_full_image(model, patches: tf.Tensor, img_shape: (int, int), shape=(224, 224, 3)):
    """
    Takes an image, splits it into patches, predicts the cell mask
    and then returns the stitched mask to match the original image size
    Args:
        img_shape: the shape of the original image
        patches: the dataset iterator of images to predict the mask for (in patches [rows * cols, h, w, c])

    Returns: the original image and the mask (tf.Tensor, tf.Tensor)

    """
    patch_shape = patches.get_shape()
    patch_annot = model.predict(patches, batch_size=patch_shape[0], steps=1)
    patch_img = tf.keras.backend.eval(patches)

    # get number of batch rows and cols in original image
    nrow = int(np.ceil(img_shape[0] / shape[0]))
    ncol = int(np.ceil(img_shape[1] / shape[1]))
    # get padding introduced by tf.image.extract_patches
    res_row = (nrow * shape[0] - img_shape[0]) // 2
    res_col = (ncol * shape[1] - img_shape[1]) // 2

    # reshape to (nrow, ncol, h, w)
    # corresponding to the tf.image.extract_patches output
    mask = patch_annot.reshape((nrow, ncol,) + shape[:-1])
    img = patch_img.reshape((nrow, ncol,) + shape)

    # stack tiles
    mask = np.concatenate(mask, axis=1).swapaxes(1, 2)
    mask = np.concatenate(mask, axis=0).swapaxes(0, 1)
    mask = mask[res_row:res_row + img_shape[0], res_col:res_col + img_shape[1]]
    img = np.concatenate(img, axis=1).swapaxes(1, 2)
    img = np.concatenate(img, axis=0).swapaxes(0, 1)
    img = img[res_row:res_row + img_shape[0], res_col:res_col + img_shape[1], :]
    return mask, img


if __name__ == '__main__':

    tf.enable_eager_execution()

    parser = argparse.ArgumentParser()

    # data in/out
    parser.add_argument('-i', '--data_dir', type=str, required=True,
                        help='Dataset path.')
    parser.add_argument('-m', '--model', default='unet', choices=['unet', 'linknet'], required=True)
    parser.add_argument('-w', '--weight_file', type=str, required=False,
                        help='Load model weights from this dir.')

    args = parser.parse_args()

    # model setup
    if args.model in ('unet'):
        model = UNet().build_unet_model()
        shape = (224,224,3)
    else:
        model = LinkNet().build_linknet_model()
        shape = (256, 256, 3)

    if args.weight_file is not None:
        model.load_weights(args.weight_file)

    data_dir = pathlib.Path(args.data_dir)

    image_path = '**/test/images/*.png'
    annotation_path = '**/test/annotations/*.png'

    # get image names without suffix
    img_names = [path.stem for path in sorted(data_dir.glob(annotation_path))]
    # make predictions folder
    pred_path = data_dir.joinpath('test/predictions')
    pred_path.mkdir(exist_ok=True)

    test_set = input_pipeline(dirname=args.data_dir,
                              imagepath=image_path,
                              annotationpath=annotation_path,
                              is_training=False,
                              use_augmentation=False,
                              batch_size=1).make_one_shot_iterator()

    for path in img_names:
        test_img, _ = test_set.get_next()
        celltype = path.split("_")
        if(len(celltype) > 2):
            if celltype[0] == '3T3' and celltype[1] == 'pos0':
                img_shape = constants.T3_pos0
            elif celltype[0] == '3T3' and celltype[1] == 'pos1':
                img_shape = constants.T3_pos1
            elif celltype[0] == 'A549' and celltype[1] == 'pos0':
                img_shape = constants.A549_pos0
            elif celltype[0] == 'Hela' and celltype[1] == 'pos0':
                img_shape = constants.HeLa_pos0
            elif celltype[0] == 'Hela' and celltype[1] == 'pos1':
                img_shape = constants.HeLa_pos1
            elif celltype[0] == 'huh' and celltype[1] == 'pos0':
                img_shape = constants.Huh_pos0
            else:
                raise Exception("Wrong file path")
            pred, _ = predict_full_image(model, test_img, img_shape=img_shape, shape=shape)
            np.save(str(pred_path.joinpath(path)), pred)
            cv2.imwrite(str(pred_path.joinpath(path + '.png')), pred * 255)
            print(str(pred_path.joinpath(path)))