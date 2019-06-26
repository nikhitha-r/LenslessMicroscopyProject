#!/usr/bin/python3

import tensorflow as tf
import tensorflow.keras.backend as K

def dice_coeff(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + K.epsilon()) / (K.sum(y_true_f) + K.sum(y_pred_f) + K.epsilon())

def dice_coeff_loss():
    def dice_coeff_loss_fixed(y_true, y_pred):
        return 1 - dice_coeff(y_true, y_pred)
    return dice_coeff_loss_fixed

def bce_dice_loss():
    def bce_dice_loss_fixed(y_true, y_pred):
        return tf.keras.losses.binary_crossentropy(y_true, y_pred) + dice_coeff_loss(y_true, y_pred)
    return bce_dice_loss_fixed