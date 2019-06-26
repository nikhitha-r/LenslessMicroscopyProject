#!/usr/bin/python3

import os
import argparse
import pathlib
import json
import numpy as np
import tensorflow as tf
from data.pipeline import input_pipeline
from model.unet import UNet
from utils.loss import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Input/ Output
    parser.add_argument('-i', '--data_dir', type=str, required=True,
                        help='Dataset path.')
    parser.add_argument('-c', '--channels', type=int, default=3,
                        help='The number of channels of images')
    parser.add_argument('-o', '--save_dir', type=str, required=True,
                        help='Save model and and logs to this dir.')
    parser.add_argument('-w', '--weight_file', type=str,
                        help='Load model weights from this dir.')

    # Data Handling
    parser.add_argument('-a', '--no-augment', action='store_true',
                        default=False, help='wether to use data augmentation')

    # Model
    parser.add_argument('-m', '--model', type=str, required=True,
                        choices=['unet'])
    parser.add_argument('-f', '--filters', type=int, nargs='+',
                        default=(64, 128, 256, 512, 1024),
                        help='the number of filters per block, the last filter'
                             'defines the bottleneck (UNet only)')
    parser.add_argument('-conv', '--conv_ops', type=int, nargs='+',
                        default=(3, 3),
                        help='the kernel size of each convolutional operation'
                             'in each block (UNet only)')
    parser.add_argument('--loss', type=str, default='mse',
                        choices=['mse', 'mae', 'binary_crossentropy', 'dice_coeff', 'bce_dice'],
                        help='the loss function')
    parser.add_argument('-cw', '--class-weight', type=float, default=None,
                        help='The class weight for binary crossentropy')

    # Specs
    parser.add_argument('-l', '--learning_rate', type=float, default=1e-5,
                        help='Adam learning rate')
    parser.add_argument('-b', '--batch_size', type=int, default=32,
                        help='Batch size during training.')
    parser.add_argument('-p', '--drop_prob', type=float, default=0.5,
                        help='The dropout probability. None is no dropout')
    parser.add_argument('-e', '--epochs', type=int, default=10,
                        help='How many epochs to run in total?')
    parser.add_argument('-steps', '--epoch-steps', type=int, default=100,
                        help='The number of steps per epoch')
    parser.add_argument('-val', '--val-steps', type=int, default=8,
                        help='The number of val steps')
    parser.add_argument('-v', '--eval_interval', type=int, default=1,
                        help='Evaluation/Checkpoint interval in epochs.')

    args = parser.parse_args()

    # data setup
    train_data = input_pipeline(dirname=args.data_dir,
                                imagepath='**/train/images/*.png',
                                annotationpath='**/train/annotations/*.png',
                                num_channels=args.channels,
                                is_training=True,
                                use_augmentation= not args.no_augment,
                                batch_size=args.batch_size)
    
    eval_data = input_pipeline(dirname=args.data_dir,
                               imagepath='**/eval/images/*.png',
                               annotationpath='**/eval/annotations/*.png',
                               num_channels=args.channels,
                               is_training=False,
                               use_augmentation=False)

    # model setup
    if args.model in ('unet',):
        model = UNet(input_shape=(224, 224, args.channels),
                     filters=args.filters,
                     dropout=args.drop_prob).build_unet_model()
    else:
        raise NotImplementedError

    if args.weight_file is not None:
        model.load_weights(args.weight_file)

    if args.loss == 'dice_coeff':
        loss = dice_coeff_loss()
    elif args.loss == 'bce_dice':
        loss = bce_dice_loss()
    else:
        loss = args.loss

    model.compile(optimizer=tf.train.AdamOptimizer(args.learning_rate),
                  loss=loss,
                  metrics=['accuracy'])

    cptp_op = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(args.save_dir, 'cpts'), save_best_only=True,
        verbose=1, period=args.eval_interval)
    tboard_op = tf.keras.callbacks.TensorBoard(
        log_dir=os.path.join(args.save_dir, 'logs'), histogram_freq=1,
        write_graph=False, write_images=False, update_freq='batch',
        batch_size=args.batch_size)

    # Print here so its after the TensorFlow warnings
    print('Parsed Arguments:\n',
          json.dumps(vars(args), indent=4, separators=(',', ':')))

    model.summary()

    model.fit(
        train_data, validation_data=eval_data, epochs=args.epochs,
        steps_per_epoch=args.epoch_steps, validation_steps=args.val_steps,
        callbacks=[cptp_op, tboard_op])

    model.save(os.path.join(args.save_dir, '{}.h5'.format(args.model)))

    data_dir = pathlib.Path(args.data_dir)