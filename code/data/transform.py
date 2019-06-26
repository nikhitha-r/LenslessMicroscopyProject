#!/usr/bin/python3

import os
import argparse
import random
import shutil

import pathlib
import numpy as np

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir',
                        help='the directory containing the annotations',
                        required=True)
    parser.add_argument('--imagepath',
                        help='the path pattern for image files',
                        default='**/images/**/*.png')
    parser.add_argument('--annotationpath',
                        help='the path pattern for annotation files',
                        default='**/annotations/**/*.png')
    parser.add_argument('--seed',
                        help='the seed for the train/eval/test split',
                        type=int,
                        default=1234)
    parser.add_argument('--sigma',
                        help='the standard deviation',
                        type=float,
                        default=1.0)
    args = parser.parse_args()

    # set seed
    random.seed(args.seed)

    # Get all annotation files with images
    root_dir = pathlib.Path(args.dir)
    annotation_paths = [
        path for path in sorted(root_dir.glob(args.annotationpath))
        if path.stem in [path2.stem for path2
                         in root_dir.glob(args.imagepath)]
    ]
    image_paths = [
        str(path) for path in sorted(root_dir.glob(args.imagepath))
        if path.stem in [path2.stem for path2
                         in root_dir.glob(args.annotationpath)]
    ]

    paths = list(zip(annotation_paths, image_paths))

    # train-eval-test split
    # TODO: change split from fixed to variable
    train = random.sample(paths, int(0.7 * len(paths)))
    val = random.sample(list(set(paths) - set(train)), int(0.15 * len(paths)))
    test = list(set(paths) - set(train) - set(val))

    data = {'train': train, 'eval': val, 'test': test}

    for folder in ['train', 'eval', 'test']:
        path = pathlib.PurePath(args.dir).joinpath(folder)
        # remove folder with content if it exists
        if pathlib.Path(path).exists():
            shutil.rmtree(path)
        # set up directory
        img_path = pathlib.Path(path.joinpath('images'))
        img_path.mkdir(parents=True)
        ann_path = pathlib.Path(path.joinpath('annotations'))
        ann_path.mkdir()

        for i in range(len(data.get(folder))):
            ann, img = data.get(folder)[i]
            # copy images to set folder
            src_img = pathlib.Path(img)
            dst_img = img_path.joinpath(src_img.name)
            shutil.copy(src_img, dst_img)

            # copy annotation to set folder
            src_ann = pathlib.Path(ann).with_suffix('.png')
            dst_ann = ann_path.joinpath(src_ann.name)
            shutil.copy(src_ann, dst_ann)
