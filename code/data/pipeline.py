#!/usr/bin/python3

import pathlib
import tensorflow as tf


def input_pipeline(dirname,
                   imagepath,
                   annotationpath,
                   is_training,
                   batch_size=32,
                   num_channels=3,
                   image_size=(224, 224),
                   standardize=True,
                   use_augmentation=True,
                   rotate=True,
                   flip=True,
                   crop=True,
                   padding=None,
                   num_parallel_calls=tf.data.experimental.AUTOTUNE,
                   use_caching=True,
                   resize=False):
    """Construct a data generator using tf.Dataset.

    Args:
        dirname: the directory containing the data
        imagepath: the path pattern for image files
        annotationpath: the path pattern for annotation files
        is_training: wether to use data augmentation and shuffling
        batch_size: the batch_size
        num_channels: the number of channels of the image
        image_size: (height, width) of images to output
        standardize: wether to standardize images
        use_augmentation: wether to use data augmentation
        rotate: wether to use rotation
        flip: wether to use flipping
        crop: wether to use cropping
        padding: the padding to use when cropping
        num_parallel_calls: the number of elements to process in parallel
        use_caching: wether to use caching after preprocessing or not. Can
        improve performance if RAM is big enough to store all the data

    Returns:
        tf.data.Dataset: Dataset with iterator that returns (image, annotation)
    """

    root_dir = pathlib.Path(dirname)

    # Create tf.data.Dataset containing the path strings
    # Important: We need to make sure that we only consider the images with
    # annotations available. We therefore loop over images/ txt files and look
    # if the associated file exist. Use sorting, just to be sure that data is
    # read in the same order.
    image_paths = tf.data.Dataset.from_tensor_slices(
        [str(path) for path in sorted(root_dir.glob(imagepath))
         if path.stem in [path2.stem for path2
                          in root_dir.glob(annotationpath)]]
    )
    annotation_paths = tf.data.Dataset.from_tensor_slices(
        [str(path) for path in sorted(root_dir.glob(annotationpath))
         if path.stem in [path2.stem for path2
                          in root_dir.glob(imagepath)]]
    )

    # Now create new datasets that load images and annotations one-the-fly
    images = image_paths.map(
        lambda x: load_image(x, resize=resize, num_channels=num_channels),
        num_parallel_calls=num_parallel_calls
    )
    if annotationpath.endswith('.txt'):
        annotations = annotation_paths.map(
            load_annotations,
            num_parallel_calls=num_parallel_calls
        )
    elif annotationpath.endswith('.png'):
        annotations = annotation_paths.map(
            lambda x: load_image(x, resize=resize, num_channels=1),
            num_parallel_calls=num_parallel_calls
        )
    else:
        raise NotImplementedError('Annotations need to be txt or png files')

    # Since the datasets are in the same order we can just zip them together
    # to get a dataset of (image, annotations) pairs.
    data = tf.data.Dataset.zip((images, annotations))

    data = data.map(
        lambda x, y: preprocessing(data=(x, y), standardize=standardize),
        num_parallel_calls=num_parallel_calls)

    if is_training:
        data = data.shuffle(100)

    if use_caching:
        data = data.cache()

    data = data.repeat()

    if use_augmentation:
        data = data.map(
            lambda x, y: augment(
                data=(x, y),
                size=image_size,
                num_channels=num_channels,
                rotate=rotate,
                flip=flip,
                crop=crop,
                padding=padding),
            num_parallel_calls=num_parallel_calls)
        data = data.batch(batch_size, drop_remainder=True)
    else:
        data = data.map(
            lambda x, y: extract_patches(
                data=(x, y),
                size=image_size,
                num_channels=num_channels),
            num_parallel_calls=num_parallel_calls
        )

    data = data.prefetch(tf.data.experimental.AUTOTUNE)

    return data


def serving_input_pipeline(imagepath,
                           num_channels=3,
                           image_size=(224, 224),
                           standardize=True,
                           num_parallel_calls=tf.data.experimental.AUTOTUNE):
    """Construct a data generator using tf.Dataset for prediction.

    Args:
        imagepath: the path pattern for image files
        num_channels: the number of channels of images
        image_size: (height, width) of images to output
        standardize: wether to standardize images
        num_parallel_calls: the number of elements to process in parallel

    Returns:
        tf.data.Dataset: Dataset with iterator that returns (image, annotation)
    """

    paths = tf.gfile.Glob(imagepath)
    image_paths = tf.data.Dataset.from_tensor_slices(paths)

    images = image_paths.map(
        lambda x: load_image(x, num_channels=num_channels),
        num_parallel_calls=num_parallel_calls
    )

    if standardize:
        images = images.map(
            lambda x: tf.image.per_image_standardization(x),
            num_parallel_calls=num_parallel_calls)

    images = images.map(
        lambda x: tf.reshape(
            tf.extract_image_patches(
                images=tf.expand_dims(x, 0),
                ksizes=(1,) + image_size + (1,),
                strides=(1,) + image_size + (1,),
                rates=[1, 1, 1, 1],
                padding='SAME'),
            shape=[-1, image_size[0], image_size[1], num_channels]),
        num_parallel_calls=num_parallel_calls)

    return images, paths

def load_image(path, resize=False, num_channels=3):
    """Load images.

    Args:
        path: the path of the image
        num_channels: the number of channels

    Returns:
        Tensor: with shape [height, width, num_channels]
    """
    image = tf.read_file(path)
    image = tf.image.decode_png(image, channels=num_channels)
    if resize:
        image = tf.image.resize_images(image, size=(1500, 2000))
    image.set_shape([None, None, num_channels])
    return tf.cast(image, tf.float32)


# TODO: Obsolet?
def load_annotations(path):
    """Load annotations.

    Args:
        path: the path of the txt file containing annotations

    Returns:
        Tensor:
    """
    annotations = tf.read_file(path)
    # Get rid of the last value, because it is an empty string
    annotations = tf.strings.split([annotations], '\n').values[:-1]
    # Extract columns by using blank space as delimiter
    x_position, y_position, _ = tf.io.decode_csv(
        records=annotations,
        record_defaults=[[], [], 0],
        field_delim=' ')
    # Transform to sparse Tensor, i.e. use x_position and y_position as
    # indices
    sparse_annotations = tf.SparseTensor(
        indices=tf.stack(
            [tf.cast(y_position, tf.int64), tf.cast(x_position, tf.int64)],
            axis=-1),
        values=tf.ones_like(y_position),
        dense_shape=[1600, 1200]
    )
    annotation_mask = tf.sparse.to_dense(
        sparse_annotations,
        validate_indices=False)

    return annotation_mask


def augment(data,
            size=(224, 224),
            num_channels=1,
            rotate=False,
            flip=False,
            crop=False,
            padding=None):
    """Applies rotating, flipping and cropping data augmentation.

    Args:
        data: the image and annotations
        size: (height, width) of image
        num_channels: the number of channels of images
        rotate: wether to use rotation
        flip: wether to use flipping
        crop: wether to use cropping
        padding: the padding to use when cropping

    Returns:
        (Tensor, Tensor): augmented image of shape (height, width,
        num_channels) and associated annotations of shape (height, width)
    """

    # Concat image and annotations so the same augmentation is applied
    image_annotations = tf.concat(data, axis=-1)

    if crop:
        if padding:
            image_annotations = tf.pad(
                image_annotations,
                ((padding, padding), (padding, padding), (0, 0)))
        image_annotations = tf.image.random_crop(
            image_annotations, size + (num_channels+1,))
    else:
        image_annotations = tf.extract_image_patches(
            images=tf.expand_dims(image_annotations, 0),
            ksizes=(1,) + size + (1,),
            strides=(1,) + size + (1,),
            rates=[1, 1, 1, 1],
            padding='SAME'
        )
        patches_shape = tf.shape(image_annotations)
        image_annotations = tf.reshape(
            image_annotations,
            (-1,) + size + (num_channels+1,)
        )
        image_annotations.set_shape((None, size[0], size[1], num_channels+1))

    if rotate:
        # Rotate 0, 90, 180 or 270 degrees randomly
        image_annotations = tf.image.rot90(
            image_annotations,
            tf.random_uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
        )

    if flip:
        image_annotations = tf.image.random_flip_left_right(image_annotations)

    image = image_annotations[..., :-1]
    annotations = image_annotations[..., -1:]
    return image, annotations


def preprocessing(data, standardize=False):
    """Apply preprocessing to image.

    Args:
        data: the image and annotations
        standardize: wether to standardize image

    Returns:
        (Tensor, Tensor): the preprocessed image, the annotations
    """
    # Unpack image and annotations form data
    image, annotations = data

    image /= 255.
    annotations /= 255.

    # Standardize to mean zero and unit variance
    if standardize:
        image = tf.image.per_image_standardization(image)

    return image, annotations


def extract_patches(data, size=(224, 224), num_channels=3):
    """Extract patches from image and annotations.

    Args:
        data: the image and annotations
        size: (height, width) of patches

    Returns:
        (Tensor, Tensor): image of shape (n_patches, height, width, n_channels)
        and annotations of shape (n_patches, height, width)
    """

    patches = tf.concat(data, axis=-1)
    patches = tf.extract_image_patches(
        images=tf.expand_dims(patches, 0),
        ksizes=(1,) + size + (1,),
        strides=(1,) + size + (1,),
        rates=[1, 1, 1, 1],
        padding='SAME'
    )

    patches = tf.reshape(patches, shape=[-1, size[0], size[1], num_channels+1])
    patches.set_shape((None, size[0], size[1], num_channels+1))
    image = patches[..., :-1]
    annotations = patches[..., -1:]
    return image, annotations
