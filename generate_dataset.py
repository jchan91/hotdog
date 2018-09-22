import os
import glob
import logging
import shutil
import datetime
import numpy as np
from hotdog.utils import transform
from hotdog.utils import utils


logger = logging.getLogger(__name__)
utils.configure_logger(logger)


def get_current_datetime_str():
    return datetime.datetime.now().strftime('%Y-%m-%d.%H.%M.%S')


def get_dst_path(
    src_path,
    dst_dir_path,
    dst_suffix):
    src_name = utils.get_filename_without_ext(src_path)
    dst_name = src_name + dst_suffix
    return os.path.join(dst_dir_path, dst_name)


def generate_dataset(
    src_data_dir_path,
    dst_data_dir_path,
    image_size,
    class_size):
    '''
    Returns nothing.

    image_size: Desired (width, height) of the output images

    Takes an image dataset and prepares it as a training dataset. Some operations include
    resizing, rgb -> gray, augmentation, etc.
    '''
    image_paths_pattern = os.path.join(src_data_dir_path, '*.jpg')
    image_paths = glob.glob(image_paths_pattern, recursive=True)

    # Augment the source dataset
    # Choose random images from the dataset to act as source images
    # Apply images operations on those example images to generate augmentation
    # Then normalize both the original images and augmented images

    # Save the augmented images to a temporary directory so that we can find
    # them later to be normalized
    temp_dir_name = 'temp.' + get_current_datetime_str()
    temp_dst_images_dir_path = os.path.join(dst_data_dir_path, temp_dir_name)
    utils.ensure_dir_exists(temp_dst_images_dir_path)

    logger.info('Augmenting dataset to %s...', temp_dst_images_dir_path)
    random_indices = np.random.randint(
        0,
        high=len(image_paths),
        size=class_size - len(image_paths))
    for idx in random_indices:
        src_image_path = image_paths[idx]
        dst_image_path = get_dst_path(
            src_image_path,
            temp_dst_images_dir_path,
            '-augmented.jpg')
        transform.generate_augmented_image(
            src_image_path,
            dst_image_path)
    
    # Normalize images in preparation to be fed into classifier
    # Will run on all images, and augmented images
    logger.info('Normalize augmented images...')
    augmented_image_paths_pattern = os.path.join(temp_dst_images_dir_path, '*.jpg')
    augmented_image_paths = glob.glob(augmented_image_paths_pattern)
    for augmented_image_path in augmented_image_paths:
        dst_image_path = get_dst_path(
            augmented_image_path,
            dst_data_dir_path,
            '-normalized.jpg')
        transform.normalize_image(
            augmented_image_path,
            dst_image_path,
            image_size)

    # Normalize the originals too
    logger.info('Normalize original images...')
    for original_image_path in image_paths:
        dst_image_path = get_dst_path(
            original_image_path,
            dst_data_dir_path,
            '-normalized.jpg')
        transform.normalize_image(
            original_image_path,
            dst_image_path,
            image_size)

    # Done
    # Rename temp dst dir to real dst dir
    logger.info('Done.')


if __name__ == '__main__':
    # TODO: Use non-debug values
    generate_dataset(
        src_data_dir_path='C:\\data\\hotdog_debug\\hotdog',
        dst_data_dir_path='C:\\data\\temp\\hotdog_debug',
        image_size=(128, 128),
        class_size=100)