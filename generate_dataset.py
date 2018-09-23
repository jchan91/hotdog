import os
import glob
import logging
import shutil
import datetime
import argparse
import numpy as np
from hotdog.utils import transform
from hotdog.utils import utils


logger = logging.getLogger(__name__)
utils.configure_logger(logger)


def get_current_datetime_str():
    return datetime.datetime.now().strftime('%Y-%m-%d.%H.%M.%S')


dst_path_counter = 0
def get_dst_path(
    src_path,
    dst_dir_path,
    dst_suffix):
    global dst_path_counter
    src_name = utils.get_filename_without_ext(src_path)
    dst_name = '{0}_{1:02}{2}'.format(
        src_name,
        dst_path_counter,
        dst_suffix
    )
    dst_path_counter = dst_path_counter + 1

    return os.path.join(dst_dir_path, dst_name)


def transform_image_folder(
    src_data_dir_path,
    dst_data_dir_path,
    image_size,
    class_size,
    temp_dir_path):
    '''
    Returns nothing.

    image_size: Desired (width, height) of the output images

    Takes an image dataset and prepares it as a training dataset. Some operations include
    resizing, rgb -> gray, augmentation, etc.
    '''
    image_paths_pattern = os.path.join(src_data_dir_path, '*.jpg')
    image_paths = glob.glob(image_paths_pattern, recursive=True)
    if len(image_paths) is 0:
        return

    # Augment the source dataset
    # Choose random images from the dataset to act as source images
    # Apply images operations on those example images to generate augmentation
    # Then normalize both the original images and augmented images

    # Save the augmented images to a temporary directory so that we can find
    # them later to be normalized
    src_data_dir_name = utils.get_filename(src_data_dir_path)
    temp_dir_name = src_data_dir_name + '-temp.' + get_current_datetime_str()
    temp_dst_images_dir_path = os.path.join(temp_dir_path, temp_dir_name)
    utils.ensure_dir_exists(temp_dst_images_dir_path)

    logger.info('Augmenting dataset to %s...', temp_dst_images_dir_path)
    random_indices = np.random.randint(
        0,
        high=len(image_paths),
        size=class_size - len(image_paths))
    logger.info('Num rand indices %d', len(random_indices))
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
    utils.ensure_dir_exists(dst_data_dir_path)
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


def generate_dataset(
    src_data_dir_path,
    dst_data_dir_path,
    image_size,
    class_size,
    temp_dir
):
    '''
    Takes a directory that has training data, and recursively runs
    transform_image_folder on each folder. Outputs the transformed images
    into a folder structure that mirrors the input structure

    Returns nothing.

    Input:
    - src_data_dir_path: Source images
    e.g.
    training_data/
        class0/
            class0_img0.jpg
            class0_img1.jpg
            ...
        class1/
            class1_img0.jpg
            class1_img1.jpg
    - dst_data_dir_path: Destination for images to be written to
    - temp_dir: Temporary working directory for this function
    '''
    for subdir, _, _ in os.walk(src_data_dir_path):
        # Transform this directory
        
        # src = current directory
        # dst = <dst_data_dir_path>/<subdir_rel_path>
        dst_dir_relative_path = os.path.relpath(subdir, src_data_dir_path)
        dst_dir_path = os.path.join(dst_data_dir_path, dst_dir_relative_path)
        logger.info('Transforming %s to %s',
            subdir,
            dst_dir_path)
        transform_image_folder(
            src_data_dir_path=subdir,
            dst_data_dir_path=dst_dir_path,
            image_size=image_size,
            class_size=class_size,
            temp_dir_path=temp_dir
        )
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate dataset.')
    parser.add_argument(
        '-i', '--input', dest='input_dir', required=True,
        help='Source dataset'
    )
    parser.add_argument(
        '-o', '--output', dest='output_dir', required=True,
        help='Output directory'
    )
    parser.add_argument(
        '-c', '--class-size', dest='class_size', type=int, required=True,
        help='Number of images per class to augment up to'
    )
    parser.add_argument(
        '-t', '--temp-dir', dest='temp_dir', default='C:\\data\\temp\\',
        help='Temporary working directory'
    )
    args = parser.parse_args()
    
    # Main logic
    generate_dataset(
        src_data_dir_path=args.input_dir,
        dst_data_dir_path=args.output_dir,
        image_size=(128, 128),
        class_size=args.class_size,
        temp_dir=args.temp_dir
    )