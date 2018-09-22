import logging
import numpy as np
import skimage.transform
import skimage.color
import skimage.filters
import skimage.exposure
from scipy import ndimage
import scipy.interpolate
from hotdog.utils import utils
from PIL import Image
from PIL import ImageOps
from PIL import ImageFilter


logger = logging.getLogger(__name__)
utils.configure_logger(logger)


def generate_augmented_image(
    src_image_path,
    dst_image_path):
    '''
    Returns nothing.
    
    Generates an image intended to augment an image dataset.
    '''
    # Open image
    logger.info('Loading image %s as augmentation example...', src_image_path)
    image = Image.open(src_image_path)

    # Blur image
    blur_sigma = 3
    image = image.filter(ImageFilter.GaussianBlur(blur_sigma))
    logger.info('Blurred image with simga: %f', blur_sigma)

    # Random rotation
    angle = np.random.randint(0, 360)
    image = image.rotate(angle)
    logger.info('Rotated image %d degrees', angle)

    # Save image
    logger.info('Saving to %s', dst_image_path)
    image.save(dst_image_path)


def normalize_image(
    src_image_path,
    dst_image_path,
    image_size):
    '''
    Returns nothing. Applies several image operations to any dataset image
    (augment or not) to ensure they are ready to be fed into the classifier
    '''
    # image = ndimage.imread(src_image_path, mode='L')
    # image = skimage.transform.resize(image, image_size)
    # image = skimage.exposure.equalize_hist(image)

    logger.info('Loading %s...', src_image_path)
    image = Image.open(src_image_path)
    if image.mode == 'RGB':
        logger.info('Convert to grayscale...')
        image = image.convert('L')
    logger.info('Resizing image...')
    print(image_size)
    image = image.resize(image_size, resample=Image.BILINEAR)
    logger.info('Equalizing histogram...')
    image = ImageOps.equalize(image)
    logger.info('Saving to %s', dst_image_path)
    image.save(dst_image_path, "JPEG")
