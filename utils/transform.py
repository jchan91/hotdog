import logging
from multiprocessing.pool import ThreadPool
import numpy as np
import skimage.transform
import skimage.color
import skimage.filters
import skimage.exposure
from scipy import ndimage
import scipy.interpolate
from hotdog.utils import utils
from memory_profiler import profile


logger = logging.getLogger(__name__)
utils.configure_logger(logger)


@profile
def generate_augmented_image(img):
    # blur the image
    # result = skimage.filters.gaussian(img, sigma=25) # TODO Make this more mem efficient
    result = ndimage.gaussian_filter(
        img,
        sigma=1.5)

    # random rotation
    angle = np.random.randint(0, 360)
    # img = skimage.transform.rotate(img, angle)
    ndimage.rotate(
        result,
        angle,
        reshape=False,
        output=result)

    return result


def img_to_class_sample(img):
    return np.expand_dims(img, axis=2)  # Make the shape -> (sz, sz, 1)


# TODO Make thie more mem efficient
# Define the operations we'll do on an image class
def to_grayscale(img):
    return skimage.color.rgb2gray(img)  # Convert RGB -> gray


def histogram_equalize_image(
    img,
    output=None
):
    ret_val = None
    if output is None:
        output = np.zeros(img.shape)

    num_bins = 256
    histogram, bins = np.histogram(img.ravel(), num_bins, density=True)
    cdf = histogram.cumsum()
    normalize = 255 / cdf[-1]
    cdf *= normalize # remap

    # interpolate
    interp_f = scipy.interpolate.interp1d(
        bins[:-1],
        cdf,
        copy=False,
        bounds_error=False,
        fill_value=(0, cdf[-1])
    )
    for (row, col), val in np.ndenumerate(img):
        output[row, col] = interp_f(val)


    # interpolate
    # for (row, col), val in np.ndenumerate(img):
    #     interp_domain_center = val
    #     interp_domain_size = 3
    #     interp_domain_radius = int(interp_domain_size / 2)
    #     beg = int(max(0, interp_domain_center - interp_domain_radius))
    #     end = int(min(interp_domain_center + interp_domain_radius + 1, num_bins - 1)) # inclusive
    #     interp_domain = bins[beg:end]
    #     interp_range = cdf[beg:end]
    #     # No need for bounds checking of val, np.interp takes care of it
    #     output[row, col] = np.interp(val, interp_domain, interp_range)


    ret_val = output
    return ret_val


@profile
def load_image_class(class_paths, class_label, class_size, img_size):
    '''
    Loads all the image paths in class_paths into memory, and assign
    'class_label' to that image.

    Additionally, this function will continue to load/augment images
    until dataset has reached 'class_size'.

    Performs a resize to img_size (2D), and any necessary augmentations
    '''
    x = []
    y = []
    
    def resize(img):
        return skimage.transform.resize(img, img_size)

    def append_class_example(img):
        x.append(img_to_class_sample(img))
        y.append(class_label)

    # Initialize a thread pool to speed things up
    imgs = []
    logger.info('Loading original images...')
    for img_path in class_paths:
        imgs.append(ndimage.imread(img_path, mode='L'))
        
    assert imgs

    if len(imgs) < class_size:
        logger.info('Augmenting image set...')
        random_indicies = np.random.randint(
            0,
            high=len(imgs),
            size=class_size - len(imgs))
        for idx in random_indicies:
            imgs.append(generate_augmented_image(imgs[idx]))

    logger.info('Normalizing images of class...')
    for img in imgs:
        # img = to_grayscale(img)
        img = resize(img)
        # histogram_equalize_image(img, output=img)
        img = skimage.exposure.equalize_hist(img)
        
        append_class_example(img)

    return x, y


import os
import glob
from scipy import ndimage
from hotdog.utils.viewer import ImageViewer
import matplotlib.pyplot as plt


def generate_augmented_image_test():
    img_p = 'c:/data/hotdog_debug/hotdog/chili.jpg'
    img = ndimage.imread(img_p, mode='L')
    aug = generate_augmented_image(img)


def load_image_class_test():
    data_dir_path = 'C:/data/hotdog_training'
    paths_pattern = os.path.join(data_dir_path, 'hotdog/**/*.jpg')
    paths = glob.glob(paths_pattern, recursive=True)
    desired_class_size = 200
    img_size = (128, 128)
    load_image_class(
        paths,
        0,
        desired_class_size,
        img_size)


if __name__ == '__main__':
    load_image_class_test()

