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


logger = logging.getLogger(__name__)
utils.configure_logger(logger)


def generate_augmented_image(img):
    # blur the image
    result = skimage.filters.gaussian(img, sigma=1.5)
    
    # random rotation
    angle = np.random.randint(0, 360)
    result = skimage.transform.rotate(result, angle)
    
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
    image = image.resize(image_size, resample=Image.BILINEAR)
    image = ImageOps.equalize(image)
    image.save(dst_image_path, "JPEG")



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
        img = to_grayscale(img)
        img = resize(img)
        img = skimage.exposure.equalize_hist(img)
        
        append_class_example(img)

    return x, y

