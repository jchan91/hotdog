import logging
from multiprocessing.pool import ThreadPool
import numpy as np
from PIL import Image
from PIL import ImageFilter
from PIL import ImageOps
from hotdog.utils import utils


logger = logging.getLogger(__name__)
utils.configure_logger(logger)


def generate_augmented_image(img):
    # img = cv2.imread(img_path)
    # angle = np.random.randint(0, 360)
    # img = rotateImage(img, angle)
    # img = cv2.blur(img,(5,5))
    # img = cv2.resize(img, img_size)
    # return img

    # blur the image
    if img.mode != 'RGB' and img.mode != 'L':
        img = img.convert('L')
    img = img.filter(ImageFilter.GaussianBlur(3))

    # random rotation
    angle = np.random.randint(0, 360)
    img = img.rotate(angle)
    
    return img


def convert_pil_image_to_nparray(img):
    return np.expand_dims(np.array(img), axis=2)  # Make the shape -> (sz, sz, 1)


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

    # Define the operations we'll do on an image class
    def to_grayscale(img_pil):
        return img_pil.convert('L')  # Convert RGB -> gray
        
    def histogram_equalize_image(img_pil):
        return ImageOps.equalize(img_pil)

    def resize(img_pil):
        return img_pil.resize(img_size, resample=Image.BILINEAR)

    def append_class_example(img_pil):
        x.append(convert_pil_image_to_nparray(img_pil))
        y.append(class_label)

    # Initialize a thread pool to speed things up
    # pool = ThreadPool(16)
    imgs_pil = []
    logger.info('Loading original images...')
    for img_path in class_paths:
        imgs_pil.append(Image.open(img_path))

    if len(imgs_pil) < class_size:
        logger.info('Augmenting image set...')
        random_indicies = np.random.randint(
            0,
            high=len(imgs_pil),
            size=class_size - len(imgs_pil))
        for idx in random_indicies:
            imgs_pil.append(generate_augmented_image(imgs_pil[idx]))

    logger.info('Normalizing images of class...')
    for img_pil in imgs_pil:
        img_pil = to_grayscale(img_pil)
        img_pil = resize(img_pil)
        img_pil = histogram_equalize_image(img_pil)
        append_class_example(img_pil)

    # pool.map(lambda p: load_original_image(p), class_paths)
    return x, y
