import logging
from multiprocessing.pool import ThreadPool
import numpy as np
from PIL import Image
from PIL import ImageFilter


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


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


def convert_pil_image_to_nparray(
        img,
        img_size
):
    img = img.resize(img_size, resample=Image.BILINEAR)
    img = img.convert('L')  # Convert RGB -> gray
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

    # Initialize a thread pool to speed things up
    pool = ThreadPool(16)
    def load_original_image(img_path):
        img_pil = Image.open(img_path)
        img_np = convert_pil_image_to_nparray(img_pil, img_size)
        x.append(img_np)
        y.append(class_label)

    def add_augmented_image(class_path_idx):
        img_pil = Image.open(class_paths[class_path_idx])
        img_pil = generate_augmented_image(img_pil)
        img_np = convert_pil_image_to_nparray(img_pil, img_size)
        x.append(img_np)
        y.append(class_label)

    print('Loading original images...')
    pool.map(lambda p: load_original_image(p), class_paths)

    if len(x) < class_size:
        print('Augmenting image set...')
        random_indices = np.random.randint(
            0,
            high=len(x),
            size=class_size - len(x))
        pool.map(lambda i: add_augmented_image(i), random_indices)
        

    return x, y
