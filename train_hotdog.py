import logging
from sklearn.model_selection import train_test_split
import numpy as np
import glob
from PIL import Image
import os


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def load_image(img_path, img_size):
    # img = cv2.imread(img_path)
    # angle = np.random.randint(0, 360)
    # img = rotateImage(img, angle)
    # img = cv2.blur(img,(5,5))
    # img = cv2.resize(img, img_size)
    # return img

    img = Image.open(img_path)
    img = img.resize(img_size, resample=Image.BILINEAR)
    img = img.convert('L')  # Convert RGB -> gray
    return np.array(img)


def load_img_class(class_paths, load_images, class_size, img_size):
    '''
    Loads all the image paths in class_paths into memory, and assign
    'class_label' to that image.

    Additionally, this function will continue to load/augment images
    until dataset has reached 'class_size'.

    Performs a resize to img_size (2D), and any necessary augmentations
    '''
    x = []
    y = []
    
    for img_path in class_paths:
        img = load_image(img_path, img_size)        
        x.append(img)
        y.append(load_images)
        
    while len(x) < class_size:
        randIdx = np.random.randint(0, len(class_paths))
        img = load_image(class_paths[randIdx], img_size)
        x.append(img)
        y.append(load_images)
        
    return x, y


def load_data(
    data_dir_path,
    img_size,
    class_size
):
    '''
    Loads all hotdog/non-hotdog data from data_dir_path into memory. Will
    resize images to img_size (1D).

    Performs any image modifications necessary for training (e.g. blurs, rotations)

    Returns images and labels:
    - X: Images. numpy array (N x img_size x img_size)
    - y: Labels. numpy array (N x 1)
    '''
    hotdogs_path_pattern = os.path.join(data_dir_path, 'hotdog/**/*.jpg')
    nonhotdogs_path_pattern = os.path.join(data_dir_path, 'not-hotdog/**/*.jpg')
    hotdogs = glob.glob(hotdogs_path_pattern, recursive=True)
    notHotdogs = glob.glob(nonhotdogs_path_pattern, recursive=True)
    
    img_size_2d = (img_size, img_size)
    xHotdog, yHotdog = load_img_class(hotdogs, 0, class_size, img_size_2d)
    xNotHotdog, yNotHotdog = load_img_class(notHotdogs, 1, class_size, img_size_2d)
    print("There are", len(xHotdog), "hotdog images")
    print("There are", len(xNotHotdog), "not hotdog images")
    
    X = np.array(xHotdog + xNotHotdog)
    y = np.array(yHotdog + yNotHotdog)
    
    return X, y


if __name__ == '__main__':
    data_path = 'C:/data/'

    img_size = 128
    X_all, y_all = load_data(
        data_path,
        img_size,
        class_size=-1)  # TODO: Implement image augmentation and then use class_size > 0

    