import os
import glob
import logging
from sklearn.model_selection import train_test_split
import numpy as np
from PIL import Image
from PIL import ImageFilter
from keras.utils.np_utils import to_categorical
from multiprocessing.pool import ThreadPool
from  hotdog import models


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
    xHotdog, yHotdog = load_image_class(hotdogs, 0, class_size, img_size_2d)
    xNotHotdog, yNotHotdog = load_image_class(notHotdogs, 1, class_size, img_size_2d)
    print("There are", len(xHotdog), "hotdog images")
    print("There are", len(xNotHotdog), "not hotdog images")

    X_all = np.array(xHotdog + xNotHotdog)
    y_all = to_categorical(np.array(yHotdog + yNotHotdog))

    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        X_all,
        y_all,
        test_size=0.2,
        random_state=rand_state)

    return X_train, X_test, y_train, y_test


def train(
        X_train,
        y_train,
        img_shape=(128, 128, 1),
        model=None):

    if model is None:
        model = models.test_model(img_shape)
    model.compile('adam', 'categorical_crossentropy', ['accuracy'])
    history = model.fit(X_train, y_train, nb_epoch=10, validation_split=0.1)
    return model, history


def evaluate(
        model,
        X_test,
        y_test):
    metrics = model.evaluate(X_test, y_test)
    for metric_i in range(len(model.metrics_names)):
        metric_name = model.metrics_names[metric_i]
        metric_value = metrics[metric_i]
        print('{}: {}'.format(metric_name, metric_value))


def run():
    data_path = 'C:/data/'

    img_size = 128

    # Load data
    X_train, X_test, y_train, y_test = load_data(
        data_path,
        img_size,
        class_size=-1)  # TODO: Implement image augmentation and then use class_size > 0

    model, history = train(
        X_train,
        y_train,
        (img_size, img_size, 1))

    evaluate(
        model,
        X_test,
        y_test)

    return model, history


if __name__ == '__main__':
    run()
