import os
import glob
import logging
import shutil
from sklearn.model_selection import train_test_split
import numpy as np
from keras.utils.np_utils import to_categorical
from hotdog import models
from hotdog.utils import transform
from hotdog.utils import utils

logger = logging.getLogger(__name__)
utils.configure_logger(logger)


def load_image_class(
    class_paths,
    class_label):
    '''
    Loads all the image paths in class_paths into memory, and assign
    'class_label' to that image, represented as a matrix.

    Returns
    - X: Images as list of numpy matrices. Each numpy matrix is (sz, sz, 1)
    - y: Labels (i.e. class_label as a list) (e.g. [class_label, class_label, ...])

    Input:
    - class_paths: List of image paths
    - class_label: Int representing the label for these images
    '''
    X = []
    y = []
    
    logger.info('Loading images...')
    for img_path in class_paths:
        # Read image into memory
        img = ndimage.imread(img_path, mode='L')

        # Append to images
        img_array = np.expand_dims(img, axis=2)  # Make the shape -> (sz, sz, 1)
        X.append(img_array)
        # Append to labels
        y.append(class_label)

    return X, y


def load_data(
    data_dir_path,
    img_size):
    '''
    Loads all hotdog/non-hotdog data from data_dir_path into memory. Will
    resize images to img_size.

    Returns images and labels in a matrix for training as a numpy matrix
    - X_train: Images (training set). numpy array (N x img_size x img_size)
    - X_test: Images (test set). numpy array (N x img_size x img_size)
    - y_train: Labels (training set). numpy array (N x 1)
    - y_test: Labels (test set). numpy array (N x 1)

    Arguments:
    - data_dir_path: A directory of folders. Each folder name will act as a label for a set of images.
    e.g.
    training_data/
        class0/
            class0_img0.jpg
            class0_img1.jpg
            ...
        class1/
            class1_img0.jpg
            class1_img1.jpg
            ...
        ...
    - img_size: Image dimensions (width, height)
    '''

    # Assign class labels to all images (according to their folder)

    # Get the image paths
    hotdogs_path_pattern = os.path.join(data_dir_path, 'hotdog/**/*.jpg')
    nonhotdogs_path_pattern = os.path.join(data_dir_path, 'not-hotdog/**/*.jpg')
    hotdogs = glob.glob(hotdogs_path_pattern, recursive=True)
    notHotdogs = glob.glob(nonhotdogs_path_pattern, recursive=True)

    # Assign class_label
    xHotdog, yHotdog = transform.load_image_class(
        hotdogs,
        0)
    xNotHotdog, yNotHotdog = transform.load_image_class(
        notHotdogs,
        1)
    logger.info("There are %d hotdog images", len(xHotdog))
    logger.info("There are %d not hotdog images", len(xNotHotdog))

    # Combine all (image, label) into on large matrix
    X_all = np.array(xHotdog + xNotHotdog)
    y_all = to_categorical(np.array(yHotdog + yNotHotdog))

    # Split into training/test set
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
        logger.info('%s: %f', metric_name, metric_value)


def run():
    data_path = 'C:/data/hotdog_training'

    img_size = 128

    # Load data
    X_train, X_test, y_train, y_test = load_data(
        data_path,
        (img_size, img_size))

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
