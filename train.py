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

from memory_profiler import profile


logger = logging.getLogger(__name__)
utils.configure_logger(logger)





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
    xHotdog, yHotdog = transform.load_image_class(hotdogs, 0, class_size, img_size_2d)
    xNotHotdog, yNotHotdog = transform.load_image_class(notHotdogs, 1, class_size, img_size_2d)
    logger.info("There are %d hotdog images", len(xHotdog))
    logger.info("There are %d not hotdog images", len(xNotHotdog))

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
        logger.info('%s: %f', metric_name, metric_value)


def run():
    data_path = 'C:/data/hotdog_training'

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
