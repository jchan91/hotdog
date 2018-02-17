'''
Taken from comma.ai's self driving DNN
'''
import logging
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Flatten, Lambda, ELU
from keras.layers.convolutional import Conv2D


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def comma_ai_model(input_shape=(128, 128, 1)):
    model = Sequential()
    model.add(Conv2D(16, (8, 8), strides=(4, 4), padding='valid', input_shape=input_shape))
    model.add(ELU())
    model.add(Conv2D(32, (5, 5), strides=(2, 2), padding="same"))
    model.add(ELU())
    model.add(Conv2D(64, (5, 5), strides=(2, 2), padding="same"))
    model.add(Flatten())
    model.add(Dropout(.2))
    model.add(ELU())
    model.add(Dense(512))
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(2))
    model.add(Activation('softmax'))
    return model


def comma_small(input_shape=(128, 128, 1)):
    model = Sequential()
    # model.add(Conv2D(16, (8, 8), strides=(4, 4), padding='valid', input_shape=input_shape))
    # model.add(ELU())
    # model.add(Conv2D(32, (5, 5), strides=(2, 2), padding="same"))
    # model.add(ELU())
    # model.add(Conv2D(64, (5, 5), strides=(2, 2), padding="same"))
    # model.add(Flatten())
    # model.add(Dropout(.2))
    # model.add(ELU())
    model.add(Dense(128, input_dim=input_shape))
    # model.add(Dense(512))
    # model.add(Dropout(.5))
    # model.add(ELU())
    model.add(Dense(2))
    model.add(Activation('softmax'))
    return model


def test_model(input_shape=(128, 128, 1)):
    model = Sequential()
    model.add(Conv2D(16, (8, 8), strides=(4, 4), padding='valid', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Dense(2))
    model.add(Activation('softmax'))
    return model
