import logging


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def print_model(model):
    for layer in model.layers:
        print(layer.name)
