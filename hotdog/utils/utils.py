import logging
import os


def configure_logger(module_logger):
    # Alternative format
    # "%(asctime)s [%(filename)s] [%(funcName)s] [%(levelname)s] [%(lineno)d] %(message)s"
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    # Setup handlers
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    # Set handlers to logger
    module_logger.handlers = [] # maybe someday make this optional
    module_logger.addHandler(console_handler)
    module_logger.setLevel(logging.DEBUG)

    # Alternative...
    # logging.config.dictConfig({
    #     'version': 1,
    #     'disable_existing_loggers': True,
    #     'formatters': {
    #         'standard': {
    #             'format': '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
    #         }
    #     },
    #     'handlers': {
    #         'console': {
    #             'level': 'INFO',
    #             'class': 'logging.StreamHandler',
    #         },
    #     },
    #     'loggers': {
    #     }
    # })


logger = logging.getLogger(__name__)
configure_logger(logger)


def print_model(model):
    for layer in model.layers:
        logger.info(layer.name)


def replace_ext(path, ext):
    return os.path.splitext(path)[0] + ext


def get_filename(path):
    _, tail = os.path.split(path)
    return tail


def get_filename_without_ext(path):
    _, tail = os.path.split(path)
    return replace_ext(tail, '')
