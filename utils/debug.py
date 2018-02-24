import logging
from hotdog.utils import utils


logger = logging.getLogger(__name__)
utils.configure_logger(logger)


def addr_of_np(nparr):
    return nparr.__array_interface__['data'][0]


def sizeof_np(nparr):
    return nparr.size * nparr.itemsize

