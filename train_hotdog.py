import logging
from utils import data_set


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


if __name__ == '__main__':
    data_path = 'C:/data/hotdog_data'
    data_set.open_data_set(
        path=data_path,
        refresh_data=False)
    data_set.clean_data(data_path)
