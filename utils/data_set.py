import logging
import os
import urllib.request
from urllib.parse import urlparse
from PIL import Image
import numpy as np


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def open_data_set(
    path,
    refresh_data=True
):
    '''
    Tries to open the data_set from path. If directory is empty, will download.
    '''
    if not os.path.exists(path):
        os.mkdir(path)
    elif not refresh_data:
        return

    hotdog_urls_req = urllib.request.urlopen('http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n07865105')

    hotdog_urls = hotdog_urls_req.read().split()
    download_count = 0
    downloads_remaining = len(hotdog_urls)
    for hotdog_url in hotdog_urls:
        url = urlparse(hotdog_url.decode('UTF-8'))
        dst_name = os.path.basename(url.path)
        dst_path = os.path.join(path, dst_name)

        try:
            urllib.request.urlretrieve(url.geturl(), dst_path)
            download_count += 1
        except urllib.error.HTTPError:
            pass
        except urllib.error.URLError:
            pass
        except Exception:
            pass

        downloads_remaining -= 1
        print('Downloads remaining: %d' % downloads_remaining)
    print('Downloaded %d' % download_count)


def clean_data(
    dirPath,
    invalidDir
):
    """
    Removes bad images in dirPath
    """
    files_to_remove = []
    
    find_badly_formatted_images(
        files_to_remove,
        dirPath)

    find_invalid_images(
        files_to_remove,
        dirPath,
        invalidDir)

    for file in files_to_remove:
        print(file)
        # os.remove(file)

    logger.info('Removed %d', len(files_to_remove))


def find_badly_formatted_images(
    files_to_remove,
    dirPath
):
    """
    Looks through images in dirPath that can't be opened

    Returns paths of bad images in files_to_remove
    """
    for root, dirs, filenames in os.walk(dirPath):
        for file in filenames:
            try:
                full_path = os.path.join(root, file)
                Image.open(full_path)
            except Exception:
                # logger.info('Removing %s', full_path)
                files_to_remove.append(full_path)


def find_invalid_images(
    files_to_remove,
    dirPath,
    invalidDir
):
    '''
    Looks through images in all dirPaths that have 'invalid' images, as specfied in the 'invalidDir'

    Returns paths of invalid images in files_to_remove
    '''
    for img in os.listdir(dirPath):
        for invalid in os.listdir(invalidDir):
            try:
                current_image_path = os.path.join(dirPath, img)
                invalid_path = os.path.join(invalidDir, invalid)
                invalid = Image.open(invalid_path)
                question = Image.open(current_image_path)
                if invalid.size == question.size and not(np.bitwise_xor(invalid,question).any()):
                    files_to_remove.append(current_image_path)
                    break

            except Exception as e:
                print(str(e))
