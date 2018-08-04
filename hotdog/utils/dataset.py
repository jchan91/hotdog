import logging
import os
import urllib.request
from urllib.parse import urlparse
from PIL import Image
import numpy as np
from multiprocessing.pool import ThreadPool


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


def download_image_net_p(
    url,
    dst_dir_path
):
    if isinstance(url, list):
        target_urls = url
    else:    
        urls_req = urllib.request.urlopen(url)
        target_urls = urls_req.read().split()

    pool = ThreadPool(32)
    pool.map(lambda u: download_url(u, dst_dir_path), target_urls)

    print('Attempted %d downloads' % len(target_urls))


def download_image_net(
    url,
    dst_dir_path
):
    if isinstance(url, list):
        target_urls = url
    else:   
        urls_req = urllib.request.urlopen(url)
        target_urls = urls_req.read().split()

    download_count = 0
    downloads_remaining = len(target_urls)
    for target_url in target_urls:
        download_url(
            target_url,
            dst_dir_path)
        download_count += 1
        downloads_remaining -= 1
        print('Downloads remaining: %d' % downloads_remaining)
    print('Downloaded %d' % download_count)


def download_url(
    target_url,
    dst_dir_path
):
    url = urlparse(target_url.decode('UTF-8'))
    dst_name = os.path.basename(url.path)
    dst_path = os.path.join(dst_dir_path, dst_name)

    try:
        urllib.request.urlretrieve(url.geturl(), dst_path)
    except urllib.error.HTTPError as e:
        print(str(e))
        # pass
    except urllib.error.URLError as e:
        print(str(e))
        # pass
    except Exception as e:
        print(str(e))
        # pass


def remove_badly_formatted_images(
    dirPath,
    execute_remove=False
):
    """
    Looks through images in dirPath that can't be opened
    """
    for root, dirs, filenames in os.walk(dirPath):
        for file in filenames:
            try:
                full_path = os.path.join(root, file)
                Image.open(full_path)
            except Exception:
                # logger.info('Removing %s', full_path)
                if execute_remove:
                    os.remove(full_path)
                else:
                    print(full_path)


def find_invalid_images(
    dirPath,
    invalidDir,
    execute_remove=False
):
    '''
    Looks through images in all dirPaths that have 'invalid' images, as specfied in the 'invalidDir'
    '''
    for img in os.listdir(dirPath):
        for invalid in os.listdir(invalidDir):
            try:
                current_image_path = os.path.join(dirPath, img)
                invalid_path = os.path.join(invalidDir, invalid)
                invalid = Image.open(invalid_path)
                question = Image.open(current_image_path)
                if invalid.size == question.size and not(np.bitwise_xor(invalid,question).any()):
                    if execute_remove:
                        os.remove(current_image_path)
                    else:
                        print(current_image_path)
                    break

            except Exception as e:
                print(str(e) + ' on %s' % current_image_path)


def ensure_all_file_ext(
    dir_path,
    execute_remove=False):
    '''
    Ensures all files in directory have an image extension
    '''
    for root, dirs, filenames in os.walk(dir_path):
        for file in filenames:
            head, tail = os.path.splitext(file)
            orig_path = os.path.join(root, file)

            if tail == '':
                dst_path = orig_path + '.jpg'  # TODO: Don't always assume jpg
                os.rename(orig_path, dst_path)
            elif tail.lower() != '.jpg' and tail.lower() != '.png' and tail.lower() != '.bmp':
                if execute_remove:
                    os.remove(orig_path)
                else:
                    print(orig_path)


def clean_data(
    dirPath,
    invalidDir,
    execute_remove=False
):
    """
    Removes bad images in dirPath
    """
    remove_badly_formatted_images(
        dirPath,
        execute_remove)

    find_invalid_images(
        dirPath,
        invalidDir,
        execute_remove)

    ensure_all_file_ext(
        dirPath,
        execute_remove)
