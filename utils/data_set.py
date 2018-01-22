import os
import urllib.request
from urllib.parse import urlparse


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
    for hotdog_url in hotdog_urls_req.read().split():
        url = urlparse(hotdog_url.decode('UTF-8'))
        dst_path = os.path.basename(url.path)

        print(url)
        try:
            urllib.request.urlretrieve(url.geturl(), dst_path)
        except urllib.error.HTTPError as e:
            pass

        print(dst_path)
