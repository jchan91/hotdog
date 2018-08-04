import logging
import matplotlib
import matplotlib.pyplot as plt
from hotdog.utils import utils


logger = logging.getLogger(__name__)
utils.configure_logger(logger)


class ImageViewer(object):
    @staticmethod
    def show_np(img_np):
        img_shape = img_np.shape
        if len(img_shape) == 3:
            if img_shape[2] == 1:
                # Single channel
                img_np = img_np[:, :, 0]
            else:
                raise NotImplementedError('Multi-channel not yet supported')

        ImageViewer.__show_L_np(img_np)


    @staticmethod
    def __show_L_np(img_np):
        ''' Uses matplotlib to show image. Assumes shape (rows, cols) '''
        assert(len(img_np.shape) == 2)
        fig = plt.figure()
        fig.clf()
        ax = fig.add_subplot(1, 1, 1)
        ax.cla()
        ax.imshow(img_np, cmap='gray')
        fig.show()

