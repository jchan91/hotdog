import logging
import matplotlib.pyplot as plt


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

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

        ImageViewer.__show_np(img_np)

    @staticmethod
    def __show_np(img_np):
        ''' Uses matplotlib to show image. Assumes shape (rows, cols) '''
        assert(len(img_np.shape) == 2)
        plt.imshow(img_np)
        plt.show()

