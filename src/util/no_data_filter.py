import numpy as np
import scipy.ndimage


class NoDataFilter:
    def __init__(self, threshold: int):
        # for longer strides than 255, need to use a larger datatype below at convolutions
        assert threshold <= 255
        self.threshold = threshold
        self.kernel = np.ones(threshold, dtype=np.uint8)

    def __call__(self, img: np.ndarray) -> bool:
        """
        Args:
            img_np (np.array): Input image array in HWC format with dtype uint8
        """
        assert img.shape == (224, 224, 3)
        assert img.dtype == np.uint8

        # create a boolean mask (N,) where each pixel is pure white (or 254), seems
        # like some images have 254 instead of 255, probably JXL compression
        white_pixels = np.all(self._extract_image_edges(img) >= 254, axis=1)

        # use 1D convolution to find runs of 'threshold' white pixels
        convolved = scipy.ndimage.convolve1d(
            white_pixels.astype(dtype=np.uint8),
            self.kernel,
            # use wrap mode so that we catch the top-left corner wraparound case
            mode="wrap",
        )

        return np.any(convolved >= self.threshold)

    def _extract_image_edges(self, img: np.ndarray) -> np.ndarray:
        top = img[0, :, :]  # (W, 3) pixels left-to-right
        bot = img[-1, :, :]  # (W, 3) pixels left-to-right
        right = img[:, -1, :]  # (H, 3) top-to-bottom
        left = img[:, 0, :]  # (H, 3) top-to-bottom

        # reverse bot and left pixel order so that bot becomes right-to-left and left becomes bottom-to-top
        # so that we have a stride of 1px in clockwise order
        bot = bot[::-1, :]
        left = left[::-1, :]

        return np.concatenate([top, right, bot, left], axis=0)
