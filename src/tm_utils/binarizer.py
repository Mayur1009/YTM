import numpy as np
from line_profiler import profile


class ThermometerBinarizer:
    def __init__(self, ch: int = 8):
        self.ch = ch

    @profile
    def binarize_gray(self, X):
        assert X.ndim == 3, "Input should be a 3D array (N, H, W)"
        thresholds = (np.arange(1, self.ch + 1) * 255 / (self.ch + 1)).reshape(1, 1, 1, self.ch)
        out = (X[:, :, :, None] >= thresholds).astype(np.uint32)
        return out

    @profile
    def binarize_rgb(self, X):
        assert X.ndim == 4, "Input should be a 4D array (N, H, W, C)"
        thresholds = (np.arange(1, self.ch + 1) * 255 / (self.ch + 1)).reshape(1, 1, 1, 1, self.ch)
        out = (X[:, :, :, :, None] >= thresholds).astype(np.uint32)
        return out.reshape((*X.shape[:-1], X.shape[-1] * self.ch))


        # out = np.zeros((*X.shape[:-1], X.shape[-1] * self.ch))
        # for color in range(X.shape[-1]):
        #     for j in range(self.ch):
        #         t1 = (j + 1) * 255 / (self.ch + 1)
        #         out[:, :, color * self.ch + j] = (X[:, :, color] >= t1) & 1
        #
        # return out.astype(np.uint32)
