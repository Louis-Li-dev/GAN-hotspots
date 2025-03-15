import numpy as np
from scipy.ndimage import gaussian_filter
def apply_gaussian(images, sigma):
    # images: numpy array with shape (n_samples, height, width)
    return np.array([gaussian_filter(img, sigma=sigma) for img in images])
