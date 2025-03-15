import pandas 
import numpy as np
from sklearn.metrics import mean_squared_error
# Function to compute the symmetric overlap (intersection over union) between two images
def compute_overlap(img1, img2):
    # Both images are assumed to have values between 0 and 1
    numerator = np.sum(np.minimum(img1, img2))
    denominator = np.sum(np.maximum(img1, img2))
    return numerator / denominator if denominator != 0 else 0

# Function to compute the Bhattacharyya coefficient between two images
def compute_bhattacharyya(img1, img2):
    # Optionally, normalize images so that their sums equal 1
    p = img1 / np.sum(img1) if np.sum(img1) != 0 else img1
    q = img2 / np.sum(img2) if np.sum(img2) != 0 else img2
    return np.sum(np.sqrt(p * q))

from scipy.spatial.distance import jensenshannon

# Function to compute the Jensen-Shannon Divergence (JSD) between two images
def compute_jsd(img1, img2):
    # Normalize images so that they sum to 1 (convert to probability distributions)
    p = img1 / np.sum(img1) if np.sum(img1) != 0 else img1
    q = img2 / np.sum(img2) if np.sum(img2) != 0 else img2
    # Compute Jensen-Shannon Divergence (JSD)
    return jensenshannon(p.flatten(), q.flatten()) ** 2  # Square to get original JSD

# Function to compute RMSE between two images
def compute_rmse(img1, img2):
    return np.sqrt(mean_squared_error(img1, img2))