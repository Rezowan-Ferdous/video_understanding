import torch
import torch.nn.functional as F
import numpy as np


def cosine_similarity(x1, x2):
    """Calculates the cosine similarity between two frames x1 and x2."""
    similarity = F.cosine_similarity(x1, x2, dim=-1)  # Calculates cosine similarity along the last dimension
    return similarity




def gaussian_kernel(window_size, sigma):
    """Creates a 1D Gaussian kernel."""
    kernel = torch.tensor([np.exp(-(x - window_size // 2) ** 2 / (2 * sigma ** 2)) for x in range(window_size)])
    kernel = kernel / kernel.sum()  # Normalize the kernel
    return kernel
