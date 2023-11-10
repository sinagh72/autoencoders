import torch

import torch
import torch.nn.functional as F
from math import exp


def gaussian_window(size, sigma):
    # Generates a 1-D tensor representing a Gaussian window.
    kernel = torch.arange(size, dtype=torch.float32) - size // 2
    kernel = torch.exp(-0.5 / sigma ** 2 * kernel ** 2)
    kernel = kernel / kernel.sum()
    return kernel


def create_gaussian_filter(window_size, sigma, channels):
    # Generates a 2D Gaussian Kernel with the specified window size and sigma.
    _1d_window = gaussian_window(window_size, sigma).unsqueeze(1)
    _2d_window = _1d_window.mm(_1d_window.t())
    window = _2d_window.expand(channels, 1, window_size, window_size).contiguous()
    return window


def ssim(img1, img2, window_size=11, sigma=1.5, data_range=1.0):
    # Compute the Structural Similarity Index (SSIM) between two images.
    channels = img1.size(1)
    window = create_gaussian_filter(window_size, sigma, channels)
    window = window.to(img1.device)

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channels)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channels)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channels) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channels) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean()


class SSIMLoss(torch.nn.Module):
    def __init__(self, window_size=11, sigma=1.5, data_range=1.0):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.data_range = data_range

    def forward(self, img1, img2):
        return 1 - ssim(img1, img2, self.window_size, self.sigma, self.data_range)
