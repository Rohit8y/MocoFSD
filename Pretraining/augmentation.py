import random

from PIL import ImageFilter
import torchvision.transforms as transforms
from scipy.fft import dct



class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)

        transImage = transforms.ToPILImage()
        transTensor = transforms.ToTensor()

        q = transTensor(transImage(q).convert('YCbCr'))
        k = transTensor(transImage(k).convert('YCbCr'))
        k = transTensor(dct(transImage(k))).float()
        return [q, k]


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x