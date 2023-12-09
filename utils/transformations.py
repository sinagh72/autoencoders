import numpy as np
import torch
from torchvision import transforms

def get_train_transformation(size=256):
    return transforms.Compose([
        transforms.Resize(size=(size, size)),
        transforms.RandomRotation(degrees=(-45, 45)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.5, contrast=0.5),
        transforms.Grayscale(3),
        transforms.ToTensor(),
        transforms.RandomApply(torch.nn.ModuleList([transforms.GaussianBlur(kernel_size=3)]), p=0.25),
        transforms.RandomApply(torch.nn.ModuleList([transforms.ElasticTransform(alpha=(28.0, 30.0),
                                                                                sigma=(3.5, 4.0))]), p=0.3),
        transforms.Normalize((0.5,), (0.5,))

    ])


def get_test_transformation(size=224):
    return transforms.Compose([
        transforms.Resize(size=(size, size)),
        transforms.Grayscale(3),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

class MaskGenerator:
    def __init__(self, input_size=192, mask_patch_size=32, model_patch_size=4, mask_ratio=0.6):
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio

        assert self.input_size % self.mask_patch_size == 0
        assert self.mask_patch_size % self.model_patch_size == 0

        self.rand_size = self.input_size // self.mask_patch_size
        self.scale = self.mask_patch_size // self.model_patch_size

        self.token_count = self.rand_size ** 2
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))

    def __call__(self):
        mask_idx = np.random.permutation(self.token_count)[:self.mask_count]
        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1

        mask = mask.reshape((self.rand_size, self.rand_size))
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1)

        return mask


class SimMIMTransform:
    def __init__(self, img_size, model_patch_size, mask_patch_size, mask_ratio, mean, std):
        self.transform_img = get_train_transformation()

        self.mask_generator = MaskGenerator(
            input_size=img_size,
            mask_patch_size=mask_patch_size,
            model_patch_size=model_patch_size,
            mask_ratio=mask_ratio,
        )

    def __call__(self, img):
        img = self.transform_img(img)
        mask = self.mask_generator()

        return img, mask