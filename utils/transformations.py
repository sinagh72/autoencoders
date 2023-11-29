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
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
