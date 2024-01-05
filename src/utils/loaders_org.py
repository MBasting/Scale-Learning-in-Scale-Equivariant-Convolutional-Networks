'''
1) "DISCO: accurate Discrete Scale Convolutions"
    by Ivan Sosnovik, Artem Moskalev, Arnold Smeulders, BMVC 2021
    arxiv: https://arxiv.org/abs/2106.02733

2) "How to Transform Kernels for Scale-Convolutions"
    by Ivan Sosnovik, Artem Moskalev, Arnold Smeulders, ICCV VIPriors 2021
    pdf: https://openaccess.thecvf.com/content/ICCV2021W/VIPriors/papers/Sosnovik_How_To_Transform_Kernels_for_Scale-Convolutions_ICCVW_2021_paper.pdf

3) "Scale Learning in Scale-Equivariant Convolutional Networks"
    by Mark Basting, Jan van Gemert, VISAPP 2024,
    pdf: ...

---------------------------------------------------------------------------

The sources of this file are parts of 
1) the official implementation of "Scale-Equivariant Steerable Networks"
    by Ivan Sosnovik, Michał Szmaja, and Arnold Smeulders, ICLR 2020
    arxiv: https://arxiv.org/abs/1910.11093
    code: https://github.com/ISosnovik/sesn

2) the official implementation of "Scale Equivariance Improves Siamese Tracking"
    by Ivan Sosnovik*, Artem Moskalev*, and Arnold Smeulders, WACV 2021
    arxiv: https://arxiv.org/abs/2007.09115
    code: https://github.com/ISosnovik/SiamSE

3) the official implemenation of "Scale Learning in Scale-Equivariant Convolutional Networks"
    by Mark Basting, Jan van Gemert, VISAPP 2024,
    arxiv : 
    code: https://github.com/MBasting/scale-equiv-learnable-cnn
'''

import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import json

from .cutout import Cutout


mean = {
    'stl10': (0.4467, 0.4398, 0.4066),
    'scale_mnist': (0.0607,),
}

std = {
    'stl10': (0.2603, 0.2566, 0.2713),
    'scale_mnist': (0.2161,),
}


#################################################
##################### STL-10 ####################
#################################################


def stl10_plus_train_loader(batch_size, root, download=True):
    transform = transforms.Compose([
        transforms.RandomCrop(96, padding=12),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean['stl10'], std['stl10']),
        Cutout(1, 32),
    ])
    dataset = datasets.STL10(root=root, transform=transform, download=download)
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=True, pin_memory=True, num_workers=2)
    return loader


def stl10_test_loader(batch_size, root, download=True):
    transform = transforms.Compose([
        transforms.CenterCrop(96),
        transforms.ToTensor(),
        transforms.Normalize(mean['stl10'], std['stl10'])
    ])
    dataset = datasets.STL10(root=root, split='test', transform=transform, download=download)
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, pin_memory=True, num_workers=2)
    return loader


#################################################
##################### SCALE #####################
#################################################
def scale_mnist_train_loader(batch_size, root, extra_scaling=1, num_workers=2):
    transform_modules = []
    if not extra_scaling == 1:
        if extra_scaling > 1:
            extra_scaling = 1 / extra_scaling
        scale = (extra_scaling, 1 / extra_scaling)
        print('-- extra scaling ({:.3f} - {:.3f}) is used'.format(*scale))
        scaling = transforms.RandomAffine(0, scale=scale, interpolation=transforms.InterpolationMode.BICUBIC)
        transform_modules.append(scaling)

    transform_modules = transform_modules + [
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean['scale_mnist'], std['scale_mnist'])
    ]

    transform = transforms.Compose(transform_modules)
    dataset = datasets.ImageFolder(os.path.join(root, 'train'), transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=True, pin_memory=True, num_workers=num_workers)
    return loader


def scale_mnist_val_loader(batch_size, root, num_workers=2):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean['scale_mnist'], std['scale_mnist'])
    ])
    dataset = datasets.ImageFolder(os.path.join(root, 'val'), transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=True, pin_memory=True, num_workers=num_workers)
    return loader


def scale_mnist_test_loader(batch_size, root, num_workers=2):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean['scale_mnist'], std['scale_mnist'])
    ])
    dataset = datasets.ImageFolder(os.path.join(root, 'test'), transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, pin_memory=True, num_workers=num_workers)
    return loader


#################################################
#################### RANDOM #####################
#################################################
def random_loader(batch_size):
    dataset = datasets.FakeData(size=10000, image_size=(1, 154, 154),
                                transform=transforms.ToTensor())
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=True, pin_memory=True, num_workers=2)
    return loader
