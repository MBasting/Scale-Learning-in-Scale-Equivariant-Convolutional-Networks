'''
1) "Scale Learning in Scale-Equivariant Convolutional Networks"
    by Mark Basting, Jan van Gemert, VISAPP 2024,
    pdf: ...

---------------------------------------------------------------------------

The sources of this file are parts of 
1) the official implemenation of "Scale Learning in Scale-Equivariant Convolutional Networks"
    by Mark Basting, Jan van Gemert, VISAPP 2024,
    arxiv : 
    code: https://github.com/MBasting/scale-equiv-learnable-cnn
'''

from enum import Enum
import math
import os
import pickle
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import scipy.stats

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import torchvision.transforms.functional as func
from PIL import Image
from tqdm import tqdm

from .cutout import Cutout


mean = {
    'stl10': (0.4467, 0.4398, 0.4066),
    'cifar10': (0.4914, 0.4822, 0.4465),
    'scale_mnist': (0.0607,),
    'scale_fashion' : (0.3299,),
}

std = {
    'stl10': (0.2603, 0.2566, 0.2713),
    'cifar10': (0.247, 0.243, 0.261),
    'scale_mnist': (0.2161,),
    'scale_fashion' : (1.2766),
}

def truncated_log_normal(mu, sigma, cutoff_a, cutoff_b, size = 10000):
    values = []
    while len(values) < size:
        val = np.random.lognormal(mean=mu,sigma=sigma,size=None)
        if val>=cutoff_a and val <= cutoff_b:
            values.append(val)
    return values

class Distribution(Enum):
    UNIFORM = scipy.stats.uniform,
    GAUSSIAN = scipy.stats.truncnorm,
    LOGUNIFORM = scipy.stats.loguniform,
    TRUNCLOGNORMAL = truncated_log_normal,

class MnistSingleScale(Dataset):
    def __init__(self, path_to_data, scale, shuffle=True):
        """
            Args:
            path_to_data (string): Path to folder with images
            scale (float): Scaling parameter that should be used to scale each sample
        """
        super().__init__()
        transform_modules = []
        name = 'scale_mnist'
        self.scale = scale
        self.mean = mean[name]
        self.std = std[name]
        transform_modules = [
            transforms.RandomAffine(0, scale=[scale, scale], interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.Grayscale(),
            transforms.ToTensor(),
        ]

        # Load unchanged MNIST but with Train/Test/Val division already done
        # Only transform on getting item
        dataset = datasets.ImageFolder(path_to_data)
        self.imgs = [img for img, _ in dataset.imgs]
        self.targets = dataset.targets 
        self.classes = dataset.classes
        self.img_size = 28

        self.update_mean_std(transforms.Compose(transform_modules))

        transform_modules.append(transforms.Normalize(self.mean, self.std))

        self.transform = transforms.Compose(transform_modules)

        if shuffle:
            indices = random.sample(range(len(self.imgs)), len(self.imgs))
            self.imgs = [self.imgs[i] for i in indices]
            self.targets = [self.targets[i] for i in indices]

        self.nr_classes = len(self.classes)
        self.nr_scales = 0
    
    def update_mean_std(self, transform):
        sum_pixels = torch.tensor([0.0])
        sum_squared_pixels = torch.tensor([0.0])
        for img in self.imgs:
            path = img
            with open(path, "rb") as f:
                img = Image.open(f)
                img.convert("RGB")
            img = transform(img)
            sum_pixels += img.sum(axis = [1, 2])
            sum_squared_pixels += (img ** 2).sum(axis = [1,2])

        count = len(self.imgs) * self.img_size * self.img_size
        self.mean = sum_pixels / count
        self.std = torch.sqrt((sum_squared_pixels / count) - (self.mean ** 2))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        path = self.imgs[idx]
        with open(path, "rb") as f:
            img = Image.open(f)
            img.convert("RGB")

        img = self.transform(img)
        return img, self.targets[idx]

class MnistMultiScale(Dataset):
    def __init__(self, path_to_data, scales_used, discrete, seed, fashion = False, dist_name = 'UNIFORM', loc=0.3, scale=0.7, cutoff = [0.3, 1.0],
                 need_scale_labels=False, perc=1, img_size=28, obj_size = None, in_depth = False, 
                normalize_per_scale=False, reuse_train_mean_std=False):
                # TODO: Implement reuse_train_mean_std

        """
        Args:
            path_to_data (string): Path to folder with images
            scales (list): List of floats (same size as dataset) with floats that hold scale applied to each sample individually
        """
        super().__init__()
        if in_depth:
            assert(not (normalize_per_scale and reuse_train_mean_std))
        transform_modules = []
        name = 'scale_mnist' if not fashion else 'scale_fashion'
        self.img_size = img_size
        self.cutoff = cutoff
        self.mean = mean[name]
        self.std = std[name]
        self.need_labels = need_scale_labels
        self.pad = 0
        if cutoff[1] > 1.4 and self.img_size == 28:
            print("PAD IMAGES")
            self.pad = 14
            self.img_size = self.img_size + 2 * self.pad
            self.obj_size = 28
        elif obj_size is not None:
            self.obj_size = obj_size
        else:
            self.obj_size = img_size
        transform_modules = [
            transforms.Grayscale(),
            transforms.ToTensor(),
        ]

        # Load unchanged MNIST but with Train/Test/Val division already done
        # Only transform on getting item
        dataset = datasets.ImageFolder(path_to_data)

        self.imgs = [img for img, _ in dataset.imgs]
        self.targets = dataset.targets 
        self.classes = dataset.classes
        self.scale_labels = None
        self.in_depth = in_depth
        self.normalize_per_scale = normalize_per_scale

        if discrete:
            # Create array holding all scale value for each sample (need shuffling because resizing fixed order)
            self.scales = np.resize(scales_used, len(self.imgs))
            np.random.shuffle(self.scales ) 
            self.scale_labels = np.searchsorted(scales_used, self.scales)
            self.nr_scales = len(scales_used)
            self.mean, self.std = self.calc_mean_std(transforms.Compose(transform_modules))

        else:
            if in_depth:
                print("Round to nearest half in log2space")
                assert cutoff[0] != 0
                min_scale = round(np.log2(cutoff[0]) * 2) / 2.0
                max_scale = round(np.log2(cutoff[1]) * 2) / 2.0
                nr_bins = int(round(max_scale - min_scale, 1) * 10) # ! Semi-Hardcoded
                # Update cutoff
                self.cutoff = [round(2**min_scale,3), round(2**max_scale,3)]
                # Seed is important since we still have the train/val/test split
                pkl_name = f'Log2_scales_[{self.cutoff[0]},{self.cutoff[1]}]_nr_bins_{nr_bins}_seed_{seed}_norm_p_scale_{normalize_per_scale}'
                if reuse_train_mean_std:
                    pkl_name += f'_reuse_params_{dist_name}_{loc}_{scale}'
            else:
                if dist_name != 'GAUSSIAN' and dist_name != 'TRUNCLOGNORMAL':
                    # Cutoff not used and thus not incorporated into name
                    pkl_name = f'{dist_name}_loc_{loc}_scale_{scale}_seed_{seed}'
                else:
                    pkl_name = f'{dist_name}_{loc}_{scale}_cutoff_[{cutoff[0]},{cutoff[1]}]_seed_{seed}'

            pkl_name += '.pkl'
            file_path = os.path.join(path_to_data, pkl_name)
            # If there already is a generated seed use
            if os.path.isfile(file_path):
                print(f"DATA FROM {pkl_name}")
                with open(file_path, 'rb') as fp:
                    data = pickle.load(fp)

                self.scales = data['scales']
                if in_depth:
                    # Load scales
                    self.nr_scales = len(self.scales)
                    self.scale_labels = np.arange(0, self.nr_scales)
                    print(f"Loaded {self.nr_scales} scales")
                
                # Load mean, std
                self.mean = torch.tensor(data['mean'])
                self.std = torch.tensor(data['std'])

                print(self.mean, self.std)
                print(self.scales)
            else:
                # Need to generate the scales
                if in_depth:
                    print("ONLY USE THIS SETTING IN TEST MODE, THIS SAMPLES EACH DIGIT ON ALL SCALES")
                    self.scales = np.logspace(min_scale, max_scale, nr_bins, base=2)
                    if min_scale == max_scale:
                        self.scales = [2**min_scale]
                    self.nr_scales = len(self.scales)
                    self.scale_labels = np.arange(0, self.nr_scales)
                    print(self.scales)
                else:
                    sampler = Distribution[dist_name].value[0]
                    if dist_name == 'GAUSSIAN':
                        a, b = (cutoff[0] - loc) / scale, (cutoff[1] - loc) / scale
                        self.scales = sampler.rvs(a, b, size=len(self.imgs), loc=loc, scale=scale)
                        # Only need to update scales if we don't pad images
                        if self.img_size > 28 and self.pad == 0:
                            self.scales = [scale_el / (self.img_size / 28) for scale_el in self.scales]
                    elif dist_name == 'TRUNCLOGNORMAL':
                        self.scales = sampler(loc, scale, cutoff[0], cutoff[1], size=len(self.imgs))
                        if self.img_size > 28 and self.pad == 0:
                            self.scales = [scale_el / (self.img_size / 28) for scale_el in self.scales]
                    else:
                        # In some cases need to cutoff tails of the distribution
                        self.scales = sampler.rvs(loc, scale, size = len(self.imgs))

                data = {'scales' : self.scales}
                
                # Special case where we want to reuse the mean and std from the train set in testing mode
                if reuse_train_mean_std and in_depth:
                        # Reload train_data:
                        print("Reloading Training Data")
                        train_data = MnistMultiScale(path_to_data, scales_used, discrete, seed, fashion, dist_name, loc, scale, cutoff,
                                                     need_scale_labels, perc, img_size, obj_size)
                        self.mean, self.std = train_data.mean, train_data.std
                else:
                    self.mean, self.std = self.calc_mean_std(transforms.Compose(transform_modules), normalize_per_scale=normalize_per_scale)
                
                data['mean'] = self.mean.tolist()
                data['std'] = self.std.tolist()
                                # Load mean, std
                print(self.mean, self.std)

                # Save to pickle file
                with open(file_path, 'wb') as fp:
                    pickle.dump(data, fp)
                print(f"Saved scale distribution to {pkl_name}")

            if self.scale_labels is None:
                print("In this case scale_labels represent pixel sizes")
                # In continuous case scales are built up of pixel sizes (binned to nearest)
                self.min_size = math.floor(min(self.scales) * self.obj_size)
                self.scale_labels = [int(round(scale * self.obj_size) - self.min_size) for scale in self.scales]
                self.nr_scales = max(self.scale_labels) + 1
        
        # Check if mean/std is already calculated
        # Update mean + create transform pipeline
        if not self.in_depth or not normalize_per_scale:
            transform_modules.append(transforms.Normalize(self.mean, self.std))
        self.transform = transforms.Compose(transform_modules)

        # Shrink dataset if necessary
        if perc < 1 and not self.in_depth:
            indices = np.arange(len(self.imgs))
            train_indices, _ = train_test_split(indices, train_size=int(perc*len(self.imgs)), stratify=self.targets)
            self.imgs = [self.imgs[i] for i in train_indices]
            self.targets = [self.targets[i] for i in train_indices]
            self.scales = [self.scales[i] for i in train_indices]
            self.scale_labels = [self.scale_labels[i] for i in train_indices]

        self.nr_classes = len(self.classes)

    def calc_mean_std(self, transform, normalize_per_scale=False):  
        # Method calculates the mean and standard deviation of the dataset in single pass for normalization
        if self.in_depth and normalize_per_scale:
            sum_pixels = torch.zeros(self.nr_scales)
            sum_squared_pixels = torch.zeros(self.nr_scales)
        else:
            sum_pixels = torch.tensor([0.0])
            sum_squared_pixels = torch.tensor([0.0])
        with tqdm(total=len(self)) as pbar:
            for i, img in enumerate(self.imgs):
                path = img
                with open(path, "rb") as f:
                    img = Image.open(f)
                    img.convert("RGB")
                if self.pad > 0:
                    img = func.pad(img, self.pad)
                if len(self.scales) < len(self.imgs):
                    # We are in in_depth test mode and thus must loop through all images at all scales:
                    for index_scale, scale in enumerate(self.scales):
                        img_t = func.affine(img.copy(), angle = 0.0, translate=(0,0), shear = 0.0, scale=scale, interpolation=transforms.InterpolationMode.BICUBIC)
                        img_t = transform(img_t)
                        if normalize_per_scale:
                            sum_pixels[index_scale] = sum_pixels[index_scale] + img_t.sum(axis = [1, 2])
                            sum_squared_pixels[index_scale] = sum_squared_pixels[index_scale] + (img_t ** 2).sum(axis = [1,2])
                            pbar.update(1)
                        else:
                            sum_pixels = sum_pixels + img_t.sum(axis = [1, 2])
                            sum_squared_pixels = sum_squared_pixels + (img_t ** 2).sum(axis = [1,2])
                            pbar.update(1)
                else:
                    img = func.affine(img, angle = 0.0, translate=(0,0), shear = 0.0, scale=self.scales[i], interpolation=transforms.InterpolationMode.BICUBIC)
                    img = transform(img)
                    sum_pixels += img.sum(axis = [1, 2])
                    sum_squared_pixels += (img ** 2).sum(axis = [1,2])
                    pbar.update(1)
        count = len(self.imgs) * self.img_size * self.img_size

        mean = sum_pixels / count
        std = torch.sqrt((sum_squared_pixels / count) - (mean ** 2))
        print("Dataset Mean", mean, std)
        return mean, std


    def __len__(self):
        if self.in_depth:
            return len(self.imgs) * self.nr_scales
        else:
            return len(self.imgs)

    def __getitem__(self, idx):
        if self.in_depth:
            index_img = idx % len(self.imgs)
            index_scale = idx // len(self.imgs)
        else:
            index_img = idx
            index_scale = idx
            
        path = self.imgs[index_img]
        scale = self.scales[index_scale]
        scale_label = self.scale_labels[index_scale]

        with open(path, "rb") as f:
            img = Image.open(f)
            img.convert("RGB")
            
        if self.pad > 0:
            img = func.pad(img, self.pad)
        # Perform scaling manually
        img = func.affine(img, angle = 0.0, translate=(0,0), shear = 0.0, scale=scale, interpolation=transforms.InterpolationMode.BICUBIC)
        img = self.transform(img)
        if self.in_depth and self.normalize_per_scale:
            # Normalize with mean / std per scale
            img = func.normalize(img, self.mean[index_scale], self.std[index_scale])
        if self.need_labels:
            return img, self.targets[index_img], scale_label
        else:
            return img, self.targets[index_img]


class MNistScale(Dataset):
    """MNIST Scale"""

    def __init__(self, path_to_data, fashion=False, need_scales = False, extra_scaling=1, perc=1, csv_file='data.csv'):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        data = pd.read_csv(os.path.join(path_to_data, csv_file))
        self.path_to_data = path_to_data
        self.fashion = fashion
        self.need_scales = need_scales
        self.imgs = data.iloc[:, 1]
        self.targets = data.iloc[:, 2]
        self.scales = data.iloc[:, 3]
        self.img_size = 28
        
        name = 'scale_mnist' if not fashion else 'scale_fashion'
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
            transforms.Normalize(mean[name], std[name])
        ]

        self.transform = transforms.Compose(transform_modules)


        if perc < 1:
            indices = np.arange(len(self.imgs))
            train_indices, _ = train_test_split(indices, train_size=int(perc*len(self.imgs)), stratify=self.targets)
            self.imgs = [j for i,j in enumerate(self.imgs) if i in train_indices]
            self.targets = [j for i,j in enumerate(self.targets) if i in train_indices]
            if self.need_scales:
                self.scales = [j for i,j in enumerate(self.scales) if i in train_indices]
        
        self.nr_classes = max(self.targets) + 1
        if self.need_scales:
            self.nr_scales = int(max(self.scales) + 1)
        else:
            self.nr_scales = 0

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        path = os.path.join(self.path_to_data, self.imgs[idx])
        with open(path, "rb") as f:
            img = Image.open(f)
            img.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        
        label = self.targets[idx]
        if not self.need_scales:
            return img, label
        scale = self.scales[idx]
        return [img, label, scale]


#################################################
##################### STL-10 ####################
#################################################

def stl10_plus_train_loader(batch_size, root, nr_workers = 2, download=True):
    transform = transforms.Compose([
        transforms.RandomCrop(96, padding=12),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean['stl10'], std['stl10']),
        Cutout(1, 32),
    ])
    dataset = datasets.STL10(root=root, transform=transform, download=download)
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=True, pin_memory=True, num_workers=nr_workers)
    return loader


def stl10_test_loader(batch_size, root, nr_workers = 2, download=True):
    transform = transforms.Compose([
        transforms.CenterCrop(96),
        transforms.ToTensor(),
        transforms.Normalize(mean['stl10'], std['stl10'])
    ])
    dataset = datasets.STL10(root=root, split='test', transform=transform, download=download)
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, pin_memory=True, num_workers=nr_workers)
    return loader

def cifar10_plus_train_loader(batch_size, root, nr_workers = 2, download=True):
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean['cifar10'], std['cifar10']),
    ])
    dataset = datasets.CIFAR10(root=root, train=True,transform=transform, download=download)
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=True, pin_memory=True, num_workers=nr_workers)
    return loader


def cifar10_test_loader(batch_size, root, nr_workers = 2, download=True):
    transform = transforms.Compose([
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize(mean['cifar10'], std['cifar10'])
    ])
    dataset = datasets.CIFAR10(root=root, train=False, transform=transform, download=download)
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, pin_memory=True, num_workers=nr_workers)
    return loader

#################################################
##################### SCALE #####################
#################################################
def scale_mnist_train_loader(batch_size, root, seed, extra_scaling=1, perc=1, fashion=False, equivariant = False, single_batch = True, dynamic=False, 
                             discrete= True, dist_name = 'UNIFORM', discrete_scale=0.3, loc=0.3, scale_param=0.7, cutoff = [0.3, 1.0], img_size=28,
                             num_workers = 2):
    
    if dynamic:
        if discrete:
            # Can have either single scale
            if isinstance(discrete_scale, float):
                dataset = MnistSingleScale(os.path.join(root, 'train'), discrete_scale)
            else:
                dataset = MnistMultiScale(os.path.join(root, 'train'), scales_used=discrete_scale, discrete=True, seed = seed, perc=perc, need_scale_labels=equivariant, img_size=img_size, fashion=fashion)        
        else:
            dataset = MnistMultiScale(os.path.join(root, 'train'), scales_used= discrete_scale, discrete=False, seed = seed, dist_name = dist_name,
                                      loc=loc, scale=scale_param, cutoff=cutoff, perc=perc, need_scale_labels=equivariant, img_size=img_size, fashion=fashion)
    else:
        # Branch to shrink the dataset, only available for train dataloader
        dataset = MNistScale(os.path.join(root, 'train'), fashion, need_scales=equivariant, extra_scaling=extra_scaling, perc=perc)
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=not single_batch, pin_memory=True, num_workers=num_workers)
    return loader


def scale_mnist_val_loader(batch_size, root, seed, fashion=False, equivariant = False, dynamic=False, 
                             discrete= True, dist_name = 'UNIFORM', discrete_scale=0.3, loc=0.3, scale_param=0.7, cutoff = [0.3, 1.0], img_size=28,
                             num_workers = 2):
    if dynamic:
        if discrete:
            # Can have either single scale
            if isinstance(discrete_scale, float):
                dataset = MnistSingleScale(os.path.join(root, 'val'), discrete_scale)
            else:
                dataset = MnistMultiScale(os.path.join(root, 'val'), discrete_scale, discrete=True, seed = seed, need_scale_labels=equivariant, img_size=img_size, fashion=fashion)      
        else:
            dataset = MnistMultiScale(os.path.join(root, 'val'), discrete_scale, discrete=False, seed = seed, dist_name = dist_name,
                                      loc=loc, scale=scale_param, cutoff=cutoff, need_scale_labels=equivariant, img_size=img_size, fashion=fashion)
    else:
        dataset = MNistScale(os.path.join(root, 'val'), fashion, need_scales=equivariant)
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, pin_memory=True, num_workers=num_workers)
    return loader


def scale_mnist_test_loader(batch_size, root, seed, fashion=False, equivariant = False, dynamic=False, 
                            discrete = True, dist_name = 'UNIFORM', discrete_scale=0.3, loc=0.3, scale_param=0.7, cutoff = [0.3, 1.0], img_size=28,
                            in_depth_test = False,
                            num_workers = 2):
    if dynamic:
        if discrete:
            # Can have either single scale
            if isinstance(discrete_scale, float):
                dataset = MnistSingleScale(os.path.join(root, 'test'), discrete_scale)
            else:
                dataset = MnistMultiScale(os.path.join(root, 'test'), discrete_scale, discrete=True, seed = seed, need_scale_labels=equivariant, img_size=img_size, fashion=fashion,
                                          in_depth = in_depth_test)      
        else:
            dataset = MnistMultiScale(os.path.join(root, 'test'), discrete_scale, discrete=False, seed = seed, dist_name = dist_name,
                                      loc=loc, scale=scale_param, cutoff=cutoff, need_scale_labels=equivariant, img_size=img_size, fashion=fashion,
                                      in_depth = in_depth_test)
    else:
        dataset = MNistScale(os.path.join(root, 'test'), fashion, equivariant)

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
