'''MIT License. Copyright (c) 2020 Ivan Sosnovik, Michał Szmaja'''
import math
import os
import random
import hashlib
from glob import glob
import warnings
from matplotlib import pyplot as plt

import numpy as np
from sklearn.model_selection import train_test_split
from scipy import logical_and
from scipy.stats import norm, cauchy, uniform
import torchvision.transforms.functional as func
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets, transforms
import PIL

import pandas as pd

BUF_SIZE = 65536


def get_md5_from_source_path(source_path):
    pattern = os.path.join(source_path, '**', '**', '*.png')
    files = sorted(list(glob(pattern)))
    assert len(files)

    md5 = hashlib.md5()

    for file_path in files:
        with open(file_path, 'rb') as f:
            while True:
                data = f.read(BUF_SIZE)
                if not data:
                    break
                md5.update(data)

    return md5.hexdigest()


def _save_images_to_folder(dataset, path, split_name, idx, min, max, format_='.png', gauss_mode='none', use_cauchy=False, resize_factor = 1.0, img_size=28):
    min_size = 0
    index = 0
    
    imgs = []
    labels = []
    scales = []
    if gauss_mode != 'none':
        mean = (min + max) / 2
        if gauss_mode == 'Middle':
            middle = mean
        elif gauss_mode == 'Small':
            middle = min
        elif gauss_mode == 'Large':
            middle = max

        sampler = None
        if not use_cauchy:
            std = (mean - min) / 3
            sampler = norm.rvs
        else:
            std = 0.03
            sampler = cauchy.rvs
        rvs = sampler(size=4 * len(dataset), loc=middle, scale=std)
        rvs = rvs[np.logical_and(rvs<=1,rvs>=0.3)][:len(dataset)]
    else:
        if max - min != 0:
            rvs = uniform.rvs(size=len(dataset), loc=min, scale=max-min)
        else:
            rvs = np.ones(len(dataset)) * min
    min_size = int(math.ceil(min * img_size))
    for img, label in dataset:
        if resize_factor > 1:
            warnings.warn('resize_factor > 1.0, this setting has priority over img_size and should thus be disabled if you want to use img_size')
            img = transforms.Resize(int(28 * resize_factor), interpolation=func.InterpolationMode.BICUBIC)(img)
        elif img_size != 28:
            # Increase img size by padding with zeros
            img = func.center_crop(img, img_size)
            warnings.warn('img_size != 28 and resize_factor == 1.0, this will result in a non-standard dataset')
            warnings.warn('Note that centercropping is used and thus some discritization might occur if img_size - 28 % 2 != 0')

        if rvs[index] != 1:
            img = func.affine(img, angle = 0, translate = (0, 0), shear = 0, scale = rvs[index], interpolation=func.InterpolationMode.BICUBIC)

        out = os.path.join(path, split_name, str(label))
    
        if not os.path.exists(out):
            os.makedirs(out)
        img_path = os.path.join(out, str(idx) + format_)
        img.save(img_path)

        imgs.append(os.path.join(str(label), str(idx) + format_))
        labels.append(label)
        scales.append(int(math.ceil(rvs[index] * img_size) - min_size))
        
        index += 1
        idx += 1
    data = {'img_path' : imgs, 'labels' : labels, 'scales' : scales}
    df = pd.DataFrame(data)
    return idx, df


def make_mnist_scale_50k(source, dest, min_scale, max_scale, download=False, seed=0, single_scale_train = False, fashion=False, equi=False, small_test = False, gauss_mode='none', cauchy=False, resize_factor = 1.0, img_size = 28, val_size = 2000, **kwargs):
    '''
    We follow the procedure described in 
    https://arxiv.org/pdf/1807.11783.pdf
    https://arxiv.org/pdf/1906.03861.pdf
    '''
    TRAIN_SIZE = 50000 if small_test else 10000
    VAL_SIZE = val_size
    TEST_SIZE = 10000 if small_test else 50000
    name = 'MNIST_scale'
    if fashion:
        name = "Fashion" + name
    if equi:
        name = name + "_equi"
    if small_test:
        name = name + "_big"
    if gauss_mode != 'none':
        name = name + "_" + gauss_mode
    if single_scale_train:
        min_test_scale = 1.0
        max_test_scale = 1.0
        name = 'MNIST_single'
        name = "Fashion" + name if fashion else name
    else:
        min_test_scale = min_scale
        max_test_scale = max_scale
    if img_size != 28:
        name += f'_img_size_{img_size}'
    if val_size != 2000:
        if val_size > 10000:
            val_size = 10000
        name += f"_val_{val_size}"
    np.random.seed(seed)
    random.seed(seed)

    if not fashion:
        dataset_train = datasets.MNIST(root=source, train=True, download=download)
        dataset_test = datasets.MNIST(root=source, train=False, download=download)
    else:
        dataset_train = datasets.FashionMNIST(
            root=source, train=True, download=download)
        dataset_test = datasets.FashionMNIST(
            root=source, train=False, download=download)

    if resize_factor > 1:
        name += f'_{resize_factor}'

    concat_dataset = ConcatDataset([dataset_train, dataset_test])

    labels = [el[1] for el in concat_dataset]
    train_val_size = TRAIN_SIZE + VAL_SIZE
    train_val, test = train_test_split(concat_dataset, train_size=train_val_size,
                                       test_size=TEST_SIZE, stratify=labels, random_state=seed)

    labels = [el[1] for el in train_val]
    train, val = train_test_split(train_val, train_size=TRAIN_SIZE,
                                  test_size=VAL_SIZE, stratify=labels, random_state=seed)
    
    dest = os.path.expanduser(dest)
    dataset_path = os.path.join(dest, name, "seed_{}".format(seed))
    
    dataset_path = os.path.join(
        dataset_path, "scale_{}_{}".format(min_scale, max_scale))
    print('OUTPUT: {}'.format(dataset_path))




    idx, df_data_train = _save_images_to_folder(
        train, dataset_path, 'train', 0, min_scale, max_scale, '.png', gauss_mode=gauss_mode, use_cauchy=cauchy, resize_factor=resize_factor, img_size=img_size)
    idx, df_data_test = _save_images_to_folder(
        test, dataset_path, 'test', idx, min_test_scale, max_test_scale, '.png', resize_factor=resize_factor, img_size=img_size) # If we use non-uniform train dataset scale distribution we would like to evaluate per scale as well
    idx, df_data_val = _save_images_to_folder(
        val, dataset_path, 'val', idx, min_scale, max_scale, '.png', resize_factor=resize_factor, img_size=img_size)
    
    df_data_train.to_csv(os.path.join(dataset_path, 'train', 'data.csv'))
    df_data_test.to_csv(os.path.join(dataset_path, 'test', 'data.csv'))
    df_data_val.to_csv(os.path.join(dataset_path, 'val', 'data.csv'))


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--source', type=str, required=True,
                        help='source folder of the dataset')
    parser.add_argument('--dest', type=str, required=True,
                        help='destination folder for the output')
    parser.add_argument('--min_scale', type=float, required=True,
                        help='min scale for the generated dataset')
    parser.add_argument('--max_scale', type=float, default=1.0,
                        help='max scale for the generated dataset')
    parser.add_argument('--download', action='store_true',
                        help='donwload stource dataset if needed.')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--fashion', default=False)
    parser.add_argument('--equi', default=False)
    parser.add_argument('--small_test', default=False)
    parser.add_argument('--gauss_mode', default='none')
    parser.add_argument('--cauchy', default=False, help='If the distribution used to sample the scales from is Normal (if gauss_mode is not None)')
    parser.add_argument('--single_scale_train', default=False, help='Only synthesize single scale MNIST - note test scale is default')
    parser.add_argument('--resize_factor', type=float, default=1.0)
    parser.add_argument('--img_size', type=int, default=28)
    parser.add_argument('--val_size', type=int, default=2000)
    parser.add_argument('--validate', action='store_true', default=False)
    args = parser.parse_args()

    if args.validate:
        dest = os.path.expanduser(args.dest)
        dataset_path = os.path.join(
            dest, 'MNIST_scale', "seed_{}".format(args.seed))
        dataset_path = os.path.join(dataset_path,
                                    "scale_{}_{}".format(args.min_scale, args.max_scale))
        print(get_md5_from_source_path(dataset_path))
    else:
        for k, v in vars(args).items():
            print('{}={}'.format(k, v))
        make_mnist_scale_50k(**vars(args))

