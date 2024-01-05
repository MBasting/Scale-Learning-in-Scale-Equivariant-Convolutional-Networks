import os
from enum import Enum, auto
from torch.utils.data import DataLoader

# Relative Imports
from utils import loaders
from utils import loaders_org


class ChoiceDataset(Enum):
    MNIST_scale = auto()
    FASHIONMNIST_scale = auto()
    STL_10 = auto()
    CIFAR_10 = auto()


def update_dist_params(dist_params, cutoff, resize_factor):
    if resize_factor != 1:
        if dist_params[0] == "UNIFORM" or dist_params[0] == "LOGUNIFORM":
            if dist_params[0] == "UNIFORM":
                dist_params[2] = dist_params[2] - dist_params[1]
                dist_params[2] = round(dist_params[2] / resize_factor, 3)
            else:
                dist_params[2] = round(dist_params[2] / resize_factor, 3)

            dist_params[1] = round(dist_params[1] / resize_factor, 3)

            cutoff[0] = round(cutoff[0] / resize_factor, 3)
            cutoff[1] = round(cutoff[1] / resize_factor, 3)
    else:
        # Scipy uniform uses distance from left boundary instead of right boundary directly!
        if dist_params[0] == "UNIFORM":
            dist_params[2] = round(dist_params[2] - dist_params[1], 2)

    return dist_params, cutoff


def get_dataset(
    type,
    path_to_data,
    d_index,
    calc_acc_per_scale,
    seed,
    discrete,
    discrete_scale,
    dynamic,
    dist_params,
    cutoff,
    resize_factor,
    size_train_perc,
    extra_scaling,
    in_depth_test,
    img_size,
    batch_size,
    eval_batch_size,
    fit_single_batch,
    nr_workers,
    **kwargs,
):
    fashion = "Fashion" in ChoiceDataset(d_index).name

    # batch_size, eval_batch_size, fit_single_batch, nr_workers
    if img_size is None:
        img_size = int(28 * resize_factor)

    # Update distribution parameters for input to the samplers (require different format)
    dist_params_new, cutoff_new = update_dist_params(
        dist_params.copy(), cutoff.copy(), resize_factor
    )
    dist_name = dist_params_new[0]
    loc = dist_params_new[1]
    scale_param = dist_params_new[2]
    need_scales = False
    if type == "train":
        in_depth_test = False

    elif type == "val":
        size_train_perc = 1.0
        extra_scaling = 1.0
        in_depth_test = False
    else:
        size_train_perc = 1.0
        extra_scaling = 1.0

        need_scales = calc_acc_per_scale and type == "test"

    if dynamic:
        if discrete:
            # Can have either single scale
            if isinstance(discrete_scale, float):
                dataset = loaders.MnistSingleScale(
                    os.path.join(path_to_data, type), discrete_scale
                )
            else:
                dataset = loaders.MnistMultiScale(
                    os.path.join(path_to_data, type),
                    scales_used=discrete_scale,
                    discrete=True,
                    seed=seed,
                    perc=size_train_perc,
                    need_scale_labels=need_scales,
                    img_size=img_size,
                    fashion=fashion,
                    in_depth=in_depth_test,
                )
        else:
            dataset = loaders.MnistMultiScale(
                os.path.join(path_to_data, type),
                scales_used=discrete_scale,
                discrete=False,
                seed=seed,
                dist_name=dist_name,
                loc=loc,
                scale=scale_param,
                cutoff=cutoff_new,
                perc=size_train_perc,
                need_scale_labels=need_scales,
                img_size=img_size,
                fashion=fashion,
                in_depth=in_depth_test,
            )
    else:
        # Branch to shrink the dataset, only available for train dataloader
        dataset = loaders.MNistScale(
            os.path.join(path_to_data, type),
            need_scales=need_scales,
            extra_scaling=extra_scaling,
            perc=size_train_perc,
        )

    if type == "train":
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True if not fit_single_batch else False,
            pin_memory=True,
            num_workers=nr_workers,
        )
    else:
        loader = DataLoader(
            dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            pin_memory=False,
            num_workers=nr_workers,
        )
    return loader


def factory(cfg):
    """
    Returns:
        train_loader: Training dataset loader.
        val_loader: Validation dataset loader. Validation is performed after
            each training epoch. If None, no validation is performed.
        test_loader: Test dataset loader. Testing is performed after fitting is
            done. If None, no testing is performed.
    """
    dataset = ChoiceDataset(cfg.dataset.d_index)
    default = True
    # Load Dataset
    if (
        dataset is ChoiceDataset.MNIST_scale
        or dataset is ChoiceDataset.FASHIONMNIST_scale
    ):
        name = dataset.name
        fashion = "Fashion" in dataset.name
        if cfg.dataset.dynamic:
            default = False
            name = "MNIST_single" if not fashion else "FashionMNIST_single"
            if cfg.dataset.img_size != 28:
                name += f"_img_size_{cfg.dataset.img_size}"
            # Since we have the labels, calculating accuracy per binned scale is easy!
            cfg.dataset.calc_acc_per_scale = True
            path_to_data = f"{cfg.dataset.root}/{name}/seed_{cfg.seed}/scale_1.0_1.0"

        else:
            if cfg.dataset.additional_train:
                name += "_big"
                default = False
            if cfg.dataset.generation_mode is not None:
                cfg.dataset.calc_acc_per_scale = True
                name += "_" + cfg.dataset.generation_mode
                default = False
            else:
                cfg.dataset.calc_acc_per_scale = False
            path_to_data = f"{cfg.dataset.root}/{name}/seed_{cfg.seed}/scale_0.3_1.0"

        if cfg.dataset.val_size != 2000:
            name += f"_val_{cfg.dataset.val_size}"
        if cfg.dataset.resize_factor > 1.0:
            name += f"_{cfg.dataset.resize_factor:.1f}"
        print(path_to_data)
        if default:
            train_loader = loaders_org.scale_mnist_train_loader(cfg.train.batch_size, f"{cfg.dataset.root}/MNIST_scale/seed_{cfg.seed}/scale_0.3_1.0/", cfg.dataset.extra_scaling, num_workers=cfg.dataset.nr_workers)
            val_loader = loaders_org.scale_mnist_val_loader(cfg.train.batch_size, f"{cfg.dataset.root}/MNIST_scale/seed_{cfg.seed}/scale_0.3_1.0/", num_workers=cfg.dataset.nr_workers)
            test_loader = loaders_org.scale_mnist_test_loader(cfg.train.batch_size, f"{cfg.dataset.root}/MNIST_scale/seed_{cfg.seed}/scale_0.3_1.0/", num_workers=cfg.dataset.nr_workers)

            # Fill in useful information
            cfg.dataset.nr_samples = len(train_loader.dataset)
            cfg.dataset.nr_classes = 10
            cfg.dataset.img_size = (
                28,
                28,
            )
            cfg.dataset.in_channels = 1
        else:
            train_loader = get_dataset(
                "train",
                path_to_data,
                seed=cfg.seed,
                **cfg.dataset,
                **cfg.debug,
                **cfg.train,
            )
            val_loader = get_dataset(
                "val", path_to_data, seed=cfg.seed, **cfg.dataset, **cfg.debug, **cfg.train
            )
            if not cfg.wandb.sweep:
                test_loader = get_dataset(
                    "test",
                    path_to_data,
                    seed=cfg.seed,
                    **cfg.dataset,
                    **cfg.debug,
                    **cfg.train,
                )
                cfg.dataset.nr_scales = test_loader.dataset.nr_scales
            else:
                test_loader = None
                cfg.dataset.nr_scales = train_loader.dataset.nr_scales

            # Fill in useful information
            cfg.dataset.nr_samples = len(train_loader.dataset)
            print(cfg.dataset.nr_samples)
            cfg.dataset.nr_classes = train_loader.dataset.nr_classes
            cfg.dataset.img_size = (
                train_loader.dataset.img_size,
                train_loader.dataset.img_size,
            )
            cfg.dataset.in_channels = 1
    elif dataset is ChoiceDataset.STL_10:
        root = f"{cfg.dataset.root}/{dataset.name}"
        train_loader = loaders.stl10_plus_train_loader(
            cfg.train.batch_size, root, nr_workers=cfg.dataset.nr_workers, download=True
        )
        val_loader = loaders.stl10_test_loader(
            cfg.train.batch_size, root, nr_workers=cfg.dataset.nr_workers, download=True
        )
        test_loader = None

        cfg.dataset.nr_samples = len(train_loader.dataset)
        cfg.dataset.nr_classes = 10
        cfg.dataset.nr_scales = 1
        cfg.dataset.img_size = (96, 96)
        cfg.dataset.in_channels = 3
    elif dataset is ChoiceDataset.CIFAR_10:
        root = f"{cfg.dataset.root}/{dataset.name}"
        train_loader = loaders.cifar10_plus_train_loader(
            cfg.train.batch_size, root, nr_workers=cfg.dataset.nr_workers, download=True
        )
        val_loader = loaders.cifar10_test_loader(
            cfg.train.batch_size, root, nr_workers=cfg.dataset.nr_workers, download=True
        )
        test_loader = None

        cfg.dataset.nr_samples = len(train_loader.dataset)
        cfg.dataset.nr_classes = 10
        cfg.dataset.nr_scales = 1
        cfg.dataset.img_size = (32, 32)
        cfg.dataset.in_channels = 3
    else:
        print("This Dataset is not implemented!")
        return
    print(f"Length Training: {cfg.dataset.nr_samples}, Val: {len(val_loader.dataset)}")
    if not cfg.wandb.sweep and test_loader is not None:
        print(f"Length Test: {len(test_loader.dataset)} ")
    print("-------------------------------------------------------------------------")

    return train_loader, val_loader, test_loader
