import math
import os
import warnings
from pathlib import Path
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
import wandb

# Relative Imports
from utils import loaders
from disco.ses_conv_learnable import SESConv_H_H_Learnable, SESConv_Z2_H_Learnable, SESConv_H_H, SESConv_Z2_H, SESMaxProjection
from utils.model_utils import ConvType, LearnMode, calculate_scales_parameter
from optimizer import MultiStepGroupSpecific, custom_CosineAnnealingLR, custom_LambdaLR

logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)

DEFAULT_CONFIG = {
    "exp_name": "test",
    "layer_name": "SESN",
    "resize_factor": 2,
    "img_size": 28,
    "obj_size": 28,
    "sample_scales": ["DISCRETE", 1],
    "init_scales": [1],
    "nr_internal": 2,
    "nr_layers": 1,
    "linear_hidden": None,
    "linear_dropout": None,
    "reference_scale": None,
    "nr_filters": 8,
    "eff_size": 7,
    "base_kernel_size": 7,
    "largest_kernel_size": None,
    "kernel_size_init_k" : None, 
    "simplified": False,
    "simple_dropout": 0.0,
    "learnable_basis_min": False,
    "learn_mode": LearnMode.OFF.value,
    "batch_norm": True,
    "bias": False,
    "epochs": 60,
    "lr": 0.01,
    "step_sizes": [20, 40],
    "gamma_lr" : 0.1,
    "scale_lr_step_sizes": None,
    "scale_lr_gamma" : 0.1,
    "scale_lr_annealing_min" : None,
    "scale_lr": 0.001,
    "weight_decay": 0,
    "scale_warmup_epochs": 10,
    "size_dataset": 10000,
    "nr_workers": 8,
    "batch_size": 64,
    "seed": np.random.randint(0, 5),
    "basis_type": "hermite_a",
    "pooling_type": "max",
    "pooling_settings": None,
    "sweep": False,
    "acc_scale": True,
    "cutoff": None,
    "normalize_per_scale": False,
    "reuse_train_mean_std" : False,
    "max_pool_output": 2,
    "max_order": 4,
    "basis_mult": 1.4,
    "basis_min_scale": 1.5,
    "project": "scale_learning",
    "wandb_dir": "../wandb",
}


def get_resp_kernel_sizes(conv_scales, base_kernel_size, basis_min_scale, kernel_size_init_k):
    if kernel_size_init_k is not None:
        kernel_sizes = [
            2 * math.ceil(kernel_size_init_k * scale_temp * basis_min_scale) + 1 for scale_temp in conv_scales
        ]
    else:
        kernel_sizes = [
            int(round(base_kernel_size * scales_temp) // 2 * 2 + 1)
            for scales_temp in conv_scales
        ]
    return kernel_sizes


def update_dist_params(dist_params, cutoff, resize_factor):
    # Update dist_params, necessary since we define uniform distributions with boundaries while scipy uses loc and scale
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
            print("Not implemented yet!")
            print(
                "BUT in theory, we just divide the scale parameter \
                    by the resize factor and it should also work!"
            )
    else:
        if dist_params[0] == "UNIFORM":
            dist_params[2] = round(dist_params[2] - dist_params[1], 2)
    return dist_params, cutoff


def create_MNIST_TOY(
    sample_scales,
    reference_scale,
    resize_factor,
    img_size,
    obj_size,
    cutoff,
    seed,
    in_depth=False,
    normalize_per_scale=False,
    reuse_train_mean_std=False,
    **kwargs,
):
    if img_size != 28:
        root = "../Data/MNIST_single_img_size_{:.0f}_val_5000/seed_{}/scale_1.0_1.0/".format(
            img_size, seed
        )
    else:
        root = "../Data/MNIST_single_val_5000/seed_{}/scale_1.0_1.0/".format(seed)
        if resize_factor != 2:
            root = "../Data/MNIST_single_val_5000_{:.1f}/seed_{}/scale_1.0_1.0/".format(
                resize_factor, seed
            )
        img_size = int(img_size * resize_factor)

    if sample_scales[0] == "DISCRETE":
        discrete = True
        sample_scales_t = sample_scales[1:].copy()
        # Convert sample_scales (which are in terms of ISR) to actual scales (in terms of reference_scale)
        scales_MNIST = [scale / reference_scale for scale in sample_scales_t]

        print("Scales Mnist - ", scales_MNIST)
        train_dataset = loaders.MnistMultiScale(
            os.path.join(root, "train"),
            scales_used=scales_MNIST,
            discrete=discrete,
            seed=seed,
            need_scale_labels=True,
            img_size=img_size,
            obj_size=obj_size,
            fashion=False,
            cutoff=cutoff,
        )

        val_dataset = loaders.MnistMultiScale(
            os.path.join(root, "val"),
            scales_used=scales_MNIST,
            discrete=discrete,
            seed=seed,
            need_scale_labels=True,
            img_size=img_size,
            obj_size=obj_size,
            fashion=False,
            cutoff=cutoff,
        )

        test_dataset = loaders.MnistMultiScale(
            os.path.join(root, "test"),
            scales_used=scales_MNIST,
            discrete=discrete,
            seed=seed,
            need_scale_labels=True,
            img_size=img_size,
            obj_size=obj_size,
            fashion=False,
            cutoff=cutoff,
        )
    else:
        sample_scales_t, cutoff = update_dist_params(
            sample_scales.copy(), cutoff, resize_factor
        )

        dist_name = sample_scales_t[0]
        loc = sample_scales_t[1]
        scale_param = sample_scales_t[2]

        if in_depth:
            train_dataset = None
            val_dataset = None
            test_dataset = loaders.MnistMultiScale(
                os.path.join(root, "test"),
                scales_used=None,
                discrete=False,
                seed=seed,
                dist_name=dist_name,
                loc=loc,
                scale=scale_param,
                cutoff=cutoff,
                need_scale_labels=True,
                img_size=img_size,
                obj_size=obj_size,
                fashion=False,
                in_depth=in_depth,
                normalize_per_scale=normalize_per_scale,
                reuse_train_mean_std=reuse_train_mean_std,
            )
        else:
            train_dataset = loaders.MnistMultiScale(
                os.path.join(root, "train"),
                scales_used=None,
                discrete=False,
                seed=seed,
                dist_name=dist_name,
                loc=loc,
                scale=scale_param,
                cutoff=cutoff,
                need_scale_labels=True,
                img_size=img_size,
                obj_size=obj_size,
                fashion=False,
            )

            val_dataset = loaders.MnistMultiScale(
                os.path.join(root, "val"),
                scales_used=None,
                discrete=False,
                seed=seed,
                dist_name=dist_name,
                loc=loc,
                scale=scale_param,
                cutoff=cutoff,
                need_scale_labels=True,
                img_size=img_size,
                obj_size=obj_size,
                fashion=False,
            )

            test_dataset = loaders.MnistMultiScale(
                os.path.join(root, "test"),
                scales_used=None,
                discrete=False,
                seed=seed,
                dist_name=dist_name,
                loc=loc,
                scale=scale_param,
                cutoff=cutoff,
                need_scale_labels=True,
                img_size=img_size,
                obj_size=obj_size,
                fashion=False,
                in_depth=in_depth,
            )
        print("NR Scales: ", test_dataset.nr_scales)
    return train_dataset, val_dataset, test_dataset, test_dataset.img_size

def create_network(
    init_scales,
    layer_name,
    nr_filters,
    nr_layers,
    learnable_basis_min,
    learn_mode,
    basis_type,
    pooling_type,
    pooling_settings,
    nr_internal,
    img_size=56,
    in_channels=1,
    max_pool_output=2,
    batch_norm=True,
    bias=False,
    eff_size=7,
    base_kernel_size=7,
    largest_kernel_size=None,
    kernel_size_init_k = 3, 
    max_order=3,
    basis_mult=1.4,
    basis_min_scale=1.5,
    linear_hidden=None,
    linear_dropout=None,
    simplified=False,
    simple_dropout=0,
    **kwargs,
):
    layers = []
    # Load dictionary for SESN/Disco Parameters
    default_kwargs = {}
    default_kwargs["basis_mult"] = basis_mult
    default_kwargs["basis_max_order"] = max_order
    learn_mode = LearnMode(learn_mode)  # Override learn_mode
    learnable = learn_mode != LearnMode.OFF
    if learnable:
        # Override basis_min_scale if necessary
        scales_parameter, basis_min_scale = calculate_scales_parameter(
            ConvType.SESN,
            learn_mode,
            nr_internal,
            learnable_basis_min,
            init_scales,
            basis_min_scale,
            decoupled_basis_min=False,
            nr_layers=1,
            **default_kwargs,
        )
        if largest_kernel_size is None:
            print("Kernel Size is now learnable")            
        # assert largest_kernel_size is not None
    else:
        # If init_scales is int, we calculate the scales based on the ISR
        if isinstance(init_scales, float) or isinstance(init_scales, int):
            if nr_internal > 1:
                init_scales = [
                    init_scales ** (scale / (nr_internal - 1))
                    for scale in range(nr_internal)
                ]
            else:
                init_scales = [init_scales]
        # if not learnable need to multiply with basis_min_scale before passing to SESN
        if layer_name == "SESN":
            scales_parameter = nn.Parameter(
                torch.tensor(init_scales) * basis_min_scale,
                requires_grad=False,
            )
        else:
            scales_parameter = nn.Parameter(
                torch.tensor(init_scales), requires_grad=False
            )
        kernel_sizes = get_resp_kernel_sizes(init_scales, base_kernel_size, basis_min_scale, kernel_size_init_k)

        largest_kernel_size = (
            kernel_sizes[-1] if largest_kernel_size is None else largest_kernel_size
        )
    if nr_layers == 1:
        nr_filters = [nr_filters]

    for i in range(nr_layers):
        if i != 0:
            layers.extend(
                [
                    nn.ReLU(),
                    nn.MaxPool3d([1, 2, 2], stride=[1, 2, 2]),
                    nn.BatchNorm3d(nr_filters[i - 1]),
                ]
            )

        if not learnable:
            if layer_name == "SESN":
                if i > 0:
                    layer = SESConv_H_H(
                        nr_filters[i - 1],
                        nr_filters[i],
                        1,
                        kernel_size=largest_kernel_size,
                        effective_size=eff_size,
                        scales=scales_parameter.detach().cpu().tolist(),
                        padding=largest_kernel_size // 2,
                        bias=bias,
                        basis_type=basis_type,
                        basis_min_scale=basis_min_scale,
                        **default_kwargs,
                    )
                else:
                    layer = SESConv_Z2_H(
                        in_channels,
                        nr_filters[i],
                        kernel_size=largest_kernel_size,
                        effective_size=eff_size,
                        scales=scales_parameter.detach().cpu().tolist(),
                        padding=largest_kernel_size // 2,
                        bias=bias,
                        basis_type=basis_type,
                        basis_min_scale=basis_min_scale,
                        **default_kwargs,
                    )
        else:
            if layer_name == "SESN":
                if i > 0:
                    layer = SESConv_H_H_Learnable(
                        nr_filters[i - 1],
                        nr_filters[i],
                        1,
                        scales_parameter,
                        learn_mode=learn_mode,
                        effective_size=eff_size,
                        nr_internal_scales=nr_internal,
                        bias=bias,
                        basis_type=basis_type,
                        largest_kernel_size=largest_kernel_size,
                        init_k = kernel_size_init_k, 
                        basis_min_scale=basis_min_scale,
                        **default_kwargs,
                    )
                else:
                    layer = SESConv_Z2_H_Learnable(
                        in_channels,
                        nr_filters[i],
                        scales_parameter,
                        learn_mode=learn_mode,
                        effective_size=eff_size,
                        nr_internal_scales=nr_internal,
                        bias=bias,
                        basis_type=basis_type,
                        largest_kernel_size=largest_kernel_size,
                        init_k = kernel_size_init_k, 
                        basis_min_scale=basis_min_scale,
                        **default_kwargs,
                    )

        layers.append(layer)

    # Scale Max Projection
    layers.append(SESMaxProjection())

    layers.append(nn.ReLU())

    if batch_norm:
        layers.append(nn.BatchNorm2d(nr_filters[-1]))

    # Finish off network with maxpooling layer
    if pooling_settings is not None:
        if pooling_type == "max":
            pool_layer = nn.MaxPool2d
        elif pooling_type == "avg":
            pool_layer = nn.AvgPool2d
        else:
            raise NotImplementedError
        max_pool_size = pooling_settings[0]
        stride = pooling_settings[1]
        max_pool_output = (img_size - max_pool_size) // stride + 1
        layers.append(pool_layer(max_pool_size, stride=stride))
    else:
        if pooling_type == "max":
            pool_layer = nn.AdaptiveMaxPool2d
        elif pooling_type == "avg":
            pool_layer = nn.AdaptiveAvgPool2d
        layers.append(pool_layer((max_pool_output, max_pool_output)))

    layers.append(nn.Flatten())
    if linear_hidden is not None and not simplified:
        layers.append(
            nn.Linear(
                nr_filters[-1] * (max_pool_output**2), linear_hidden, bias=False
            )
        )
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm2d(linear_hidden))
        if linear_dropout is not None:
            layers.append(nn.Dropout(linear_dropout))
    if simplified and simple_dropout != 0:
        layers.append(nn.Dropout(simple_dropout))
    layers.append(nn.Linear(nr_filters[-1] * (max_pool_output**2), 10))
    return nn.Sequential(*layers)


class Runner(pl.LightningModule):
    def __init__(self, model, cfg):
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.learnable = cfg.learnable
        self.lr = cfg.lr
        self.scale_lr = cfg.scale_lr
        self.loss_fn = nn.CrossEntropyLoss()

        self.train_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=10)
        self.val_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=10)
        self.test_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=10)

        if self.cfg.acc_scale:
            self.test_accuracy_scale = torchmetrics.Accuracy(
                task="multiclass", num_classes=self.cfg.nr_scales + 1, average="none"
            )

        wandb.define_metric("val/acc", summary="max")

    def on_fit_start(self) -> None:
        print(self.model)
        if self.learnable:
            current_scales = self.model[0].calculate_scales().detach().cpu().tolist()
        else:
            if self.cfg.layer_name == "SESN":
                current_scales = self.model[0].scales
        if isinstance(current_scales, list) and len(current_scales) > 1:
            wandb.config.update(
                {"ISR_start": round(current_scales[-1] / current_scales[0], 3)},
                allow_val_change=True,
            )
        return super().on_fit_start()

    def forward(self, x):
        # Runner needs to redirect any model.forward() calls to the actual
        # network
        return self.model(x)

    def configure_optimizers(self):
        if not self.learnable:
            optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self.lr, weight_decay=self.cfg.weight_decay
            )
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=self.cfg.step_sizes, gamma=self.cfg.gamma_lr
            )
            return [optimizer], [lr_scheduler]
        else:
            params = []
            # Since the first layer is the convolution!
            learn_params = [self.model[0].scales_param]
            if self.cfg.learnable_basis_min:
                learn_params.append(self.model[0].basis_min_param)
            other_params = list(set(self.model.parameters()) - set(learn_params))
            params.append(
                {"params": learn_params, "lr": self.scale_lr, "name": "Scale LR"}
            )
            params.append({"params": other_params, "name": "Main LR"})
            optimizer = torch.optim.Adam(params, lr=self.lr)

            nr_of_steps_p_epoch = math.ceil(
                self.cfg.size_dataset / float(self.cfg.batch_size)
            )

            # Learning rate Scheduler for main LR : Need to take into accoutn warmup
            step_sizes = self.cfg.step_sizes.copy()
            step_sizes = [step_size * nr_of_steps_p_epoch for step_size in step_sizes]

            scale_lr_scheduler = None
            if self.cfg.scale_warmup_epochs > 0:
                warmup_steps = self.cfg.scale_warmup_epochs * nr_of_steps_p_epoch
                # Only use warmup scheduler for scale_LR
                lambda_learn = lambda iter: min(iter / warmup_steps, 1)
                lambda_stock = lambda iter: 1

                scale_lr_scheduler = custom_LambdaLR(
                    optimizer,
                    [lambda_learn, lambda_stock],
                    warmup_steps,
                    excluded_group_name="Main LR",
                )

            if self.cfg.scale_lr_step_sizes:
                step_lr_scheduler = MultiStepGroupSpecific(
                    optimizer,
                    [
                        (scale_step_size - self.cfg.scale_warmup_epochs)
                        * nr_of_steps_p_epoch for scale_step_size in self.cfg.scale_lr_step_sizes
                    ],
                    excluded_group_name="Main LR",
                    gamma=self.cfg.scale_lr_gamma,
                )
                if scale_lr_scheduler != None:
                    scale_lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
                        optimizer,
                        schedulers=[scale_lr_scheduler, step_lr_scheduler],
                        milestones=[self.cfg.scale_warmup_epochs * nr_of_steps_p_epoch],
                    )
                else:
                    scale_lr_scheduler = step_lr_scheduler

            elif self.cfg.scale_lr_annealing_min:
                T_max = (self.cfg.epochs - self.cfg.scale_warmup_epochs) * math.ceil(
                    nr_of_steps_p_epoch
                )  # - warmup epochs
                annealing_scheduler = custom_CosineAnnealingLR(
                    optimizer,
                    T_max=T_max,
                    excluded_group_name= "Main LR",
                    eta_min=self.cfg.scale_lr_annealing_min,
                )
            
                if scale_lr_scheduler is not None:
                    scale_lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
                        optimizer,
                        schedulers=[scale_lr_scheduler, annealing_scheduler],
                        milestones=[self.cfg.scale_warmup_epochs * nr_of_steps_p_epoch],
                    )
                else:
                    scale_lr_scheduler = annealing_scheduler
            
            if scale_lr_scheduler is not None:
                if self.cfg.scale_lr_annealing_min is not None or self.cfg.scale_lr_step_sizes is not None:
                    step_scheduler_main = MultiStepGroupSpecific(
                        optimizer,
                        # [step_size - warmup_steps for step_size in step_sizes],
                        step_sizes,
                        excluded_group_name="Scale LR",
                        gamma=self.cfg.gamma_lr,
                    )
                else:
                    step_scheduler_main = torch.optim.lr_scheduler.MultiStepLR(
                        optimizer,
                        step_sizes,
                        gamma=self.cfg.gamma_lr
                    )

                lr_scheduler = torch.optim.lr_scheduler.ChainedScheduler(
                    [scale_lr_scheduler, step_scheduler_main]
                )

                # lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
                #     optimizer,
                #     schedulers=[scale_lr_scheduler, step_scheduler_main],
                #     milestones=[warmup_steps],
                # )
            else:
                lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                        optimizer,
                        step_sizes,
                        gamma=self.cfg.gamma_lr
                    )

            return [optimizer], [{"scheduler": lr_scheduler, "interval": "step"}]

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step()
        
    def _step(self, batch):
        y_hat = self.model(batch[0])
        loss = self.loss_fn(y_hat, batch[1])
        return loss, y_hat

    def training_step(self, batch, _):
        loss, y_hat = self._step(batch)
        preds = torch.argmax(y_hat, dim=1)
        self.train_accuracy(preds, batch[1])

        # Log internal scales
        if self.learnable:
            scales = self.model[0].calculate_scales().detach().cpu().tolist()
            index = 0
            for scale in scales:
                self.log(f"Scale_{index}", scale, sync_dist=True, prog_bar=True)
                index += 1
            if (
                self.cfg.learn_mode == LearnMode.SINGLE_RATIO.value
                or self.cfg.learn_mode == LearnMode.SINGLE_RATIO_SQUARED.value
            ):
                self.log("ISR", round(scales[-1] / scales[0], 3), sync_dist=True)
                scale_param = self.model[0].scales_param.detach().cpu().tolist()
                self.log(f"Scale Parameter", scale_param, sync_dist=True)

            # scale_params = self.model[0].scales_param.detach().cpu().tolist()
            # if isinstance(scale_params, list):
            #     for i, scale_param in enumerate(scale_params):
            #         self.log(f"ISR_{index}", scale_param, sync_dist=True)
            # else:
            #     self.log(f"ISR", scale_params, sync_dist=True, prog_bar=True)

        # Log step-level loss & accuracy
        self.log("train/loss_step", loss)
        self.log("train/acc", self.train_accuracy, on_epoch=True)
        return loss

    def validation_step(self, batch, _):
        loss, y_hat = self._step(batch)
        preds = torch.argmax(y_hat, dim=1)
        self.val_accuracy(preds, batch[1])

        # Log step-level loss & accuracy
        self.log("val/loss", loss, sync_dist=True)
        self.log("val/acc", self.val_accuracy, on_epoch=True)
        return loss

    def test_step(self, batch, _):
        loss, y_hat = self._step(batch)
        preds = torch.argmax(y_hat, dim=1)
        self.test_accuracy(preds, batch[1])

        if self.cfg.acc_scale:
            scales_batch = batch[2] + 1
            self.test_accuracy_scale(
                scales_batch * preds.eq(batch[1].view_as(preds)), scales_batch
            )
            # self.log('test/acc_scale', self.test_accuracy_scale, on_epoch=True)

        # Log test loss
        self.log("test/loss", loss)
        self.log("test/acc", self.test_accuracy, on_epoch=True)
        return loss

    # def on_train_epoch_end(self):
    #     # Log the epoch-level training accuracy
    #     self.log('train/acc', self.train_accuracy)
    #     self.train_accuracy.reset()

    # def on_validation_epoch_end(self):
    #     # Log the epoch-level validation accuracy
    #     self.log('val/acc', self.val_accuracy)
    #     self.val_accuracy.reset()

    def on_test_epoch_end(self) -> None:
        if self.learnable:
            current_scales = self.model[0].calculate_scales().detach().cpu().tolist()
            scale_param = self.model[0].scales_param.detach().cpu().tolist()
            if isinstance(scale_param, list):
                for i, scale_p in enumerate(scale_param):
                    wandb.config.update(
                        {f"Scale Parameter {i}": round(scale_p, 3)},
                        allow_val_change=True,
                    )
            else:
                wandb.config.update(
                    {"Scale Parameter": round(scale_param, 3)},
                    allow_val_change=True,
                )
        else:
            if self.cfg.layer_name == "SESN":
                current_scales = self.model[0].scales
        wandb.config.update(
            {"Final Conv Scales": current_scales}, allow_val_change=True
        )
        if isinstance(current_scales, list) and len(current_scales) > 1:
            wandb.config.update(
                {"Final ISR": round(current_scales[-1] / current_scales[0], 3)},
                allow_val_change=True,
            )
            wandb.config.update(
                {"Final Basis Min Scale": round(current_scales[0], 3)},
                allow_val_change=True,
            )
        if self.cfg.acc_scale:
            res = self.test_accuracy_scale.compute()[1:]
            scale_ind = 0
            for el in res:
                self.log(f"test/acc scale {scale_ind}", el, sync_dist=True)

                scale_ind += 1
            self.test_accuracy_scale.reset()

        return super().on_test_epoch_end()

def run_experiment(cfg):
    config = DEFAULT_CONFIG.copy()
    # Override default config with any new values
    for key, value in cfg.items():
        config[key] = value

    if config["reference_scale"] is None and config["sample_scales"][0] == "DISCRETE":
        config["reference_scale"] = config["sample_scales"][-1]
    if config["sample_scales"][0] != "DISCRETE":
        print(
            "WARNING: Continuous distributions need to be expressed in terms of original MNIST SIZE"
        )
        assert config["cutoff"] is not None

    config["seed"] = pl.seed_everything(config["seed"], workers=True)
    os.environ["WANDB_CACHE_DIR"] = os.path.join(config["wandb_dir"], "cache")

    # Create dataset
    train_dataset, val_dataset, test_dataset, _ = create_MNIST_TOY(**config)

    # Create network
    network = create_network(**config)
    config["nr_scales"] = test_dataset.nr_scales
    # Train
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        num_workers=config["nr_workers"],
        batch_size=config["batch_size"],
        shuffle=True,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        num_workers=config["nr_workers"],
        batch_size=config["batch_size"],
        shuffle=False,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        num_workers=config["nr_workers"],
        batch_size=config["batch_size"],
        shuffle=False,
    )

    cfg = OmegaConf.create(config)


    if config["sweep"]:
        wandb.config.update(config)
    wandb_logger = WandbLogger(
        save_dir=cfg.wandb_dir,
        project=cfg.project,
        tags=["MNIST", config["exp_name"]],
        offline=False,
        log_model=True,
        # Keyword args passed to wandb.init()
        entity="mbasting",
        config=config,
    )

    checkpoint_callback_class = ModelCheckpoint(
        filename='epoch={epoch}-val_acc={val/acc:.2f}', 
        auto_insert_metric_name=False,
        save_top_k=1,  # save only the best ckpt
        monitor="val/acc",
        mode="max",
    )
    call_backs = [
        checkpoint_callback_class,
        LearningRateMonitor(logging_interval="step"),
    ]
    

    # Tie it all together with PyTorch Lightning: Runner contains the model,
    # optimizer, loss function and metrics; Trainer executes the
    # training/validation loops and model checkpointing.
    runner = Runner(network, cfg)

    # profiler = AdvancedProfiler(dirpath=".", filename="perf_logs")
    trainer = pl.Trainer(
        max_epochs=config["epochs"],
        # Use DDP training by default, even for CPU training
        strategy="ddp_find_unused_parameters_false",
        callbacks=call_backs,
        logger=wandb_logger,
        benchmark=True,
        devices=torch.cuda.device_count(),
        # profiler=profiler
    )
    # Train + validate (if validation dataset is implemented)
    trainer.fit(runner, train_loader, val_loader)

    if not config["sweep"]:
        # Test
        if config["learnable"]:
            runner = Runner.load_from_checkpoint(
                checkpoint_callback_class.best_model_path,
                model=network,
                cfg=cfg,
                strict=False,
            )
            trainer.test(runner, test_loader)
        else:
            trainer.test(
                runner, test_loader, ckpt_path=checkpoint_callback_class.best_model_path
            )
    else:
        # Test on Val dataloader
        if config["learnable"]:
            runner = Runner.load_from_checkpoint(
                checkpoint_callback_class.best_model_path,
                model=network,
                cfg=cfg,
                strict=False,
            )
            trainer.test(runner, val_loader)
        else:
            trainer.test(
                runner, val_loader, ckpt_path=checkpoint_callback_class.best_model_path
            )

    wandb.finish(0)


def evaluate_model(cmd_cfg, test_dataloader=None, test_seed=None):
    # Test dataloader and test seed are included to speed up testing of multiple models by reusing the same test set
    config = DEFAULT_CONFIG.copy()
    def_cfg = OmegaConf.create(config)

    # Load artifact
    api = wandb.Api()
    run = api.run(f"{cmd_cfg.entity}/{cmd_cfg.project}/{cmd_cfg.run_id}")
    cfg = OmegaConf.merge(run.config, cmd_cfg)
    # Override default config with any new values
    cfg = OmegaConf.merge(def_cfg, cfg)

    os.environ["WANDB_CACHE_DIR"] = os.path.join(cfg.wandb_dir, "cache")

    run = wandb.init(entity=cfg.entity, project=cfg.project, tags=cfg.tags)

    # download checkpoint locally (if not already cached)
    if not cfg.cluster:
        artifact = wandb.use_artifact(
            f"{cfg.entity}/{cfg.artifact_project}/model-{cfg.run_id}:best_k", type="model"
        )
        artifact_dir = artifact.download()
    else:
        artifact_dir = f"artifacts/model-{cfg.run_id}:v0"
    # Load dataloader if not loaded yet or test seed is different
    if test_dataloader is None or test_seed != cfg["seed"]:
        _, _, test_dataset, _ = create_MNIST_TOY(**cfg)
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            num_workers=cfg["nr_workers"],
            batch_size=cfg["batch_size"],
            shuffle=False,
        )
        test_seed = cfg["seed"]
        cfg.min_scale = test_dataset.cutoff[0]
        cfg.max_scale = test_dataset.cutoff[1]

    # Needs to be overridden to calculate correct statistics
    cfg.nr_scales = test_dataloader.dataset.nr_scales

    config = OmegaConf.to_object(cfg)

    # Create network (no weights)
    network = create_network(**config)

    wandb.config.update(config)

    wandb_logger = WandbLogger(
        save_dir=cfg.wandb_dir,
        project=cfg.project,
        tags=cfg.tags,
        offline=False,
        # log_model=True,
        # Keyword args passed to wandb.init()
        entity="mbasting",
        config=config,
    )

    # Load Empty runner
    runner = Runner.load_from_checkpoint(checkpoint_path=Path(artifact_dir) / "model.ckpt", model=network, cfg=cfg, strict=True)

    # profiler = AdvancedProfiler(dirpath=".", filename="perf_logs")
    trainer = pl.Trainer(
        # Use DDP training by default, even for CPU training
        strategy="ddp_find_unused_parameters_false",
        callbacks=[],
        logger=wandb_logger,
        benchmark=True,
        devices=torch.cuda.device_count(),
        # profiler=profiler
    )

    trainer.test(runner, test_dataloader)
    wandb.finish(0)
    return test_dataloader, test_seed


def check(cmd_cfg):
    if "sweep" in cmd_cfg.keys() and cmd_cfg.sweep:
        # need parameters sweep_id, count, project, all other parameters are saved in wandb
        wandb.agent(
            cmd_cfg.sweep_id,
            run_sweep,
            count=cmd_cfg.count,
            entity="mbasting",
            project="scale_learning",
        )
    else:
        run_experiment(cmd_cfg)


def run_sweep():
    with wandb.init(
        config=wandb.config,
        tags=["Sweep"],
        dir="../wandb",
    ):
        # Load defaults (overwrite for sweep only done through wandb config)
        run_experiment(wandb.config)


if __name__ == "__main__":
    CMD_CFG = OmegaConf.from_cli()
    check(CMD_CFG)
