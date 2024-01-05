"""

"""
import logging

# Package imports, from conda or pip
import os
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
import wandb
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar,
)
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins.environments import SLURMEnvironment


logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)

# Relative Imports
import dataset_factory
import model_factory
from optimizer import (
    construct_optimizer_MNIST_SCALE,
    construct_optimizer_STL_10,
    construct_scheduler_MNIST_SCALE,
    construct_scheduler_STL_10,
)
from utils.model_utils import ConvType, LearnMode

# os.environ["WANDB_SILENT"] = "true"
CFG_DEFAULT_GENERAL = OmegaConf.load("configs/main_config.yaml")
CFG_MNIST_DEFAULT = OmegaConf.load("configs/default_train_MNISTS.yaml")
CFG_STL_10_DEFAULT = OmegaConf.load("configs/default_train_STL_10.yaml")
CFG_CIFAR_10_DEFAULT = OmegaConf.load("configs/default_train_CIFAR_10.yaml")

class Runner(pl.LightningModule):
    def __init__(self, cfg, model):
        super().__init__()
        self.cfg = cfg
        self.model = model
        # self.test_best_scale = False
        self.loss_fn_class = nn.CrossEntropyLoss()
        self.loss_fn_scale = nn.CrossEntropyLoss()

        self.train_accuracy_class = torchmetrics.Accuracy(
            task="multiclass", num_classes=cfg.dataset.nr_classes
        )
        self.val_accuracy_class = torchmetrics.Accuracy(
            task="multiclass", num_classes=cfg.dataset.nr_classes
        )
        self.test_accuracy_class = torchmetrics.Accuracy(
            task="multiclass", num_classes=cfg.dataset.nr_classes
        )
        if cfg.dataset.calc_acc_per_scale:
            self.train_accuracy_scale = torchmetrics.Accuracy(
                task="multiclass", num_classes=cfg.dataset.nr_scales
            )
            self.val_accuracy_scale = torchmetrics.Accuracy(
                task="multiclass", num_classes=cfg.dataset.nr_scales
            )

            # If we perform some sort of scale generalization experiment
            # we need accuracy per scale as well!
            self.test_accuracy_scale = torchmetrics.Accuracy(
                task="multiclass",
                num_classes=cfg.dataset.nr_scales + 1,
                average="none" if cfg.dataset.calc_acc_per_scale else "micro",
            )

    def on_fit_start(self) -> None:
        print(self.model)
        if self.cfg.model.scale_learn_mode != LearnMode.OFF.value:
            current_scales = self.model.get_scales()
            # Add more helpful config settings
            if self.global_rank == 0:
                wandb.config.update(
                    {
                        "dataset_name": dataset_factory.ChoiceDataset(
                            self.cfg.dataset.d_index
                        ).name
                    }
                )
                wandb.config.update({"model_name": ConvType(self.cfg.model.conv_index).name})
                wandb.config.update(
                    {
                        "nr_parameters": sum(
                            p.numel() for p in self.model.parameters() if p.requires_grad
                        )
                    }
                )
                wandb.config.update({"Conv Scales": current_scales})

                if isinstance(current_scales, list) and len(current_scales) > 1:
                    wandb.config.update(
                        {"ISR_start": round(current_scales[-1] / current_scales[0], 3)},
                        allow_val_change=True,
                    )

            print("SCALE USED IN CONVOLUTION (START):", current_scales)
        return super().on_fit_start()

    def forward(self, x):
        # Runner needs to redirect any model.forward() calls to the actual
        # network
        return self.model(x)

    def configure_optimizers(self):
        if self.cfg.dataset.d_index == dataset_factory.ChoiceDataset.STL_10.value or self.cfg.dataset.d_index == dataset_factory.ChoiceDataset.CIFAR_10.value:
            optimizer = construct_optimizer_STL_10(self.model, self.cfg)
            lr_scheduler = construct_scheduler_STL_10(self.model, optimizer, self.cfg)
        else:
            optimizer = construct_optimizer_MNIST_SCALE(self.model, self.cfg)
            lr_scheduler = construct_scheduler_MNIST_SCALE(
                self.model, optimizer, self.cfg
            )
        return [optimizer], lr_scheduler

    def _step(self, batch):
        out = self.model(batch[0])
        out_class = out[0]
        loss_class = self.loss_fn_class(out_class, batch[1])

        return loss_class, out_class

    def training_step(self, batch, _):
        loss_class, out_class = self._step(batch)

        preds_class = torch.argmax(out_class, dim=1)
        self.train_accuracy_class(preds_class, batch[1])

        
        # Log internal scales
        if self.cfg.model.scale_learn_mode != LearnMode.OFF.value:
            self.model.log_scales(self.log)

        # Log step-level loss & accuracy
        self.log("Train/Loss_step", loss_class, sync_dist=True)
        self.log(
            "Train/Accuracy",
            self.train_accuracy_class,
            sync_dist=True,
            on_step=True,
            on_epoch=True,
        )

        return loss_class

    def validation_step(self, batch, _):
        loss_class, out_class = self._step(batch)

        preds_class = torch.argmax(out_class, dim=1)
        self.val_accuracy_class(preds_class, batch[1])

        # Log step-level loss & accuracy
        self.log("Val/Loss_step", loss_class, sync_dist=True)
        self.log(
            "Val/Accuracy",
            self.val_accuracy_class,
            sync_dist=True,
            on_step=False,
            on_epoch=True,
        )
        return loss_class

    def test_step(self, batch, _):
        loss_class, out_class = self._step(batch)

        preds_class = torch.argmax(out_class, dim=1)
        self.test_accuracy_class.update(preds_class, batch[1])
        # self.test_accuracy_class(preds_class, batch[1])
        if self.cfg.dataset.calc_acc_per_scale:
            scales_batch = batch[2] + 1
            self.test_accuracy_scale.update(
                scales_batch * preds_class.eq(batch[1].view_as(preds_class)),
                scales_batch,
            )

        # Log step-level loss & accuracy
        self.log("Test/Loss_step", loss_class, sync_dist=True)

        return loss_class

    def on_test_epoch_end(self):
        name_class_acc = "Test/Accuracy"
        name_class_error = "Test/Error"
        if self.cfg.model.scale_learn_mode != LearnMode.OFF.value:
            current_scales = self.model.get_scales()
            if self.global_rank == 0:
                wandb.config.update(
                    {"Final Conv Scales": current_scales}, allow_val_change=True
                )
                if isinstance(current_scales, list) and len(current_scales) > 1:
                    wandb.config.update(
                        {"Final ISR": round(current_scales[-1] / current_scales[0], 3)},
                        allow_val_change=True,
                    )

        # Log the epoch-level validation accuracy
        result_class = self.test_accuracy_class.compute()
        self.log(name_class_acc, result_class, sync_dist=True)
        self.log(name_class_error, (1 - result_class) * 100)
        self.test_accuracy_class.reset()

        if self.cfg.dataset.calc_acc_per_scale:
            # Split of first scale since not used
            scale_ind = 0
            res = self.test_accuracy_scale.compute()[1:]
            for el in res:
                self.log(f"Test/Accuracy Scale {scale_ind}", el, sync_dist=True)
                self.log(
                    f"Test/Error Scale {scale_ind}",
                    (1 - el) * 100,
                    sync_dist=True,
                )

                scale_ind += 1
            self.test_accuracy_scale.reset()


def unflatten_dot(dictionary):
    # Unflatten config dictionary code
    # by jphdotam: https://github.com/wandb/wandb/issues/982#issuecomment-1167612861
    resultDict = dict()
    for key, value in dictionary.items():
        parts = key.split(".")
        d = resultDict
        for part in parts[:-1]:
            if part not in d:
                d[part] = dict()
            d = d[part]
        d[parts[-1]] = value
    return resultDict


def check(cmd_cfg, test_dataset_loader=None, test_seed=None):
    if "sweep" in cmd_cfg.keys() and cmd_cfg.sweep:
        wandb.agent(
            cmd_cfg.sweep_id,
            prepare_and_run_exp,
            count=cmd_cfg.count,
            entity="mbasting",
            project=cmd_cfg.wandb.project,
        )
    elif "testing_mode" in cmd_cfg.keys() and cmd_cfg.sweep:
        # Check which dataset we are dealing with
        # Load API and run configuration
        api = wandb.Api()
        run = api.run(
            f"{CFG_DEFAULT_GENERAL.wandb.entity}/{cmd_cfg.wandb.project}/{cmd_cfg.run_id}"
        )
        if run.config.dataset.d_index == dataset_factory.ChoiceDataset.STL_10.value:
            cfg_local = OmegaConf.merge(CFG_DEFAULT_GENERAL, CFG_STL_10_DEFAULT)
        elif run.config.dataset.d_index == dataset_factory.ChoiceDataset.CIFAR_10.value:
            cfg_local = OmegaConf.merge(CFG_DEFAULT_GENERAL, CFG_CIFAR_10_DEFAULT)
        else:
            cfg_local = OmegaConf.merge(CFG_DEFAULT_GENERAL, CFG_MNIST_DEFAULT)

        cfg = OmegaConf.merge(cfg_local, run.config)

        # Note that we can still overwrite some settings that we load using the API
        cfg = OmegaConf.merge(cfg, cmd_cfg)
        return evaluate(cfg, test_dataset_loader, test_seed)
    else:
        if cmd_cfg.dataset.d_index == dataset_factory.ChoiceDataset.STL_10.value:
            cfg_local = OmegaConf.merge(CFG_DEFAULT_GENERAL, CFG_STL_10_DEFAULT)
        elif cmd_cfg.dataset.d_index == dataset_factory.ChoiceDataset.CIFAR_10.value:
            cfg_local = OmegaConf.merge(CFG_DEFAULT_GENERAL, CFG_CIFAR_10_DEFAULT)
        else:
            cfg_local = OmegaConf.merge(CFG_DEFAULT_GENERAL, CFG_MNIST_DEFAULT)
        # Load defaults and overwrite by command-line arguments
        cfg = OmegaConf.merge(cfg_local, cmd_cfg)
        train_evaluate(cfg)


def prepare_and_run_exp():
    with wandb.init(
        config=wandb.config, tags=["Sweep"], dir=CFG_DEFAULT_GENERAL["wandb"]["dir"]
    ):
        # Load defaults (overwrite for sweep only done through wandb config)
        cfg_wandb = OmegaConf.create(unflatten_dot(wandb.config))
        if cfg_wandb.dataset.d_index == dataset_factory.ChoiceDataset.STL_10.value or cfg_wandb.dataset.d_index == dataset_factory.ChoiceDataset.CIFAR_10.value:
            cfg_local = OmegaConf.merge(CFG_DEFAULT_GENERAL, CFG_STL_10_DEFAULT)
        else:
            cfg_local = OmegaConf.merge(CFG_DEFAULT_GENERAL, CFG_MNIST_DEFAULT)

        conf = OmegaConf.merge(
            cfg_local, cfg_wandb
        )  # Second config replaces duplicates values in first


        train_evaluate(conf, sweep=True)


def train_evaluate(cfg, sweep=False):
    device = cfg.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Seed everything. Note that this does not make training entirely
    # deterministic.
    cfg.seed = pl.seed_everything(
        cfg.get("seed", np.random.randint(0, 6)), workers=True
    )
    cfg.precision = cfg.get("precision", 'highest')
    if cfg.precision == 'highest':
        precision = "32-true"
    else:
        precision = "bf16-mixed"
    torch.set_float32_matmul_precision(cfg.precision)
    # Create datasets using factory pattern
    (
        train_dataset_loader,
        val_dataset_loader,
        test_dataset_loader,
    ) = dataset_factory.factory(cfg)
    model = model_factory.factory(cfg)

    # Set cache dir to W&B logging directory
    os.environ["WANDB_CACHE_DIR"] = os.path.join(cfg.wandb.dir, "cache")
    config = OmegaConf.to_object(cfg)

    lr_monitor = LearningRateMonitor(logging_interval="step")

    checkpoint_callback_class = ModelCheckpoint(
        filename="epoch={epoch}-val_acc={Val/Accuracy:.3f}",  # the name of the best ckpt
        save_top_k=1,  # save only the best ckpt
        monitor="Val/Accuracy",  # monitor metric and save best
        mode="max",
    )

    wandb_logger = WandbLogger(
        save_dir=cfg.wandb.dir,
        project=cfg.wandb.project,
        tags=cfg.wandb.tags,
        log_model=True if not cfg.wandb.local else None,
        offline=cfg.wandb.local,
        # Keyword args passed to wandb.init()
        entity=cfg.wandb.entity,
        config=config,
    )
    if sweep:
        wandb.config.update(config)
    call_backs = [
        lr_monitor,
        checkpoint_callback_class,
        TQDMProgressBar(refresh_rate=cfg.train.refresh_rate_prog * cfg.train.accumulate_grad_batches),
    ]
    # Add Early Stopper
    if config["train"]["early_stop"]:
        early_stop_callback = EarlyStopping(
            monitor="Val/Accuracy",
            min_delta=0.00,
            patience=10,
            verbose=False,
            mode="max",
        )
        call_backs.append(early_stop_callback)

    # Tie it all together with PyTorch Lightning: Runner contains the model,
    # optimizer, loss function and metrics; Trainer executes the
    # training/validation loops and model checkpointing.
    print("NUMBER OF DEVICES: ", torch.cuda.device_count())
    runner = Runner(cfg, model)
    trainer = pl.Trainer(
        precision = precision,
        max_epochs=cfg.train.epochs,
        logger=wandb_logger,
        benchmark=True,
        callbacks=call_backs,
        enable_model_summary=True,
        accumulate_grad_batches=cfg.train.accumulate_grad_batches,
        check_val_every_n_epoch = cfg.train.check_val_every_n_epoch,
        # Use DDP training by default, even for CPU training
        strategy="ddp_find_unused_parameters_false",
        sync_batchnorm=True if torch.cuda.device_count() > 1 else False,
        devices=torch.cuda.device_count(),
    )

    # Train + validate (if validation dataset is implemented)
    trainer.fit(runner, train_dataset_loader, val_dataset_loader)

    if cfg.dataset.d_index == dataset_factory.ChoiceDataset.STL_10.value or cfg.dataset.d_index == dataset_factory.ChoiceDataset.CIFAR_10.value:
        test_dataset_loader = val_dataset_loader

    # Test (if test dataset is implemented)
    if test_dataset_loader is not None and not cfg.wandb.sweep:
        # If learnable scale need to reload model with best checkpoint since discrepancy can occur
        trainer.test(
            runner,
            test_dataset_loader,
            ckpt_path=checkpoint_callback_class.best_model_path,
        )
    else:
        # Get checkpoint of best model and update config to overwrite conv scales
        best = Runner.load_from_checkpoint(
            checkpoint_callback_class.best_model_path, model=model, cfg=cfg
        )
        if trainer.global_rank == 0 and cfg.model.scale_learn_mode != LearnMode.OFF.value:
            # Log best model scales + ISR
            current_scales = best.model.get_scales()
            wandb.config.update(
                {"Final Conv Scales": current_scales}, allow_val_change=True
            )
            if isinstance(current_scales, list) and len(current_scales) > 1:
                wandb.config.update(
                    {"Final ISR": round(current_scales[1] / current_scales[0], 3)},
                    allow_val_change=True,
                )

    wandb.finish(0)


def evaluate(cfg, test_dataset_loader=None, test_seed=None):
    # download checkpoint locally (if not already cached)
    if not cfg.cluster:
        artifact = wandb.use_artifact(
            f"{cfg.wandb.entity}/{cfg.wandb.project}/model-{cfg.run_id}:best_k",
            type="model",
        )
        artifact_dir = artifact.download()
    else:
        artifact_dir = f"artifacts/model-{cfg.run_id}:v0"

    device = cfg.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    # torch.set_float32_matmul_precision('high')

    # # Seed everything. Note that this does not make training entirely
    # # deterministic.

    # # Create datasets using factory pattern
    if test_dataset_loader is None or test_seed != cfg.seed:
        _, _, test_dataset_loader = dataset_factory.factory(cfg)
        test_seed = cfg.seed
    else:
        # Needs to be overridden to calculate correct statistics
        cfg.dataset.nr_scales = test_dataset_loader.dataset.nr_scales

    model = model_factory.factory(cfg)

    # # Set cache dir to W&B logging directory
    os.environ["WANDB_CACHE_DIR"] = os.path.join(cfg.wandb.dir, "cache")
    config = OmegaConf.to_object(cfg)

    wandb.config.update(config)

    wandb_logger = WandbLogger(
        save_dir=cfg.wandb.dir,
        project=cfg.wandb.project,
        tags=cfg.wandb.tags,
        log_model=True if not cfg.wandb.local else None,
        offline=cfg.wandb.local,
        # Keyword args passed to wandb.init()
        entity=cfg.wandb.entity,
        config=config,
    )

    call_backs = [TQDMProgressBar(refresh_rate=cfg.train.refresh_rate_prog)]

    # Load Empty runner
    runner = Runner(cfg, model)

    # Update weights from runner
    runner.load_state_dict(
        torch.load(Path(artifact_dir) / "model.ckpt")["state_dict"], strict=False
    )

    trainer = pl.Trainer(
        max_epochs=cfg.train.epochs,
        logger=wandb_logger,
        benchmark=True,
        callbacks=call_backs,
        enable_model_summary=False,
        devices=torch.cuda.device_count(),
        check_val_every_n_epoch = cfg.train.check_val_every_n_epoch,
        # Use DDP training by default, even for CPU training
        strategy="ddp_find_unused_parameters_false",
    )

    trainer.test(runner, test_dataset_loader)
    wandb.finish(0)
    return test_dataset_loader, test_seed


if __name__ == "__main__":
    CMD_CFG = OmegaConf.from_cli()
    check(CMD_CFG)
