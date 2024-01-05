
import math
import warnings

import torch
from omegaconf import OmegaConf

from utils.model_utils import ConvType, LearnMode


def construct_optimizer_STL_10(
        model: torch.nn.Module,
        cfg: OmegaConf
):  
    # DEFAULT CONFIGURATION OF SESN/DISCO
    if cfg.model.scale_learn_mode == LearnMode.OFF.value:
        parameters = filter(lambda x: x.requires_grad, model.parameters())
        optimizer = torch.optim.SGD(parameters, lr=cfg.train.learning_rate, 
                                    momentum=cfg.train.momentum, weight_decay=cfg.train.weight_decay, 
                                    nesterov=cfg.train.nesterov)
    else:
        # For Learnable scale need to have separate learning rate for scales
        # and other parameters
        params = []        
        # Since the first layer is the convolution!
        learn_params = [model.conv1.scales_param]
        if cfg.model.learnable_basis_min:
            # Add each basis min scale as a separate parameter
            if cfg.model.decoupled_basis_min:
                for i in range(len(model.basis_min_scale)):
                    learn_params.append(model.basis_min_scale[i])
            else:
                learn_params.append(model.basis_min_scale)


        other_params = list(set(model.parameters()) - set(learn_params))
        params.append(
            {"params": learn_params, "lr": cfg.train.scale_lr, "name": "Scale LR"}
        )
        params.append({"params": other_params, "name": "Main LR"})
        optimizer = torch.optim.SGD(params, lr=cfg.train.learning_rate, momentum=cfg.train.momentum,
                      weight_decay=cfg.train.weight_decay, nesterov=cfg.train.nesterov)
    return optimizer

def construct_optimizer_MNIST_SCALE(
        model: torch.nn.Module,
        cfg: OmegaConf
):
    if cfg.model.scale_learn_mode == LearnMode.OFF.value:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=cfg.train.learning_rate
        )
    else:
        params = []
        # Since the second layer is the convolution!
        # Since the first layer is the convolution!
        learn_params = [model.main.conv_1.scales_param]
        if cfg.model.learnable_basis_min:
            if cfg.model.decoupled_basis_min:
                for i in range(len(model.basis_min_scale)):
                    learn_params.append(model.basis_min_scale[i])
            else:
                learn_params.append(model.basis_min_scale)

        other_params = list(set(model.parameters()) - set(learn_params))
        params.append(
            {"params": learn_params, "lr": cfg.train.scale_lr, "name": "Scale LR"}
        )
        params.append({"params": other_params, "name": "Main LR"})
        optimizer = torch.optim.Adam(params, lr=cfg.train.learning_rate)

       
    return optimizer 

def separate_lr_scheduler_scale_learning(
        optimizer,
        cfg: OmegaConf,
):
    # Warmup is calculated in terms of number of steps instead of epochs
    # So need to convert lr_steps (in epoch) to in steps

    nr_of_steps_p_epoch = math.ceil(
                cfg.dataset.nr_samples / cfg.train.batch_size
            )
    # Learning rate Scheduler for main LR : Need to take into accoutn warmup
    step_sizes = [
        step_size * nr_of_steps_p_epoch for step_size in cfg.train.lr_steps
    ]
    scale_lr_scheduler = None
    if cfg.train.scale_warmup_epochs > 0:
        warmup_steps = cfg.train.scale_warmup_epochs * nr_of_steps_p_epoch
        # Only use warmup scheduler for scale_LR
        lambda_learn = lambda iter: min(
            iter / warmup_steps, 1
        )
        lambda_stock = lambda iter: 1

        scale_lr_scheduler = custom_LambdaLR(
            optimizer,
            [lambda_learn, lambda_stock],
            warmup_steps,
            excluded_group_name="Main LR",
        )
    #     # Update steps sizes based on warmup necessary    
    # step_scheduler_main = MultiStepGroupSpecific(
    #         optimizer, step_sizes, excluded_group_name="Scale LR", gamma=cfg.train.lr_gamma)
    
    if cfg.train.scale_lr_step_sizes:
        step_lr_scheduler = MultiStepGroupSpecific(
            optimizer,
            [
                (scale_step_size - cfg.train.scale_warmup_epochs)
                * nr_of_steps_p_epoch for scale_step_size in cfg.train.scale_lr_step_sizes
            ],
            excluded_group_name="Main LR",
            gamma=cfg.train.scale_lr_gamma,
        )
        if scale_lr_scheduler != None:
            scale_lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[scale_lr_scheduler, step_lr_scheduler],
                milestones=[cfg.train.scale_warmup_epochs * nr_of_steps_p_epoch],
            )
        else:
            scale_lr_scheduler = step_lr_scheduler

    elif cfg.train.scale_lr_annealing_min: 
       
        T_max = (cfg.train.epochs - cfg.train.scale_warmup_epochs) * nr_of_steps_p_epoch # - warmup epochs
        annealing_scheduler = custom_CosineAnnealingLR(
            optimizer,
            T_max=T_max,
            excluded_group_name= "Main LR",
            eta_min=cfg.train.scale_lr_annealing_min,
        )
       
        if scale_lr_scheduler is not None:
            scale_lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[scale_lr_scheduler, annealing_scheduler],
                milestones=[warmup_steps],
            )
        else:
            scale_lr_scheduler = annealing_scheduler

    if scale_lr_scheduler is not None:
        if cfg.train.scale_lr_step_sizes is not None or cfg.train.scale_lr_annealing_min is not None:
                step_scheduler_main = MultiStepGroupSpecific(
                        optimizer, step_sizes, excluded_group_name="Scale LR", gamma=cfg.train.lr_gamma)
        else:
             step_scheduler_main = MultiStepGroupSpecific(
                        optimizer, step_sizes, excluded_group_name="Scale LR", gamma=cfg.train.lr_gamma)
        lr_scheduler = torch.optim.lr_scheduler.ChainedScheduler(
                    [scale_lr_scheduler, step_scheduler_main]
                )
    else:
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, step_sizes, gamma=cfg.train.lr_gamma)

    return [{"scheduler": lr_scheduler, "interval": "step"}]

def construct_scheduler_STL_10(
    model,
    optimizer,
    cfg: OmegaConf,
):
    if cfg.model.scale_learn_mode == LearnMode.OFF.value:
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, cfg.train.lr_steps, cfg.train.lr_gamma)
        return [lr_scheduler]
    else:
        return separate_lr_scheduler_scale_learning(optimizer, cfg)

def construct_scheduler_MNIST_SCALE(
        model,
        optimizer,
        cfg: OmegaConf
):
    if cfg.model.scale_learn_mode == LearnMode.OFF.value:
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, cfg.train.lr_steps, cfg.train.lr_gamma
            )
        return [lr_scheduler]
    else:
        return separate_lr_scheduler_scale_learning(optimizer, cfg)

class MultiStepGroupSpecific(torch.optim.lr_scheduler.MultiStepLR):
    def __init__(self, optimizer, milestones, excluded_group_name, gamma=0.1, last_epoch=-1, verbose=False):
        self.excluded_group_name = excluded_group_name
        super().__init__(optimizer, milestones, gamma, last_epoch, verbose)

    # Override get_lr_method
    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if self.last_epoch not in self.milestones:
            return [group['lr'] for group in self.optimizer.param_groups]
        return [group['lr'] * self.gamma ** self.milestones[self.last_epoch] if group['name'] != self.excluded_group_name else 
                group['lr'] for group in self.optimizer.param_groups]

class custom_CosineAnnealingLR(torch.optim.lr_scheduler.CosineAnnealingLR):
    def __init__(self, optimizer, T_max, eta_min=0, excluded_group_name = "Main LR", last_epoch=-1, verbose=False):
        self.exluded_group_name = excluded_group_name
        super().__init__(optimizer, T_max, eta_min, last_epoch, verbose)

    # Override get_lr_method
    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if self.last_epoch == 0:
            return [group['lr'] for group in self.optimizer.param_groups]
        elif self._step_count == 1 and self.last_epoch > 0:
            return [self.eta_min + (base_lr - self.eta_min) *
                    (1 + math.cos((self.last_epoch) * math.pi / self.T_max)) / 2 if group['name'] != self.exluded_group_name else group['lr']
                    for base_lr, group in
                    zip(self.base_lrs, self.optimizer.param_groups)]
        elif (self.last_epoch - 1 - self.T_max) % (2 * self.T_max) == 0:
            return [group['lr'] + (base_lr - self.eta_min) *
                    (1 - math.cos(math.pi / self.T_max)) / 2 if group['name'] != self.exluded_group_name else group['lr']
                    for base_lr, group in
                    zip(self.base_lrs, self.optimizer.param_groups)]
        return [(1 + math.cos(math.pi * self.last_epoch / self.T_max)) /
                (1 + math.cos(math.pi * (self.last_epoch - 1) / self.T_max)) *
                (group['lr'] - self.eta_min) + self.eta_min if group['name'] != self.exluded_group_name else group['lr']
                for group in self.optimizer.param_groups]
    
class custom_LambdaLR(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, lr_lambda, nr_steps, excluded_group_name, last_epoch=-1, verbose=False):
        self.nr_steps = nr_steps
        self.excluded_group_name = excluded_group_name
        super().__init__(optimizer, lr_lambda, last_epoch, verbose)

    # Override get_lr_method
    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.")
        if self.last_epoch >= self.nr_steps:
            return [group['lr'] for group in self.optimizer.param_groups]

        lr_return = []
        for i, group in enumerate(self.optimizer.param_groups):
            if group['name'] == self.excluded_group_name:
                lr_return.append(group['lr'])
            else:    
                lr_return.append(self.base_lrs[i] * self.lr_lambdas[i](self.last_epoch))
        return lr_return

