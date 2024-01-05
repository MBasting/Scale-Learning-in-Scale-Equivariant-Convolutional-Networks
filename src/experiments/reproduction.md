# Commands used for reproduction
All commands are run in the `src` directory.

## Validation
### Do Internal Scales really matter?
```python scale_learn_sweep.py shared.sample_scales=[LOGUNIFORM,1,8] var.init_scales=[1.68,2.12,2.67,3.37,4.24,5.35,6.73] shared.learnable=False shared.pooling_type=max shared.batch_size=128 var.seed=[0,1,2] shared.exp_name=LogUniform_exp_4 shared.img_size=168 shared.project=scale_influence_2 shared.lr=0.01 shared.epochs=60 shared.step_sizes=[20,40] shared.simplified=True shared.nr_filters=16 shared.kernel_size_init_k=4 shared.resize_factor=1 shared.cutoff=[1,10] shared.nr_internal=3 shared.basis_min_scale=2 shared.pooling_settings=[42,42] shared.eff_size=5 shared.basis_type=hermite_a```

### Can we learn the internal scales?

```python scale_learn_sweep.py var.sample_scales=[[LOGUNIFORM,1,2.83],[LOGUNIFORM,1,4.76],[LOGUNIFORM,1,8]] shared.init_scales=3.0 shared.learnable=True shared.pooling_type=max shared.batch_size=128 var.seed=[0,1,2] shared.exp_name=scale_learning_16 shared.img_size=168 shared.project=scale_learning shared.lr=0.01 shared.epochs=60 shared.step_sizes=[20,40] shared.simplified=True shared.nr_filters=16 shared.resize_factor=1 shared.cutoff=[1,10] shared.nr_internal=3 shared.basis_min_scale=2 shared.pooling_settings=[42,42] shared.eff_size=5 shared.basis_type=hermite_a shared.learn_mode=6 shared.scale_lr=0.01 shared.learnable_basis_min=True var.scale_warmup_epochs=[5] shared.scale_lr_step_sizes=[20,40] shared.scale_lr_gamma=0.75 shared.kernel_size_init_k=4```
## Model Choices
### How does initialisation of scales, when scale learning, affect the search for best internal scales?
```python scale_learn_sweep.py shared.sample_scales=[LOGUNIFORM,1,8] var.init_scales=[1.5,3,6] shared.learnable=True shared.pooling_type=max shared.batch_size=128 var.seed=[0,1,2] shared.exp_name=scale_learning_init_exp_4 shared.img_size=168 shared.project=scale_learning shared.lr=0.01 shared.epochs=60 shared.step_sizes=[20,40] shared.simplified=True shared.nr_filters=16 shared.resize_factor=1 shared.cutoff=[1,10] shared.nr_internal=3 var.basis_min_scale=[1,2,4] shared.pooling_settings=[42,42] shared.eff_size=5 shared.basis_type=hermite_a shared.learn_mode=6 shared.scale_lr=0.01 shared.learnable_basis_min=True shared.scale_warmup_epochs=5 shared.scale_lr_step_sizes=[20,40] shared.scale_lr_gamma=0.75 shared.kernel_size_init_k=4```

### How does parameterisation of learnable scales affect learnability?

```python scale_learn_sweep.py var.sample_scales=[[LOGUNIFORM,1,2.83],[LOGUNIFORM,1,4.76],[LOGUNIFORM,1,8]] shared.init_scales=3.0 shared.learnable=True shared.pooling_type=max shared.batch_size=128 var.seed=[0,1,2] shared.exp_name=Compare_Parameterization_Methods shared.img_size=168 shared.project=scale_learning shared.lr=0.01 shared.epochs=60 shared.step_sizes=[20,40] shared.simplified=True shared.nr_filters=16 shared.resize_factor=1 shared.cutoff=[1,10] shared.nr_internal=3 shared.basis_min_scale=2 shared.pooling_settings=[42,42] shared.eff_size=5 shared.basis_type=hermite_a var.learn_mode=[2,4] shared.scale_lr=0.01 shared.learnable_basis_min=True var.scale_warmup_epochs=[5] shared.scale_lr_step_sizes=[20,40] shared.scale_lr_gamma=0.75 shared.kernel_size_init_k=4```

## Baseline
### Does Learnable scales improve baseline on popular scale-equivariant image classification baselines?
```python run_exp.py shared.wandb.project=scale_learning shared.wandb.local=False shared.model.conv_index=3 shared.dataset.d_index=1 shared.wandb.tags=[MNIST_SCALE_SOTA_2] shared.dataset.nr_workers=8 var.seed=[0,1,2,3,4,5] shared.model.upsample_factor=2.0 var.model.scale_learn_mode=[2,5,6] shared.model.kernel_size_init_k=4 shared.model.learnable_basis_min=true shared.model.basis_type_variant=b shared.train.scale_lr=0.005 shared.train.scale_warmup_epochs=10 shared.train.scale_lr_gamma=0.1```