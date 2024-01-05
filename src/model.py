# system imports
from collections import OrderedDict
import math
import warnings
import torch
import torch.nn as nn

import disco.ses_conv_learnable as SESN
from utils.model_utils import LearnMode, ConvType, calculate_scales_parameter


def get_resp_kernel_sizes(
    conv_scales, base_kernel_size, basis_min_scale, kernel_size_init_k
):
    if kernel_size_init_k is not None:
        kernel_sizes = [
            2 * math.ceil(kernel_size_init_k * scale_temp * basis_min_scale) + 1
            for scale_temp in conv_scales
        ]
    else:
        kernel_sizes = [
            int(round(base_kernel_size * scales_temp) // 2 * 2 + 1)
            for scales_temp in conv_scales
        ]
    return kernel_sizes


# def get_resp_kernel_sizes(sample_scales, base_kernel_size):
#     kernel_sizes = [
#         int(round(base_kernel_size * scales_temp) // 2 * 2 + 1)
#         for scales_temp in sample_scales
#     ]
#     return kernel_sizes


class Network(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        nr_classes: int,
        nr_scales: int,
        conv_index: int,  # Model_Index
        scale_learn_mode: int,  # Determines Scale Learning Mode
        learnable_basis_min: bool,
        decoupled_basis_min: bool,
        img_size: (int, int),
        # SESN/DISCO Parameters
        init_scales=2.0,
        nr_internal_scales=4,
        eff_size=7,
        base_kernel_size=None,
        largest_kernel_size=None,
        kernel_size_init_k=None,
        kernel_size=7,
        # Kernel Size only used for CNN Simple
        use_HH=False,
        # Enables keeps the vector represention in the intermediate layer, projects in the end
        basis_type_variant="a",
        # Other layer Parameters
        padding_mode="constant",
        batch_norm=True,
        Relu=True,
        pool_type_inter="Max",
        pool_size_inter=2,
        pool_type_final="Max",
        pool_size_final=None,
        pool_padding_final=2,
        linear=True,
        linear_hidden_class=256,
        dropout_class=0.7,
        upsample_factor=2.0,
        # Conv parameters
        conv_bias=True,
        # Basis Parameters SESN/Disco
        basis_max_order=4,
        basis_mult=1.4,
        basis_min_scale=1.5,
        basis_save_dir="../disco/precalculated_basis/",
        **kwargs,
    ):
        super().__init__()
        self.out_classes = nr_classes
        self.out_scales = nr_scales
        self.nr_layers = len(hidden_channels)
        self.conv_type = ConvType(conv_index)
        self.learn_type = (
            LearnMode(scale_learn_mode)
            if self.conv_type == ConvType.SESN
            else LearnMode.OFF
        )
        self.learnable_scale = (
            True
            if (
                self.conv_type == ConvType.SESN
            )
            and self.learn_type != LearnMode.OFF
            else False
        )
        self.learnable_basis_min = (
            learnable_basis_min if self.learnable_scale else False
        )
        self.decoupled_basis_min = (
            decoupled_basis_min if self.learnable_scale else False
        )
        self.kernel_size = kernel_size
        self.largest_kernel_size = largest_kernel_size
        self.kernel_size_init_k = kernel_size_init_k

        # Scale Related Parameters
        self.scales_parameter = None  # Initialize scales_Parameter

        self.vector_rep = (
            use_HH and self.conv_type.value >= ConvType.SESN.value
        )
        self.nr_internal_scales = nr_internal_scales
        self.basis_type = None
        self.effective_size = eff_size

        # Other Layer Parameters
        self.padding_mode = padding_mode
        self.batch_norm = batch_norm
        self.Relu = Relu
        self.linear = linear

        self.conv_bias = conv_bias

        # Load dictionary for SESN/Disco Parameters
        default_kwargs = {}
        default_kwargs["basis_max_order"] = basis_max_order
        default_kwargs["basis_mult"] = basis_mult
        # default_kwargs["basis_min_scale"] = basis_min_scale
        default_kwargs["basis_save_dir"] = basis_save_dir

        self.basis_min_scale = basis_min_scale

        # Initialize Layer list to populate
        final_modules_class = []

        # Override basis_min_scale if necessary
        self.scales_parameter, self.basis_min_scale = calculate_scales_parameter(
            self.conv_type,
            self.learn_type,
            self.nr_internal_scales,
            self.learnable_basis_min,
            init_scales,
            basis_min_scale,
            decoupled_basis_min=self.decoupled_basis_min,
            nr_layers=self.nr_layers,
            **default_kwargs,
        )

        # Kernel size in SESN/DISCO is either expressed by calculating it using formula
        # from base_kernel_size (eff_size) or by using overriding largest_kernel_size
        if base_kernel_size is None:
            base_kernel_size = self.effective_size

        if not self.learnable_scale:
            kernel_sizes = get_resp_kernel_sizes(
                self.scales_parameter,
                base_kernel_size,
                basis_min_scale,
                kernel_size_init_k,
            )
            self.largest_kernel_size = (
                kernel_sizes[-1] if largest_kernel_size is None else largest_kernel_size
            )

        # If we are dealing with scale-convolution layers, first max project
        if self.conv_type.value >= ConvType.SESN.value:
            final_modules_class.append(SESN.SESMaxProjection())
        final_modules_class.append(nn.ReLU(True))

        # If we have pooling settings available
        if pool_size_final is None:
            pool_size_final = self.calculate_pool_size(
                (img_size[0] * upsample_factor) // (2 * (self.nr_layers - 1)),
                output_size=2,
                padding=pool_padding_final,
            )

            pool_output = 2
        else:
            pool_output = (
                img_size[0] - pool_size_final + 2 * pool_padding_final
            ) // pool_size_final + 1

        if pool_type_final == "AvgMax":
            final_modules_class.extend(
                [
                    nn.MaxPool2d(pool_size_final - 1, stride=1),
                    nn.AvgPool2d(2, stride=pool_size_final // 2),
                ]
            )
        else:
            if pool_size_final == "Avg":
                pool_layer = nn.AvgPool2d
            else:
                pool_layer = nn.MaxPool2d
            final_modules_class.append(
                pool_layer(
                    pool_size_final, stride=pool_size_final, padding=pool_padding_final
                )
            )

        final_modules_class.append(nn.BatchNorm2d(hidden_channels[-1]))
        # Update nr. of output channels
        in_channels_final = (pool_output**2) * hidden_channels[-1]

        if linear:
            final_modules_class.extend(
                [
                    nn.Flatten(),
                    nn.Linear(in_channels_final, linear_hidden_class, bias=False),
                    nn.BatchNorm1d(linear_hidden_class),
                    nn.ReLU(True),
                    nn.Dropout(dropout_class),
                    nn.Linear(linear_hidden_class, nr_classes),
                ]
            )
        else:
            final_modules_class.extend(
                [
                    nn.Conv2d(in_channels_final // 4, linear_hidden_class, 2),
                    nn.BatchNorm2d(linear_hidden_class),
                    nn.ReLU(True),
                    nn.Dropout(dropout_class),
                    nn.Conv2d(linear_hidden_class, nr_classes, 1),
                    nn.Flatten(),
                ]
            )

        # Create Classification branch
        self.final_class = nn.Sequential(*final_modules_class)

        if self.conv_type.value >= ConvType.SESN.value:
            # For SESN/DISCO models additionally have basis_types (a,b)
            if self.conv_type.value == ConvType.SESN.value:
                self.basis_type = f"hermite_{basis_type_variant}"
            elif self.conv_type.value == ConvType.DISCO_FREE.value:
                self.basis_type = "disco_free_form"
            else:
                self.basis_type = f"disco_{basis_type_variant}"

        cv_1 = self.get_layer(
            in_channels,
            hidden_channels[0],
            conv_type=self.conv_type,
            kernel_size=self.kernel_size,
            effective_size=self.effective_size,
            # Vector rep = False because first layer
            scales=self.scales_parameter,
            vector_rep=False,
            padding_mode=self.padding_mode,
            basis_type=self.basis_type,
            learn_mode=self.learn_type,
            conv_layer_index=0,
            **default_kwargs,
        )
        # Initialize list containing conv_layers
        self.conv_layers = OrderedDict([("conv_1", cv_1)])

        layer_index = 2
        # Initialize Main Module Placeholder
        self.module_list = OrderedDict(
            [("upsample", nn.Upsample(scale_factor=upsample_factor)), ("conv_1", cv_1)]
        )
        # Populate Module List Dynamically
        for i in range(self.nr_layers - 1):
            # Add intermediate Layers
            layer_index = self.add_intermediate(
                hidden_channels[i],
                layer_index,
                pool_type_inter,
                pool_size_inter,
                self.vector_rep,
            )

            # Add Special Convolution Layer
            layer_cur = self.get_layer(
                hidden_channels[i],
                hidden_channels[i + 1],
                conv_type=self.conv_type,
                kernel_size=self.kernel_size,
                effective_size=self.effective_size,
                scales=self.scales_parameter,
                vector_rep=self.vector_rep,
                padding_mode=self.padding_mode,
                basis_type=self.basis_type,
                learn_mode=self.learn_type,
                conv_layer_index=i + 1,
                **default_kwargs,
            )

            # Add layer to module list + track convolutional layers
            self.module_list[f"conv_{layer_index}"] = layer_cur
            self.conv_layers[f"conv_{layer_index}"] = layer_cur
            layer_index += 1

        # Create Main Module
        self.main = nn.Sequential(self.module_list)

        # Placeholder of gradients
        self.gradients = []
        self.feature_maps = []
        self.hooks = []

    def calculate_pool_size(self, input_size, output_size, padding):
        return int((input_size + 2 * padding) // output_size - 1)

    def enable_hooks(self):
        # Reset gradients
        self.gradients = []
        self.feature_maps = []

        warnings.warn(
            "Gradients and feature maps are saved and added until disable_hooks is called"
        )
        warnings.warn("Gradients are saved from last to first layer")

        for _, conv_layer in self.conv_layers.items():
            # Register forward and backward hook we might need when generating saliency maps
            self.hooks.append(conv_layer.register_forward_hook(self.save_feature_map))
            self.hooks.append(conv_layer.register_backward_hook(self.save_gradients))

    def disable_hooks(self):
        for hook in self.hooks:
            hook.remove()

    # Saves gradients of conv_layer for first sample
    # These are called in reverse order so we save gradient to the front of the list
    def save_gradients(self, module, grad_input, grad_output):
        self.gradients.insert(0, grad_output[0])

    # Save Feature maps of conv layer for first sample
    def save_feature_map(self, module, input, output):
        self.feature_maps.append(output.detach())

    def get_scales(self):
        if self.learn_type != LearnMode.OFF:
            scales = self.main[1].calculate_scales()
            return scales.detach().cpu().tolist()
        else:
            if self.conv_type.value >= ConvType.SESN.value:
                return [round(scale, 2) for scale in self.scales_parameter]
            else:
                return None

    def log_scales(self, logger):
        # Logs the scales and ISR if used
        scales = self.get_scales()
        if scales is not None and self.learn_type != LearnMode.OFF:
            for index, scale in enumerate(scales):
                logger(f"SL/Scale/{index+1}", scale, sync_dist=True, prog_bar=True)
            if self.learn_type not in [LearnMode.RATIO, LearnMode.RATIO_SQUARED, LearnMode.DIRECT_SCALES]:
                logger(
                    "SL/Scale Parameter",
                    self.scales_parameter.detach().cpu().tolist(),
                    sync_dist=True,
                )
                 
                logger(
                    "SL/ISR",
                    round(scales[-1] / scales[0], 3),
                    sync_dist=True,
                )
            if self.decoupled_basis_min:
                for i, basis_min in enumerate(self.basis_min_scale):
                    logger(
                        f"SL/Basis_min/Layer {i+1}",
                        basis_min.detach().cpu(),
                        sync_dist=True,
                    )

    def get_layer(
        self,
        in_channels,
        out_channels,
        conv_type,
        kernel_size=None,
        effective_size=None,
        scales=None,
        vector_rep=False,
        padding_mode="constant",
        basis_type="hermite_a",
        learn_mode=LearnMode.OFF,
        conv_layer_index=0,
        **default_kwargs,
    ):
        if conv_type == ConvType.CNN_SIMPLE:
            return nn.Conv2d(
                in_channels, out_channels, kernel_size, padding=kernel_size // 2
            )
        else:
            if learn_mode == LearnMode.OFF:
                if vector_rep:
                    return SESN.SESConv_H_H(
                        in_channels,
                        out_channels,
                        1,
                        self.largest_kernel_size,
                        effective_size,
                        scales=scales,
                        padding=self.largest_kernel_size // 2,
                        padding_mode=padding_mode,
                        bias=self.conv_bias,
                        basis_type=basis_type,
                        basis_min_scale=self.basis_min_scale,
                        **default_kwargs,
                    )
                else:
                    return SESN.SESConv_Z2_H(
                        in_channels,
                        out_channels,
                        self.largest_kernel_size,
                        effective_size,
                        scales=scales,
                        padding=self.largest_kernel_size // 2,
                        padding_mode=padding_mode,
                        bias=self.conv_bias,
                        basis_type=basis_type,
                        basis_min_scale=self.basis_min_scale,
                        **default_kwargs,
                    )
            else:
                if self.decoupled_basis_min:
                    current_basis_min_scale = self.basis_min_scale[conv_layer_index]
                else:
                    current_basis_min_scale = self.basis_min_scale
                if vector_rep:
                    return SESN.SESConv_H_H_Learnable(
                        in_channels,
                        out_channels,
                        1,
                        scales,
                        learn_mode,
                        effective_size,
                        self.nr_internal_scales,
                        padding_mode=padding_mode,
                        bias=self.conv_bias,
                        basis_type=basis_type,
                        basis_min_scale=current_basis_min_scale,
                        largest_kernel_size=self.largest_kernel_size,
                        init_k=self.kernel_size_init_k,
                        **default_kwargs,
                    )
                else:
                    return SESN.SESConv_Z2_H_Learnable(
                        in_channels,
                        out_channels,
                        scales,
                        learn_mode,
                        effective_size,
                        self.nr_internal_scales,
                        padding_mode=padding_mode,
                        bias=self.conv_bias,
                        basis_type=basis_type,
                        basis_min_scale=current_basis_min_scale,
                        largest_kernel_size=self.largest_kernel_size,
                        init_k=self.kernel_size_init_k,
                        **default_kwargs,
                    )

    def add_intermediate(
        self, hidden_channels, layer_index, pool_type, pool_size, vector_rep=False
    ):
        # Only add max projection if vector_rep == OFF but we do want to project
        if not vector_rep and self.conv_type.value >= ConvType.SESN.value:
            self.module_list[f"Max_p_{layer_index}"] = SESN.SESMaxProjection()
            layer_index += 1
        if self.Relu and self.conv_type.value != ConvType.SESN.value:
            self.module_list[f"ReLU_{layer_index}"] = nn.ReLU(True)
            layer_index += 1

        if pool_type is not None:
            pool = None
            if not vector_rep:
                if pool_type == "Max":
                    pool = nn.MaxPool2d(pool_size)
                elif pool_type == "Avg" or pool_type == "AvgMax":
                    pool = nn.AvgPool2d(pool_size)
            else:
                if pool_type == "Max":
                    pool = nn.MaxPool3d(
                        [1, pool_size, pool_size], stride=[1, pool_size, pool_size]
                    )
                elif pool_type == "Avg" or pool_type == "AvgMax":
                    pool = nn.AvgPool3d(
                        [1, pool_size, pool_size], stride=[1, pool_size, pool_size]
                    )
            self.module_list[f"Pool_{layer_index}"] = pool
            layer_index += 1

        if self.batch_norm:
            if not vector_rep:
                self.module_list[f"BatchNorm_{layer_index}"] = nn.BatchNorm2d(
                    hidden_channels
                )
            else:
                self.module_list[f"BatchNorm_{layer_index}"] = nn.BatchNorm3d(
                    hidden_channels
                )
            layer_index += 1
        return layer_index

    def get_nr_of_parameters(self):
        trainable_params_count = [
            p.numel() for p in self.parameters() if p.requires_grad
        ]
        return trainable_params_count, sum(trainable_params_count)

    def forward(self, x):
        temp = self.main(x)
        out = self.final_class(temp)

        return out.unsqueeze(0)
