'''
This file is a part of the official implementation of
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

2) an unofficial reimplemmentaion of "Wide Residual Networks"
    by Sergey Zagoruyko, Nikos Komodakis, BMVC 2016
    arxiv: https://arxiv.org/abs/1605.07146
    the reimplementation is performed by https://github.com/xternalz
    code: https://github.com/xternalz/WideResNet-pytorch

3) the official implemenation of "Scale Learning in Scale-Equivariant Convolutional Networks"
    by Mark Basting, Jan van Gemert, VISAPP 2024,
    arxiv : 
    code: https://github.com/MBasting/scale-equiv-learnable-cnn

---------------------------------------------------------------------------

MIT License. Copyright (c) 2023 Mark Basting
MIT License. Copyright (c) 2021 Ivan Sosnovik, Artem Moskalev
MIT License. Copyright (c) 2020 Ivan Sosnovik, Michał Szmaja
MIT License. Copyright (c) 2019 xternalz
'''
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import math


from .stl_ses import wrn_SESN
from .stl_disco_models import BasicBlock
from .ses_conv_learnable import SESConv_Z2_H_Learnable, SESConv_H_H_Learnable, SESConv_H_H, SESConv_Z2_H, SESConv_H_H_1x1, SESMaxProjection
from utils.model_utils import LearnMode, ConvType, calculate_scales_parameter


class BasicBlock_Learnable(BasicBlock):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0, scales=[1.0],
                 pool=False, interscale=False,basis_type='hermite_a', basis_min_scale = 0.9, learn_mode = LearnMode.OFF, nr_internal_scales = 3, largest_kernel_size = 11, kernel_size_init_k = 1.5, **kwargs):

        # During initialization need to use basis_min_scale as a float
        basis_min_init = basis_min_scale if learn_mode != LearnMode.OFF else basis_min_scale.item()
        temp_scales = [scales.item()] if (learn_mode == LearnMode.OFF or learn_mode == LearnMode.SINGLE_RATIO or learn_mode == LearnMode.SINGLE_RATIO_SQUARED) else scales.tolist()
        super().__init__(in_planes, out_planes, stride, dropRate, temp_scales,
                                                   pool=pool, interscale=interscale, basis_type=basis_type, basis_min_scale=basis_min_init, **kwargs) 
               
        # Depending on LearnMode we override the default convolutions
        if learn_mode != LearnMode.OFF:
            if pool:
                self.conv1 = nn.Sequential(
                    SESMaxProjection(),
                    SESConv_Z2_H_Learnable(in_planes, out_planes, scales, learn_mode, 3,
                                 nr_internal_scales, largest_kernel_size=largest_kernel_size,
                                init_k = kernel_size_init_k, stride=stride, padding=3, bias=False,
                                basis_type=basis_type,  basis_min_scale=basis_min_scale, **kwargs)
                )
            else:
                if interscale:
                    return NotImplementedError
                    self.conv1 = SESConv_H_H_Learnable(in_planes, out_planes, 2, scales, learn_mode, effective_size=3, nr_internal_scales=nr_internal_scales, 
                                                       stride=stride, padding=2, bias=False, largest_kernel_size=largest_kernel_size, init_k = kernel_size_init_k, 
                                                       basis_type=basis_type,  basis_min_scale=basis_min_scale, **kwargs)
                else:
                    self.conv1 = SESConv_H_H_Learnable(in_planes, out_planes, 1, scales, learn_mode, effective_size=3, nr_internal_scales=nr_internal_scales, 
                                                       stride=stride, padding=3, bias=False, largest_kernel_size=largest_kernel_size, init_k = kernel_size_init_k,  
                                                       basis_type=basis_type,  basis_min_scale=basis_min_scale, **kwargs)

            self.conv2 = SESConv_H_H_Learnable(out_planes, out_planes, 1, scales, learn_mode, effective_size=3, nr_internal_scales=nr_internal_scales, 
                                                       stride=1, padding=3, bias=False, largest_kernel_size=largest_kernel_size, init_k = kernel_size_init_k, 
                                                       basis_type=basis_type,  basis_min_scale=basis_min_scale, **kwargs)

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0,
                 scales=[0.0], pool=False, interscale=False, basis_type='hermite_a', basis_min_scale=0.9,
                 learn_mode = LearnMode.OFF, nr_internal_scales = 3, largest_kernel_size = 11, **kwargs):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride,
                                      dropRate, scales, pool, interscale, basis_type, basis_min_scale, learn_mode = learn_mode, nr_internal_scales = nr_internal_scales, largest_kernel_size = largest_kernel_size,**kwargs)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride,
                    dropRate, scales, pool, interscale, basis_type, basis_min_scale, **kwargs):

        layers = []
        for i in range(nb_layers):
            pool_layer = pool and (i == 0)
            interscale_layer = interscale and (i == 0)
            layers.append(block(i == 0 and in_planes or out_planes,
                                out_planes, i == 0 and stride or 1, dropRate, scales,
                                pool=pool_layer, interscale=interscale_layer,
                                basis_type=basis_type,  basis_min_scale=basis_min_scale, **kwargs))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet_Learnable(nn.Module):
    def __init__(self, depth, num_classes, conv_index = ConvType.SESN, learn_mode = LearnMode.OFF, learnable_basis_min = False, nr_internal = 3, largest_kernel_size = 11, kernel_size_init_k = 2, widen_factor=1, dropRate=0.0, init_scales=[1.0],
                 pools=[False, False, False], interscale=[False, False, False],
                 basis_type_variant='a', basis_min_scale=0.9, **kwargs):
        super(WideResNet_Learnable, self).__init__()
        self.learnable_scale = learn_mode != LearnMode.OFF 
        self.learn_type = learn_mode
        self.conv_type = ConvType(conv_index)
        if self.conv_type == ConvType.SESN:
            basis_type = 'hermite_' + basis_type_variant
        elif self.conv_type == ConvType.DISCO:
            basis_type = 'disco_' + basis_type_variant
        
        self.basis_min_scale = basis_min_scale

        # Override basis_min_scale if necessary
        self.scales_parameter, self.basis_min_scale = calculate_scales_parameter(
            self.conv_type, learn_mode, nr_internal,
            learnable_basis_min, init_scales,
            basis_min_scale=basis_min_scale, basis_type=basis_type, **kwargs)
        # print(self.basis_min_scale, self.scales_parameter, learn_mode.name)
        
        new_kwargs = {key: value for key, value in kwargs.items() if 'basis' in key}
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) // 6
        if learn_mode != LearnMode.OFF:
            block = BasicBlock_Learnable
        else:
            block = BasicBlock
        # 1st conv before any network block
        if learn_mode != LearnMode.OFF:
            self.conv1 = SESConv_Z2_H_Learnable(3, nChannels[0], self.scales_parameter, learn_mode, 3,
                                    nr_internal, largest_kernel_size=largest_kernel_size,
                                    init_k = kernel_size_init_k, stride=1, padding=3, bias=False,
                                    basis_type=basis_type,  basis_min_scale=self.basis_min_scale, **new_kwargs)
        else:
            self.conv1 = SESConv_Z2_H(3, nChannels[0], kernel_size=7, effective_size=3, stride=1,
                                    padding=3, bias=False, scales=self.scales_parameter, basis_type=basis_type,  basis_min_scale=self.basis_min_scale, **new_kwargs)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate,
                                   scales=self.scales_parameter, pool=pools[0], interscale=interscale[0],
                                  basis_type=basis_type,  basis_min_scale=self.basis_min_scale, 
                                   learn_mode = learn_mode, nr_internal_scales = nr_internal, largest_kernel_size = largest_kernel_size,
                                   kernel_size_init_k = kernel_size_init_k,**new_kwargs)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate,
                                   scales=self.scales_parameter, pool=pools[1], interscale=interscale[1],
                                   basis_type=basis_type,  basis_min_scale=self.basis_min_scale, 
                                   learn_mode = learn_mode, nr_internal_scales = nr_internal, largest_kernel_size = largest_kernel_size,
                                   kernel_size_init_k = kernel_size_init_k,
                                   **new_kwargs)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate,
                                   scales=self.scales_parameter, pool=pools[2], interscale=interscale[2],
                                   basis_type=basis_type,  basis_min_scale=self.basis_min_scale, 
                                   learn_mode = learn_mode, nr_internal_scales = nr_internal, largest_kernel_size = largest_kernel_size,
                                   kernel_size_init_k = kernel_size_init_k,
                                   **new_kwargs)

        # global average pooling and classifier
        self.proj = SESMaxProjection()
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]
        for m in self.modules():
            if isinstance(m, (SESConv_H_H, SESConv_Z2_H, SESConv_H_H_1x1, SESConv_H_H_Learnable, SESConv_Z2_H_Learnable)):
                nelement = m.weight.nelement()
                n = nelement / m.in_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def get_scales(self):
        if self.learn_type != LearnMode.OFF:
            scales = self.conv1.calculate_scales()
            return scales.detach().cpu().tolist()
        else:
            if self.conv_type.value >= ConvType.FLEXCONVSCALE.value:
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
                        
    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.proj(out)
        out = self.relu(self.bn1(out))

        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(-1, self.nChannels)
        out = self.fc(out)
        return [out]


# def wrn_disco(basis_save_dir, ** kwargs):
#     scales = [1.0, 1.41, 2.0]
#     return WideResNet(depth=16, num_classes=10, widen_factor=8, dropRate=0.3,
#                       scales=scales, pools=[False, True, True], basis_type='disco_a',
#                       basis_save_dir=basis_save_dir, basis_min_scale=0.9)

def wrn_learnable(depth=16, nr_classes=10, scale_learn_mode = LearnMode.OFF.value, learnable_basis_min = False, nr_internal = 3, largest_kernel_size = 11, kernel_size_init_k = 1.5, widen_factor=8, drop_rate=0.3, init_scales=2.0,
                 pools=[False, True, True], interscale=[False, False, False],
                 conv_index = ConvType.SESN.value, basis_type_variant='a', basis_min_scale=0.9, **kwargs):
    
    learn_mode = LearnMode(scale_learn_mode)
    if learn_mode == LearnMode.OFF and conv_index == ConvType.SESN.value:
        scales = [basis_min_scale * init_scales**(i/(nr_internal-1)) for i in range(nr_internal)]
        return wrn_SESN(depth=depth, nr_classes=nr_classes, widen_factor = widen_factor, drop_rate=drop_rate, scales = scales, pools=pools, interscale = interscale)
    else:
        return WideResNet_Learnable(depth, nr_classes, conv_index, learn_mode, learnable_basis_min, nr_internal, largest_kernel_size, kernel_size_init_k, widen_factor, drop_rate, init_scales,
                    pools, interscale, basis_type_variant, basis_min_scale=basis_min_scale, **kwargs)


