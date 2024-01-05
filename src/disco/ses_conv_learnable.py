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

2) the official implementation of "Scale Equivariance Improves Siamese Tracking"
    by Ivan Sosnovik*, Artem Moskalev*, and Arnold Smeulders, WACV 2021
    arxiv: https://arxiv.org/abs/2007.09115
    code: https://github.com/ISosnovik/SiamSE

3) the official implemenation of "Scale Learning in Scale-Equivariant Convolutional Networks"
    by Mark Basting, Jan van Gemert, VISAPP 2024,
    arxiv : 
    code: https://github.com/MBasting/scale-equiv-learnable-cnn

---------------------------------------------------------------------------

MIT License. Copyright (c) 2023 Mark Basting
MIT License. Copyright (c) 2020-2021 Ivan Sosnovik, Artem Moskalev
MIT License. Copyright (c) 2020 Ivan Sosnovik, Michał Szmaja
'''


import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.model_utils import calculate_scales

from .basis import HermiteBasisA, HermiteBasisB, HermiteBasisA_ref, HermiteBasisB_ref
from .basis import LRHarmonicsBasis
from .basis import BSplineBasis
from .basis import FourierBesselBasis
from .basis import DISCOBasisA, DISCOBasisB, DISCOBasisFree_ref

def _parse_basis(basis_type, kernel_size, scales, effective_size, learn_scales, **kwargs):
    if basis_type == 'hermite_a':
        if learn_scales:
            return HermiteBasisA_ref(kernel_size, scales, effective_size)
        else:
            return HermiteBasisA(kernel_size, scales, effective_size)

    if basis_type == 'hermite_b':
        if learn_scales:
            return HermiteBasisB_ref(kernel_size, scales, effective_size,
                                basis_mult=kwargs['basis_mult'],
                                basis_max_order=kwargs['basis_max_order'])
        else:
            return HermiteBasisB(kernel_size, scales, effective_size,
                                basis_mult=kwargs['basis_mult'],
                                basis_max_order=kwargs['basis_max_order'])

    if basis_type == 'lr_harmonics':
        return LRHarmonicsBasis(kernel_size, scales, effective_size,
                                basis_max_order=kwargs['basis_max_order'],
                                basis_num_rotations=kwargs['basis_num_rotations'],
                                basis_sigma=kwargs['basis_sigma'])

    if basis_type == 'bspline':
        return BSplineBasis(kernel_size, scales, effective_size,
                            basis_mult=kwargs['basis_mult'],
                            basis_max_order=kwargs['basis_max_order'], cropped=True)

    if basis_type == 'fb':
        return FourierBesselBasis(kernel_size, scales, effective_size)

    if basis_type == 'disco_a':
        return DISCOBasisA(kernel_size, scales, effective_size, basis_save_dir=kwargs['basis_save_dir'],
                           basis_min_scale=kwargs['basis_min_scale'])

    if basis_type == 'disco_b':
        return DISCOBasisB(kernel_size, scales, effective_size, basis_save_dir=kwargs['basis_save_dir'],
                           basis_mult=kwargs['basis_mult'], basis_max_order=kwargs['basis_max_order'],
                           basis_min_scale=kwargs['basis_min_scale'])
    if basis_type == 'disco_free_form':
        return DISCOBasisFree_ref(kernel_size, scales, effective_size)

    if basis_type == 'disco_random':
        return DISCOBasisFree_ref(kernel_size, scales, effective_size, random=True)



    raise NotImplementedError

class SESConv_Z2_H(nn.Module):
    '''Scale Equivariant Steerable Convolution: Z2 -> (S x Z2)
    [B, C, H, W] -> [B, C', S, H', W']
    Args:
        in_channels: Number of channels in the input image
        out_channels: Number of channels produced by the convolution
        kernel_size: Size of the convolving kernel
        effective_size: The effective size of the kernel with the same # of params
        scales: List of scales of basis
        stride: Stride of the convolution
        padding: Zero-padding added to both sides of the input
        bias: If ``True``, adds a learnable bias to the output
    '''

    def __init__(self, in_channels, out_channels, kernel_size, effective_size,
                 scales=[1.0], stride=1, padding=0, bias=False, basis_type='hermite_a',
                 dilation=1, padding_mode='constant', **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.effective_size = effective_size
        self.scales = scales
        self.num_scales = len(scales)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.padding_mode = padding_mode
        self.basis = _parse_basis(basis_type, kernel_size, scales, effective_size, False, **kwargs)
        assert self.basis.size == self.kernel_size

        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, self.basis.num_funcs))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        # Get basis based on current scales:

        kernel = self.basis(self.weight)
        kernel = kernel.permute(0, 2, 1, 3, 4).contiguous()
        kernel = kernel.view(-1, self.in_channels, self.kernel_size, self.kernel_size)

        # convolution
        if self.padding > 0:
            x = F.pad(x, 4 * [self.padding], mode=self.padding_mode)
        y = F.conv2d(x, kernel, bias=None, stride=self.stride, dilation=self.dilation)
        B, C, H, W = y.shape
        y = y.view(B, self.out_channels, self.num_scales, H, W)

        if self.bias is not None:
            y = y + self.bias.view(1, -1, 1, 1, 1)

        return y

    def extra_repr(self):
        s = '{in_channels}->{out_channels}, padding={padding}, padding_mode={padding_mode}'
        return s.format(**self.__dict__)



class SESConv_Z2_H_Learnable(nn.Module):
    '''Scale Equivariant Steerable Convolution: Z2 -> (S x Z2)
    [B, C, H, W] -> [B, C', S, H', W']
    Args:
        in_channels: Number of channels in the input image
        out_channels: Number of channels produced by the convolution
        scales_param: nn.Parameter which contains the learnable scales
        learn_mode : Learn Mode to use
        effective_size: The effective size of the kernel with the same # of params
        stride: Stride of the convolution
        bias: If ``True``, adds a learnable bias to the output
    '''

    def __init__(self, in_channels, out_channels, scales_param, learn_mode, effective_size, 
                 nr_internal_scales, stride=1, bias=False, basis_type='hermite_a',
                 largest_kernel_size = None, init_k = 4, dilation=1, padding_mode='constant', 
                 **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # Kernel size becomes dependent on effective size and max scale instead! 
        self.effective_size = effective_size
        
        self.scales_param = scales_param
        self.basis_min_param = kwargs['basis_min_scale']
        self.learn_mode = learn_mode
        self.num_scales = nr_internal_scales # Temporary value
        self.stride = stride
        self.dilation = dilation
        self.padding = None
        self.padding_mode = padding_mode
        self.basis_type = basis_type
        self.basis = torch.Tensor()
        self.kwargs = kwargs
        self.kernel_size = None
        self.largest_kernel_size = largest_kernel_size
        self.init_k = init_k if largest_kernel_size is None else None
        # assert self.basis.size == self.kernel_size

        self.num_funcs = self.effective_size**2
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, self.num_funcs))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def calculate_scales(self):
        return calculate_scales(self.learn_mode, self.scales_param, self.num_scales, self.basis_min_param)

    def forward(self, x):
        scales = self.calculate_scales()

        if self.largest_kernel_size is None:
            max_scale = max(scales)
            # Calculate kernel_size
            self.kernel_size = torch.nan_to_num(2 * torch.ceil(self.init_k * max_scale) + 1,3)
            self.kernel_size = max(int(self.kernel_size),3)
        else:
            self.kernel_size = self.largest_kernel_size
        # Load or create the basis
        basis = _parse_basis(self.basis_type, self.kernel_size, scales, self.effective_size, True, **self.kwargs)
        # Create Kernel by multiplying kernel weights and basis
        kernel = basis(self.weight)
        kernel = kernel.permute(0, 2, 1, 3, 4).contiguous()

        self.kernel = kernel
        kernel = kernel.view(-1, self.in_channels, self.kernel_size, self.kernel_size)

        self.padding = self.kernel_size // 2

        # convolution
        if self.padding > 0:
            x = F.pad(x, 4 * [self.padding], mode=self.padding_mode)

        y = F.conv2d(x, kernel, bias=None, stride=self.stride, dilation=self.dilation)
        B, C, H, W = y.shape
        y = y.view(B, self.out_channels, scales.size(dim=0), H, W)

        if self.bias is not None:
            y = y + self.bias.view(1, -1, 1, 1, 1)

        return y

    def extra_repr(self):
        s = '{in_channels}->{out_channels}, padding={padding}, padding_mode={padding_mode}'
        return s.format(**self.__dict__)


class SESConv_H_H(nn.Module):
    '''Scale Equivariant Steerable Convolution: (S x Z2) -> (S x Z2)
    [B, C, S, H, W] -> [B, C', S', H', W']
    Args:
        in_channels: Number of channels in the input image
        out_channels: Number of channels produced by the convolution
        scale_size: Size of scale filter
        kernel_size: Size of the convolving kernel
        effective_size: The effective size of the kernel with the same # of params
        scales: List of scales of basis
        stride: Stride of the convolution
        padding: Zero-padding added to both sides of the input
        bias: If ``True``, adds a learnable bias to the output
    '''

    def __init__(self, in_channels, out_channels, scale_size, kernel_size, effective_size,
                 scales=[1.0], stride=1, padding=0, bias=False, basis_type='hermite_a',
                 dilation=1, padding_mode='constant', **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_size = scale_size
        self.kernel_size = kernel_size
        self.effective_size = effective_size
        self.scales = scales
        self.num_scales = len(scales)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.padding_mode = padding_mode
        self.basis = _parse_basis(basis_type, kernel_size, scales, effective_size, False, **kwargs)
        assert self.basis.size == self.kernel_size

        self.weight = nn.Parameter(torch.Tensor(
            out_channels, in_channels, scale_size, self.basis.num_funcs))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        
        # get kernel
        kernel = self.basis(self.weight)

        # expand kernel
        kernel = kernel.permute(3, 0, 1, 2, 4, 5).contiguous()
        kernel = kernel.view(-1, self.in_channels, self.scale_size,
                             self.kernel_size, self.kernel_size)

        # calculate padding
        if self.scale_size != 1:
            value = x.mean()
            x = F.pad(x, [0, 0, 0, 0, 0, self.scale_size - 1])

        output = 0.0
        for i in range(self.scale_size):
            x_ = x[:, :, i:i + self.num_scales]
            # expand X
            B, C, S, H, W = x_.shape
            x_ = x_.permute(0, 2, 1, 3, 4).contiguous()
            x_ = x_.view(B, -1, H, W)
            if self.padding > 0:
                x_ = F.pad(x_, 4 * [self.padding], mode=self.padding_mode)
            output += F.conv2d(x_, kernel[:, :, i], groups=S,
                               stride=self.stride, dilation=self.dilation)

        # squeeze output
        B, C_, H_, W_ = output.shape
        output = output.view(B, S, -1, H_, W_)
        output = output.permute(0, 2, 1, 3, 4).contiguous()
        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1, 1, 1)
        return output

    def extra_repr(self):
        s = '{in_channels}->{out_channels}, scale_size={scale_size}'
        s += '\npadding={padding}, padding_mode={padding_mode}'
        return s.format(**self.__dict__)

class SESConv_H_H_Learnable(nn.Module):
    '''Scale Equivariant Steerable Convolution: (S x Z2) -> (S x Z2)
    [B, C, S, H, W] -> [B, C', S', H', W']
    Args:
        in_channels: Number of channels in the input image
        out_channels: Number of channels produced by the convolution
        scale_size: Size of scale filter
        scales_param: nn.Parameter which contains the learnable scales
        learn_mode : Learn Mode to use
        effective_size: The effective size of the kernel with the same # of params
        stride: Stride of the convolution
        bias: If ``True``, adds a learnable bias to the output
    '''
    def __init__(self, in_channels, out_channels, scale_size, scales_param, learn_mode, effective_size, 
                 nr_internal_scales, stride=1, bias=False, basis_type='hermite_a',
                 largest_kernel_size = None, init_k = 4, dilation=1, padding_mode='constant', **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # Kernel size becomes dependent on effective size and max scale instead! 
        self.effective_size = effective_size
        self.scale_size = scale_size
        self.scales_param = scales_param
        self.basis_min_param = kwargs['basis_min_scale']
        self.learn_mode = learn_mode
        self.num_scales = nr_internal_scales # Temporary value
        self.stride = stride
        self.dilation = dilation
        self.padding = None
        self.padding_mode = padding_mode
        self.basis_type = basis_type
        self.basis = torch.Tensor()
        self.kwargs = kwargs
        self.num_funcs = self.effective_size**2
        self.largest_kernel_size = largest_kernel_size
        self.init_k = init_k if largest_kernel_size is None else None
        
        self.weight = nn.Parameter(torch.Tensor(
            out_channels, in_channels, scale_size, self.num_funcs))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def calculate_scales(self):
        return calculate_scales(self.learn_mode, self.scales_param, self.num_scales, self.basis_min_param)

    def forward(self, x):
        scales = calculate_scales(self.learn_mode, self.scales_param, self.num_scales, self.basis_min_param)

        max_scale = max(scales)
        if self.largest_kernel_size is None:
            max_scale = max(scales)
            # Calculate kernel_size
            self.kernel_size = torch.nan_to_num(2 * torch.ceil(self.init_k * max_scale) + 1,3)
            self.kernel_size = max(int(self.kernel_size),3)
        else:
            self.kernel_size = self.largest_kernel_size
        # Load or create the basis
        basis = _parse_basis(self.basis_type, self.kernel_size, scales, self.effective_size, True, **self.kwargs)
        self.padding = self.kernel_size // 2

        # Create Kernel by multiplying kernel weights and basis
        kernel = basis(self.weight)
        # expand kernel
        kernel = kernel.permute(3, 0, 1, 2, 4, 5).contiguous()
        self.kernel = kernel
        kernel = kernel.view(-1, self.in_channels, self.scale_size,
                             self.kernel_size, self.kernel_size)

        # calculate padding
        if self.scale_size != 1:
            value = x.mean()
            x = F.pad(x, [0, 0, 0, 0, 0, self.scale_size - 1])

        if self.scale_size != 1:
            output = 0.0
            for i in range(self.scale_size):
                x_ = x[:, :, i:i + scales.size(dim=0)]
                # expand X
                B, C, S, H, W = x_.shape
                x_ = x_.permute(0, 2, 1, 3, 4).contiguous()
                x_ = x_.view(B, -1, H, W)
                if self.padding > 0:
                    x_ = F.pad(x_, 4 * [self.padding], mode=self.padding_mode)
                output += F.conv2d(x_, kernel[:, :, i], groups=S,
                                stride=self.stride, dilation=self.dilation)
        else:
            x_ = x
            # expand X
            B, C, S, H, W = x_.shape
            x_ = x_.permute(0, 2, 1, 3, 4).contiguous()
            x_ = x_.view(B, -1, H, W)
            if self.padding > 0:
                x_ = F.pad(x_, 4 * [self.padding], mode=self.padding_mode)
            output = F.conv2d(x_, kernel[:, :, 0], groups=S,
                            stride=self.stride, dilation=self.dilation)

        # squeeze output
        B, C_, H_, W_ = output.shape
        output = output.view(B, S, -1, H_, W_)
        output = output.permute(0, 2, 1, 3, 4).contiguous()
        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1, 1, 1)
        return output

    def extra_repr(self):
        s = '{in_channels}->{out_channels}, padding={padding}, padding_mode={padding_mode}'
        return s.format(**self.__dict__)

class SESConv_H_H_1x1(nn.Module):
    """The implementation was proposed in
    'Scale Equivariance Improves Siamese Tracking'
    pdf: https://arxiv.org/pdf/2007.09115v1.pdf
    """

    def __init__(self, in_channels, out_channels, scale_size=1, stride=1, num_scales=1, bias=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_size = scale_size
        self.stride = (1, stride, stride)
        self.num_scales = num_scales

        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, scale_size, 1, 1))
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)

    def forward(self, x):
        pad = self.scale_size - 1
        return F.conv3d(x, self.weight, padding=[pad, 0, 0], stride=self.stride)[:, :, pad:]

    def extra_repr(self):
        s = '{in_channels}->{out_channels} | scale_size={scale_size}'
        return s.format(**self.__dict__)


class SESMaxProjection(nn.Module):

    def forward(self, x):
        return x.max(2)[0]


def ses_max_projection(x):
    return x.max(2)[0]


# class SESMaxProjection(nn.Module):

#     def forward(self, x):
#         return x.max(2)

# class SESMaxProjectionPool(nn.Module):
#     def forward(self, x):
#         x_max_h_w = x.max(4)[0].max(3)[0]
#         return x_max_h_w.max(2)

# def ses_max_projection(x):
#     return x.argmax(2)[0], x.max(2)[0]
