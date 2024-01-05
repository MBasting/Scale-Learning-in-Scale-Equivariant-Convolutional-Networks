from enum import auto, Enum
import torch


class LearnMode(Enum):
    OFF = auto()
    DIRECT_SCALES = auto()
    RATIO = auto()
    RATIO_SQUARED = auto()
    SINGLE_RATIO = auto()
    SINGLE_RATIO_SQUARED = auto()

class ConvType(Enum):
    CNN_SIMPLE = auto()
    SRF = auto()
    # FLEXCONV = auto()
    # FLEXCONVSCALE = auto()
    SESN = auto()
    DISCO = auto()
    DISCO_FREE = auto()
    DISCO_DILATED = auto()
    # DIRECT = "SESN_Direct"
    # DISTANCE = "SESN_Distance"


def configure_learnable_scale_parameter(init_scale, learn_mode, nr_internal, basis_min_scale=1.0):
    print(
        "Initialization factor is either a float (expressed in ISR) or a list of floats"
    )
    if learn_mode == LearnMode.RATIO:
        if isinstance(init_scale, float) or isinstance(init_scale, int):
            init_scale = [init_scale**(1/(nr_internal - 1))] * (nr_internal-1)

        # NOTE: Input is a list of floats expressed in ISR
        scale_param = torch.tensor([scale - 1 for scale in init_scale])
        print(scale_param)
    if learn_mode == LearnMode.RATIO_SQUARED:
        if isinstance(init_scale, float) or isinstance(init_scale, int):
            init_scale = [init_scale**(1/(nr_internal - 1))] * (nr_internal-1)
        scale_param = torch.tensor([(scale - 1) ** 0.5 for scale in init_scale])
    elif learn_mode == LearnMode.SINGLE_RATIO:
        assert isinstance(init_scale, float) or isinstance(init_scale, int)
        # NOTE: Input is a float expressed in ISR
        scale_param = torch.tensor((init_scale**(1/(nr_internal-1)) - 1))

    elif learn_mode == LearnMode.SINGLE_RATIO_SQUARED:
        assert isinstance(init_scale, float) or isinstance(init_scale, int)
        # NOTE: Input is a float expressed in ISR
        scale_param = torch.tensor((init_scale**(1/(nr_internal-1)) - 1) ** 0.5)
    else:
        if isinstance(init_scale, float) or isinstance(init_scale, int):
            print("init scale is a float, converting ISR to scales")
            scale_param = torch.tensor([basis_min_scale * (init_scale ** (i / (nr_internal - 1))) for i in range(nr_internal)])
    return torch.nn.Parameter(scale_param, requires_grad=True)


def calculate_scales_parameter(
    conv_type: ConvType,
    learn_type: LearnMode,
    nr_internal_scales: int,
    learnable_basis_min : bool,
    init_scales,
    basis_min_scale, 
    decoupled_basis_min : bool = False,
    nr_layers : int = 1,
    **default_kwargs,
):
    # Scale Network
    if conv_type.value >= ConvType.SESN.value:
        print(
            "NOTE that at the moement we only support scales on STL_10 for SESN/DISCO"
        )
        # Determine if we need gradients or not based on learn type
        learnable = learn_type != LearnMode.OFF
        if learnable:
            scales_parameter = configure_learnable_scale_parameter(
                init_scales, learn_type,nr_internal_scales, basis_min_scale,
            )
            if decoupled_basis_min:
                basis_min_list = []
                for i in range(nr_layers):
                    basis_min_list.append(torch.nn.Parameter(torch.tensor([basis_min_scale], dtype=scales_parameter.dtype), requires_grad=learnable_basis_min))
                basis_min_scale = basis_min_list
            else:
                basis_min_scale = torch.nn.Parameter(torch.tensor([basis_min_scale], dtype=scales_parameter.dtype), requires_grad=learnable_basis_min and learn_type != LearnMode.DIRECT_SCALES)
        else:
            if isinstance(init_scales, float) or isinstance(init_scales, int):
                ISR = init_scales
                init_scales = [
                    (ISR ** (i / (nr_internal_scales - 1)))
                    for i in range(nr_internal_scales)
                ]
            # if not learnable need to multiply with basis_min_scale before passing to SESN
            # For dilated Disco with no intermediate scales we only take in scales that are integer values
            if conv_type.value >= ConvType.DISCO_FREE.value:
                init_scales = [s for s in init_scales if s.is_integer()]

            if conv_type == ConvType.SESN:
                init_scales = [
                    s * basis_min_scale for s in init_scales
                ]
            scales_parameter = init_scales
        return scales_parameter, basis_min_scale


def calculate_scales(learn_mode, scales_param, num_scales, basis_min_param):
    if learn_mode == LearnMode.RATIO:
        return torch.cumprod(
            torch.cat(
                [
                    basis_min_param,
                    torch.ones_like(scales_param) + scales_param,
                ]
            ),
            dim=0,
        )
    elif learn_mode == LearnMode.RATIO_SQUARED:
        return torch.cumprod(
            torch.cat(
                [
                    basis_min_param,
                    torch.ones_like(scales_param) + (scales_param**2),
                ]
            ),
            dim=0,
        )
    elif learn_mode == LearnMode.SINGLE_RATIO:
        return torch.cumprod(
            torch.cat(
                [
                    basis_min_param,
                    1
                    + torch.nn.functional.leaky_relu(scales_param).repeat(
                        num_scales - 1
                    ),
                ]
            ),
            dim=0,
        )
    elif learn_mode == LearnMode.SINGLE_RATIO_SQUARED:
        return torch.cumprod(
            torch.cat(
                [
                    basis_min_param,
                    1 + (scales_param**2).repeat(num_scales - 1),
                ]
            ),
            dim=0,
        )
    else:
        # incorporate basis_min_param
        if learn_mode == LearnMode.DIRECT_SCALES:
            return scales_param
        else:
            raise NotImplementedError(f"Learn mode {learn_mode} not implemented")
