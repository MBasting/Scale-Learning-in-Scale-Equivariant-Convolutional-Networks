from dataset_factory import ChoiceDataset
from model import Network
from disco.mnist_disco_models import mnist_disco
from disco.mnist_models import mnist_hermite
from disco.stl_se_models import wrn_learnable
from disco.stl_wrn import wrn_CNN
from utils.model_utils import ConvType


def factory(cfg):
    # Check if dataset is STL-10 or not:
    if cfg.dataset.d_index == ChoiceDataset.STL_10.value or cfg.dataset.d_index == ChoiceDataset.CIFAR_10.value:
        if cfg.model.conv_index == ConvType.CNN_SIMPLE.value:
            model = wrn_CNN(**cfg.model)
        elif cfg.model.conv_index == ConvType.SESN.value or cfg.model.conv_index == ConvType.DISCO.value:
            model = wrn_learnable(**cfg.model)
        else:
            return NotImplementedError  
    else:

        def scale_hidden(x, s):
            return [int(y * s) for y in x]

        cfg.model.base_kernel_size = cfg.model.get(
            "base_kernel_size", cfg.model.eff_size
        )

        # Hidden Channel count Calculation
        cfg.model.hidden_channels = scale_hidden(
            cfg.model.hidden_channels, cfg.model.hidden_channels_scale
        )

        # If this flag is raised we directly use the models defined for in the Disco Repo
        if cfg.model.full_original and cfg.model.conv_index >= ConvType.SESN.value:
            if cfg.model.conv_index == ConvType.SESN.value:
                model = mnist_hermite(
                    basis_min_scale=1.5,
                    basis_mult=1.4,
                    C1=cfg.model.hidden_channels[0],
                    C2=cfg.model.hidden_channels[1],
                    C3=cfg.model.hidden_channels[2],
                    linear_hidden=cfg.model.linear_hidden_class,
                )
            elif cfg.model.conv_index == ConvType.DISCO.value:
                model = mnist_disco(
                    1.9,
                    "disco/precalculated_basis/",
                    1.5,
                    C1=cfg.model.hidden_channels[0],
                    C2=cfg.model.hidden_channels[1],
                    C3=cfg.model.hidden_channels[2],
                    linear_hidden=cfg.model.linear_hidden_class,
                )
            else:
                return NotImplementedError()
        else:
            # Create Model
            model = Network(
                **cfg.dataset,
                **cfg.model
            )
    return model
