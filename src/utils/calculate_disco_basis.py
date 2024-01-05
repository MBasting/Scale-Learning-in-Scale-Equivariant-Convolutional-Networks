"""
This file is a part of the official implementation of
1) "DISCO: accurate Discrete Scale Convolutions"
    by Ivan Sosnovik, Artem Moskalev, Arnold Smeulders, BMVC 2021
    arxiv: https://arxiv.org/abs/2106.02733

2) "How to Transform Kernels for Scale-Convolutions"
    by Ivan Sosnovik, Artem Moskalev, Arnold Smeulders, ICCV VIPriors 2021
    pdf: https://openaccess.thecvf.com/content/ICCV2021W/VIPriors/papers/Sosnovik_How_To_Transform_Kernels_for_Scale-Convolutions_ICCVW_2021_paper.pdf

---------------------------------------------------------------------------

MIT License. Copyright (c) 2021 Ivan Sosnovik, Artem Moskalev
"""

import os
import time
from argparse import ArgumentParser

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn

from disco.basis import ApproximateProxyBasis
from disco.basis.disco_ref import get_basis_filename
from utils import loaders
from utils.train_utils import train_equi_loss


#########################################
# arguments
#########################################
parser = ArgumentParser()
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--epochs", type=int, default=40)

parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--lr_steps", type=int, nargs="+", default=[20, 30])
parser.add_argument("--lr_gamma", type=float, default=0.1)

parser.add_argument("--cuda", action="store_true", default=True)

# basis hyperparameters
parser.add_argument("--free_form", type=bool, default=False)
parser.add_argument("--basis_nr_scales", type=int, default=4)
parser.add_argument("--basis_max_scale", type=float, default=2.0)
# parser.add_argument('--basis_size', type=int, default=15)
parser.add_argument("--basis_effective_size", type=int, default=7)
# parser.add_argument('--basis_scales', type=float, nargs='+', default=[2**(i/3) for i in range(0, 4)])
parser.add_argument(
    "--basis_save_dir", type=str, default="../disco/precalculated_basis"
)


args = parser.parse_args()

print("Args:")
for k, v in vars(args).items():
    print("  {}={}".format(k, v))

print(flush=True)
print(args)


def calculate_disco_basis(
    basis_save_dir,
    basis_effective_size,
    batch_size=64,
    epochs=40,
    lr=0.001,
    lr_steps=[20, 30],
    lr_gamma=0.1,
    cuda=True,
    free_form=False,
    basis_nr_scales=4,
    basis_max_scale=2.0,
):
    #########################################
    # Data
    #########################################
    loader = loaders.random_loader(batch_size)

    basis_scales = [basis_max_scale ** (i / 3) for i in range(0, basis_nr_scales)]

    def get_basis_size(eff_size, s):
        return round(eff_size * s) // 2 * 2 + 1

    basis_size = get_basis_size(basis_effective_size, max(basis_scales))
    print(basis_scales, basis_size)

    print("Dataset:")
    print(loader.dataset)

    if free_form:
        print("Free Form")
        basis_scales = [s for s in basis_scales if s.is_integer()]
    print(basis_scales)
    #########################################
    # Model
    #########################################
    basis = ApproximateProxyBasis(
        size=basis_size, scales=basis_scales, effective_size=basis_effective_size
    )

    print("\nBasis:")
    print(basis)
    print()

    use_cuda = cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Device: {}".format(device))

    if use_cuda:
        cudnn.enabled = True
        cudnn.benchmark = True
        print("CUDNN is enabled. CUDNN benchmark is enabled")
        basis.cuda()

    print(flush=True)

    #########################################
    # Paths
    #########################################

    save_basis_postfix = get_basis_filename(
        size=basis_size, effective_size=basis_effective_size, scales=basis_scales
    )
    save_basis_path = os.path.join(basis_save_dir, save_basis_postfix)
    print("Basis path: ", save_basis_path)
    print()

    if not os.path.isdir(basis_save_dir):
        os.makedirs(basis_save_dir)

    #########################################
    # optimizer
    #########################################
    parameters = filter(lambda x: x.requires_grad, basis.parameters())
    params = list(parameters)
    if len(params) == 0:
        print("No trainable params")
        torch.save(basis.get_basis().cpu(), save_basis_path)
        return

    optimizer = optim.Adam(params, lr=lr)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, lr_steps, lr_gamma)

    #########################################
    # Training
    #########################################

    print("\nTraining\n" + "-" * 30)
    start_time = time.time()
    best_loss = float("inf")

    for epoch in range(epochs):
        loss = train_equi_loss(basis, optimizer, loader, device)
        print(
            "Epoch {:3d}/{:3d}| Loss: {:.2e}".format(epoch + 1, epochs, loss),
            flush=True,
        )
        if loss < best_loss:
            best_loss = loss

            with torch.no_grad():
                torch.save(basis.get_basis().cpu(), save_basis_path)

        lr_scheduler.step()

    print("-" * 30)
    print("Training is finished")
    print("Best Loss: {:.2e}".format(best_loss), flush=True)
    end_time = time.time()
    elapsed_time = end_time - start_time
    time_per_epoch = elapsed_time / epochs

    print("Total Time Elapsed: {:.2f}".format(elapsed_time))
    print("Time per Epoch: {:.2f}".format(time_per_epoch))


calculate_disco_basis(
    args.basis_save_dir,
    args.basis_effective_size,
    args.batch_size,
    args.epochs,
    args.lr,
    args.lr_steps,
    args.lr_gamma,
    args.cuda,
    args.free_form,
    args.basis_nr_scales,
    args.basis_max_scale,
)
