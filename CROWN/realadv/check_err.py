from .utils import ModelHelper, torch_as_npy
from .cvt_model import print_test_err
from .cudnn_wrap import enforce_conv_fwd_algo

import numpy as np
import torch
import torch.nn.functional as F

import argparse

plt = None

def check(args, model, model_sub, device):
    err_xs = []
    err_ys = []
    for inputs, _ in model.make_testset_loader():
        inputs = inputs.to(device)
        model.use_unstable_conv(False)
        y0 = torch_as_npy(model_sub(inputs))
        if args.epsilon:
            inputs_pert = (torch.
                           empty_like(inputs).
                           uniform_(-args.epsilon, args.epsilon).
                           add_(inputs).
                           clamp_(-1, 1))
            y1 = model_sub(inputs_pert)
        else:
            model.use_unstable_conv(True)
            y1 = model_sub(inputs)
        y1 = torch_as_npy(y1)

        err_xs.extend(y0.flat)
        err_ys.extend((y0 - y1).flat)

    err_xs = np.ascontiguousarray(err_xs, dtype=np.float32)
    err_ys = np.ascontiguousarray(err_ys, dtype=np.float32)
    print(f'err: L1={np.abs(err_ys).mean()} max={np.abs(err_ys).max()}')
    idx = np.random.choice(err_xs.size, args.size, replace=False)
    fig, axs = plt.subplots(2)
    axs[0].plot(err_xs[idx], err_ys[idx], '.')
    axs[1].hist(err_ys, bins=50, log=True)
    if args.output:
        plt.savefig(args.output)
    else:
        plt.show()


def find_conv_patterns(model, device):
    conv_w = model.features[0].weight
    assert conv_w.shape[2:] == (5, 5)
    # kernel name is cudnn::winograd_nonfused::winogradForwardOutput9x9_5x5
    max_err = 0
    max_err_inp = None
    max_err_label = None
    for inputs, labels in model.make_testset_loader():
        inputs = inputs.to(device)
        with enforce_conv_fwd_algo('IMPLICIT_GEMM'):
            out_ref = F.conv2d(inputs, conv_w)
        with enforce_conv_fwd_algo('WINOGRAD_NONFUSED'):
            out_got = F.conv2d(inputs, conv_w)
        cur_err = (out_ref - out_got).abs_()
        cur_err_max = cur_err.max().item()

        if cur_err_max > max_err:
            max_err = cur_err_max
            idx = np.argmax(torch_as_npy(cur_err), axis=None)
            idx = np.unravel_index(idx, tuple(cur_err.shape))
            n, c, h, w = idx
            max_err_inp = inputs[n:n+1].clone()
            max_err_label = labels[n:n+1].clone()
    print(max_err)
    return max_err_inp, max_err_label

def main():
    parser = argparse.ArgumentParser(
        description='check numerical error of unstable algorithm',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('model')
    parser.add_argument('-o', '--output', help='plot output')
    parser.add_argument('--size', type=int, default=20000,
                        help='number of points to sample for plotting')
    parser.add_argument('-t', '--type', required=True,
                        choices=['first', 'score'])
    parser.add_argument('-e', '--epsilon', type=float,
                        help='add input perturbation instead of '
                        'using unstable conv')
    args = parser.parse_args()
    import matplotlib
    if args.output:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt_
    global plt
    plt = plt_

    model = ModelHelper.create_with_load(args.model)
    device = 'cuda'
    model = model.to(device)
    print_test_err(model, device)

    sub_map = {
        'first': model.features[0],
        'score': model
    }
    with torch.no_grad():
        check(args, model, sub_map[args.type], device)
        if args.type == 'first':
            find_conv_patterns(model, device)

if __name__ == '__main__':
    main()
