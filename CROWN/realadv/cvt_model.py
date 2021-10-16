from .utils import (ModelHelper, iter_prod_range, assign_param, Flatten,
                    default_dataset_root, get_nr_correct, torch_as_npy)
from .cudnn_wrap import enforce_conv_fwd_algo

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import scipy.io as sio
import numpy as np

import weakref
import argparse
import lzma

class Conv2d(nn.Conv2d):
    owner = None

    def forward(self, x):
        # fix the 1-offset introduced during model converting
        ih, iw = x.shape[2:]
        ph, pw = self.padding
        sh, sw = self.stride
        kh, kw = self.weight.shape[2:]
        oh = (ih + ph*2 - kh) // sh + 1
        ow = (iw + pw*2 - kw) // sw + 1
        x = F.pad(x, [pw-1, pw+1, ph-1, ph+1])

        if self.owner._use_unstable_conv:
            algo_name = 'WINOGRAD_NONFUSED'
        else:
            algo_name = 'IMPLICIT_GEMM'

        impl = lambda: F.conv2d(x, self.weight, self.bias, (1, 1), (0, 0),
                                self.dilation, self.groups)

        if x.device.type == 'cuda':
            with enforce_conv_fwd_algo(algo_name):
                y = impl()
        else:
            y = impl()

        y = y[:, :, ::sh, ::sw]
        y = y[:, :, :oh, :ow].contiguous()
        return y


def read_mat_file(fpath):
    if fpath.endswith('.xz'):
        with lzma.open(fpath, 'rb') as fin:
            return sio.loadmat(fin)
    else:
        return sio.loadmat(fpath)

class MnistModel(nn.Module, ModelHelper):
    input_size = 28
    input_chl = 1

    _use_unstable_conv = False

    def __init__(self, features, features_chk):
        super().__init__()
        self.features = features
        self.features_chk = features_chk

    def use_unstable_conv(self, flag: bool):
        """whether to use the numerically unstable conv implementation"""
        self._use_unstable_conv = bool(flag)
        return self

    @classmethod
    def conv_weight_from_fc(cls, inp_shape, oc, weight: np.ndarray,
                            kern, stride, padding):
        """
        :param inp_shape: ``(ic, ih, iw)``
        weight: with ``(isize, osize)``, where image is in ``(h, w, c)`` format
        :return: kernel tensor, ``(OC, OH, OW)``
        """
        IC, IH, IW = inp_shape
        KH, KW = kern
        SH, SW = stride
        PH, PW = padding
        OC = oc
        OH = (IH + PH*2 - KH) // SH + 1
        OW = (IW + PW*2 - KW) // SW + 1

        filled = np.zeros((OC, IC, KH, KW), dtype=np.bool)
        kernel = np.zeros((OC, IC, KH, KW), dtype=np.float32)

        assert weight.shape == (IH*IW*IC, OH*OW*OC)
        wflat = weight.flatten()
        wflat_idx = 0

        for ih, iw, ic, oh, ow, oc in iter_prod_range(IH, IW, IC, OH, OW, OC):
            kval = wflat[wflat_idx]
            wflat_idx += 1
            # I don't know why there is an offset of 1 ... due to tf padding ?
            kh = ih - (oh * SH + 1 - PH)
            kw = iw - (ow * SW + 1 - PW)
            if min(kh, kw) < 0 or kh >= KH or kw >= KW:
                assert kval == 0, ((ih, iw, ic), (oh, ow, oc), (kh, kw), kval)
                continue
            if not filled[oc, ic, kh, kw]:
                kernel[oc, ic, kh, kw] = kval
                filled[oc, ic, kh, kw] = True
            else:
                assert kernel[oc, ic, kh, kw] == kval
        assert np.all(filled)
        return kernel, (OC, OH, OW)

    @classmethod
    def make_conv(cls, params, inp_shape, oc, name, kern, stride, padding):
        weight = params.pop(f'{name}/weight')
        bias = params.pop(f'{name}/bias')
        kern_w, (_, oh, ow) = cls.conv_weight_from_fc(
            inp_shape, oc, weight, kern, stride, padding)
        bias = bias.reshape(oh * ow, oc)
        assert np.all(bias.max(axis=0) == bias.min(axis=0))

        opr = Conv2d(inp_shape[0], oc, kern, stride, padding)
        assign_param(opr.weight, kern_w)
        assign_param(opr.bias, bias[0])
        return opr

    @classmethod
    def make_fc(cls, params, name, inp_spatial=None):
        weight = params.pop(f'{name}/weight')
        bias = params.pop(f'{name}/bias')
        ic, oc = weight.shape
        if inp_spatial is not None:
            spih, spiw = inp_spatial
            spic = ic // (spih * spiw)
            weight = weight.reshape(spih, spiw, spic, oc)
            weight = np.transpose(weight, (2, 0, 1, 3))
            weight = weight.reshape(ic, oc)
        opr = nn.Linear(ic, oc)
        assign_param(opr.weight, weight.T)
        assign_param(opr.bias, bias.flatten())
        return opr

    @classmethod
    def from_mat(cls, fpath):
        isize = cls.input_size
        ichl = cls.input_chl
        params = {k: v for k, v in read_mat_file(fpath).items()
                  if not k.startswith('_')}
        params_chk = params.copy()

        conv_setting = dict(kern=(5, 5), stride=(2, 2), padding=(2, 2))
        features = nn.Sequential(
            cls.make_conv(params, (ichl, isize, isize), 16, 'fc1',
                          **conv_setting),
            nn.ReLU(),
            cls.make_conv(params, (16, isize//2, isize//2), 32, 'fc2',
                          **conv_setting),
            nn.ReLU(),
            Flatten(),
            cls.make_fc(params, 'fc3', (isize//4, isize//4)),
            nn.ReLU(),
            cls.make_fc(params, 'softmax'),
        )
        features.layer_take = [1, 3, 6]
        features_chk = nn.Sequential(
            Flatten(),
            cls.make_fc(params_chk, 'fc1', (isize, isize)),
            nn.ReLU(),
            cls.make_fc(params_chk, 'fc2'),
            nn.ReLU(),
            cls.make_fc(params_chk, 'fc3'),
            nn.ReLU(),
            cls.make_fc(params_chk, 'softmax'),
        )
        features_chk.layer_take = [2, 4, 6]
        return cls(features, features_chk)._on_load_from_file()

    @classmethod
    def make_testset_loader(cls, batch_size=256):
        dataset = torchvision.datasets.MNIST(
            root=default_dataset_root(), train=False, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
            ]))
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False)
        return loader

    def forward(self, x):
        return self.features(x)

    def forward_chk(self, x):
        return self.features_chk(x)

    def _on_load_from_file(self):
        for i in self.features:
            if isinstance(i, Conv2d):
                # bypass pytorch hooks
                i.__dict__['owner'] = weakref.proxy(self)
        return self


class Cifar10Model(MnistModel):
    input_size = 32
    input_chl = 3

    @classmethod
    def make_testset_loader(cls, batch_size=256):
        dataset = torchvision.datasets.CIFAR10(
            root=default_dataset_root(), train=False, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
            ]))
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False)
        return loader


def print_test_err(model, device):
    testloader = model.make_testset_loader()
    nr_test_correct = 0
    nr_test_correct_chk = 0
    nr_test_tot = 0
    err = np.zeros(len(model.features.layer_take) + 1)
    for inputs, labels in testloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        for idx, (i, j) in enumerate(zip(model.features.layer_take,
                                         model.features_chk.layer_take)):
            o1 = torch_as_npy(model.features[:i](inputs))
            o2 = torch_as_npy(model.features_chk[:j](inputs))
            if o1.ndim == 4:
                n, c, h, w = o1.shape
                o2 = np.transpose(o2.reshape(n, h, w, c), (0, 3, 1, 2))
            err[idx] = max(err[idx], np.abs(o1 - o2).max())

        o1 = model(inputs)
        o2 = model.forward_chk(inputs)

        nr_test_correct += get_nr_correct(o1, labels)
        nr_test_correct_chk += get_nr_correct(o2, labels)
        nr_test_tot += inputs.size(0)
        err[-1] = max(err[-1], (o1 - o2).abs_().max().item())
    print(f'test acc: {nr_test_correct/nr_test_tot*100:.2f}% '
          f'chk={nr_test_correct_chk/nr_test_tot*100:.2f}% err={err}')

def main():
    parser = argparse.ArgumentParser(
        description='convert weight matrix saved by MadryLab/relu_stable to a '
        'pytorch model'
    )
    parser.add_argument('input')
    parser.add_argument('output')
    parser.add_argument('model', choices=['mnist', 'cifar10'])
    args = parser.parse_args()

    model_dict = {
        'mnist': MnistModel,
        'cifar10': Cifar10Model,
    }
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model_dict[args.model].from_mat(args.input).to(device)
    model.save_to_file(args.output)
    print_test_err(model, device)
