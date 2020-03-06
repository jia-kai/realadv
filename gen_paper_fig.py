#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# so X11 would not be needed
import matplotlib
matplotlib.use('Agg')

from realadv.utils import ModelHelper, torch_as_npy
from realadv.find_edge_input import select_robust_examples
from realadv.find_edge_model import cw_loss_vec

import torch

import cv2
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
import os
import functools
import inspect
import argparse
import itertools
import pickle
import hashlib

class PercentValue:
    vnum = None

    def __init__(self, s, precision=0):
        self.precision = precision
        if isinstance(s, str):
            s = s.strip()
            assert s.endswith('%')
            self.vnum = float(s[:-1]) / 100
        else:
            self.vnum = float(s)

    def set_precision(self, prec):
        self.precision = prec
        return self

    def neg(self):
        """1 - self"""
        return PercentValue(1 - self.vnum, self.precision)

    def __repr__(self):
        val = self.vnum * 100
        fmt = '{{:.{}f}}\\%'.format(self.precision).format
        ret = fmt(val)
        if val != 0 and ret == fmt(0.0):
            return r'{:.1g}\%'.format(val)
        return ret


def cache_by_source(name):
    def wrap_caller(fn):
        fn_src = inspect.getsource(fn)
        sha256 = hashlib.sha256()
        sha256.update(fn_src.encode('utf-8'))
        fn_dig = sha256.hexdigest()
        @functools.wraps(fn)
        def wrapped(self):
            opath = self.out_dir / name
            if opath.exists():
                with opath.open('r') as fin:
                    get = fin.read()
                if get == fn_dig:
                    print(f'skip cached {fn.__name__}')
                    return
            fn(self)
            with opath.open('w') as fout:
                fout.write(fn_dig)
        return wrapped
    return wrap_caller


class GenPaperFig:
    impl_configs = {
        'C,M': ('cpu', 'features_chk', True, 'cpumm'),
        'C,C': ('cpu', 'features', True, 'cpuconv'),
        'G,M': ('cuda', 'features_chk', True, 'gpumm'),
        'G,C': ('cuda', 'features', True, 'gpuconv'),
        'G,CWG': ('cuda', 'features', False, 'gpuconvwg'),
    }
    data_dir = Path(__file__).resolve().parent / 'output'

    CIFAR10_CLASSES = [
        'airplane', 										
        'automobile', 										
        'bird', 										
        'cat', 										
        'deer', 										
        'dog', 										
        'frog', 										
        'horse', 										
        'ship', 										
        'truck',
    ]

    def __init__(self, args):
        self.out_dir = Path(args.out_dir)

    @cache_by_source('error_plot')
    def gen_error_plot(self):
        fpath = './data/mnist.pth'
        verify_fpath = './data/mnist-linf-0.1.csv'
        n = None

        def make_subplots(r, c):
            fig, ax = plt.subplots(r, c, figsize=(c * 4, r * 3))
            fig.tight_layout(pad=4.0)
            return fig, ax.flatten()

        fig_local, axes_local = make_subplots(3, 2)
        axes_local[-1].axis('off')
        axes_local = axes_local.flatten()
        fig_rela, axes_rela = make_subplots(2, 2)

        img = None
        for idx, (name, (device, ftr, use_stable, _)) in enumerate(
                self.impl_configs.items()):
            model = (ModelHelper.
                     create_with_load(fpath).
                     to(device).
                     use_unstable_conv(not use_stable))
            ftr = getattr(model, ftr)
            if img is None:
                rela_img = [i for i, _ in model.make_testset_loader()]
                rela_img = torch.cat(rela_img, dim=0)
                img, _ = select_robust_examples(model, device, verify_fpath)
                img = img[:1].detach()
                img.requires_grad = True
            img = img.to(device)

            f0 = ftr[:ftr.layer_take[0]]
            out_ref = ftr(img)
            if not idx:
                f0_out_ref = f0(rela_img)
            else:
                rela_img = rela_img.to(device)
                f0_out_ref = f0_out_ref.to(device)

            if n is None:
                (out_ref * torch.ones_like(out_ref)).sum().backward()
                gx = torch_as_npy(img.grad)
                n, c, h, w = np.unravel_index(np.argmax(np.abs(gx)), gx.shape)

            plot_x = np.arange(-1e-6, 1e-6, 1e-8)
            plot_y = []
            plot_y_rela = []
            timg = img.clone()
            with torch.no_grad():
                for dx in plot_x:
                    timg[n, c, h, w] = img[n, c, h, w] + dx
                    out = ftr(timg)
                    plot_y.append((out - out_ref).abs_().max().item())

            title = r'$\operatorname{NN_{%s}}$' % name
            ax = axes_local[idx]
            ax.plot(plot_x, plot_y)
            ax.plot(plot_x, np.abs(plot_x), ':', color='grey')
            ax.set_xlabel(r'$\delta$')
            ax.set_ylabel(r'$\| y - y_0 \|_{\infty}$')
            ax.set_title(title)
            ax.ticklabel_format(axis='both', style='sci', scilimits=(-4, 0))
            ax.grid()

            if idx:
                # relative error
                ax = axes_rela[idx - 1]
                f0_out: torch.Tensor = f0(rela_img)
                assert f0_out_ref.ndim == 2
                if f0_out.ndim == 4:
                    f0_out = f0_out.permute((0, 2, 3, 1))
                    f0_out = f0_out.reshape(f0_out.shape[0], -1)
                diff = torch_as_npy(
                    (f0_out - f0_out_ref).abs_().max(dim=1).values)
                ax.hist(diff, 50, weights=np.ones(diff.size)/diff.size,
                        alpha=0.8, log=True)
                ax.set_ylabel('Probability')
                ax.set_xlabel(r'$\| y_0 - y_1 \|_{\infty}$')
                ax.set_title((title[:-1] + ' - \operatorname{NN_{C,M}}$').
                             replace('NN', 'NN^{first}'))
                ax.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
                ax.grid()

        fig_local.savefig(str(self.out_dir / f'local-error.pdf'))
        fig_rela.savefig(str(self.out_dir / f'rela-error.pdf'))

    def gen_result_summary_table_nogap(self):
        idx_row = [
            'MNIST', 'CIFAR10'
        ]
        nr_model = [0, 0]
        nr_selected = [0, 0]
        ext2col = {}
        idx_col = [r'\#quasi-adv / \#tested']
        for idx, (name, (dev, ftr, stable, latex)) in enumerate(
                self.impl_configs.items()):
            idx_col.append(r'$\operatorname{NN_{%s}}$' % name)
            ext2col[f'.adv_nogap1_stable{int(stable)}_'
                    f'mm{int(ftr=="features_chk")}_'
                    f'cpu{int(dev=="cpu")}'] = idx + 1

        data = np.zeros((len(idx_row), len(idx_col)), dtype=np.int32)
        for row_i, dset in enumerate(idx_row):
            for i in (self.data_dir / dset.lower()).iterdir():
                if i.name.isdigit():
                    nr_selected[row_i] += 1
                    continue

                ext = os.path.splitext(i)[1]
                if ext == '.model':
                    nr_model[row_i] += 1
                    continue

                col = ext2col.get(ext)
                if col is not None:
                    data[row_i, col] += 1

        data = np.frompyfunc(str, 1, 1)(data)
        for i in range(len(idx_row)):
            data[i, 0] = f'{nr_model[i]} / {nr_selected[i]}'
        df = pd.DataFrame(data, index=idx_row, columns=idx_col)
        with self._open_outfile('result-defs.tex') as fout:
            self._write_latex_defs(fout, [
                ('nrModelMnist', nr_model[0]),
                ('nrModelCifar', nr_model[1]),
            ])
        with self._open_outfile('table-result-nogap.tex') as fout:
            latex: str = df.to_latex(
                escape=False,
                column_format='|'.join('l' + 'r'*len(idx_col)))
            fout.write(latex)

    def gen_result_summary_table_gap(self):
        idx_row = list(itertools.product(
            ['MNIST', 'CIFAR10'],
            [r'\# attack','min test acc']
        ))
        ext2col = {}
        idx_col = []
        for idx, (name, (dev, ftr, stable, latex)) in enumerate(
                self.impl_configs.items()):
            idx_col.append(r'$\operatorname{NN_{%s}}$' % name)
            ext2col[f'.adv_nogap0_stable{int(stable)}_'
                    f'mm{int(ftr=="features_chk")}_'
                    f'cpu{int(dev=="cpu")}'] = idx

        data = np.zeros((len(idx_row), len(idx_col)), dtype=object)
        for i in range(data.shape[0]):
            if i % 2:
                data[i] = [PercentValue(1, 2) for _ in range(data.shape[1])]
            else:
                data[i] = [0] * data.shape[1]
        for row_i, dset in enumerate(['mnist', 'cifar10']):
            for i in (self.data_dir / dset).iterdir():
                if (col := ext2col.get(os.path.splitext(i)[1])) is not None:
                    data[row_i * 2, col] += 1
                    with i.open('rb') as fin:
                        result = pickle.load(fin)
                    t = data[row_i * 2 + 1, col]
                    t.vnum = min(t.vnum, result['test_acc'])

        df = pd.DataFrame(data, index=pd.MultiIndex.from_tuples(idx_row),
                          columns=idx_col)
        with self._open_outfile('table-result-gap.tex') as fout:
            latex: str = df.to_latex(
                escape=False,
                column_format='|'.join('ll' + 'r'*len(idx_col)),
                multirow=True,
            )
            fout.write(latex)

    def gen_adv_imshow(self):
        col_titles = []
        ext2col = {}
        col2ext = []
        for idx, (name, (dev, ftr, stable, latex)) in enumerate(
                self.impl_configs.items()):
            col_titles.append(r'$\operatorname{NN_{%s}}$' % name)
            ext = (f'.adv_nogap1_stable{int(stable)}_'
                   f'mm{int(ftr=="features_chk")}_'
                   f'cpu{int(dev=="cpu")}')
            ext2col[ext] = idx
            col2ext.append(ext)

        col_test_nums = [[set() for _ in range(len(col_titles))]
                         for _ in range(2)]
        datasets = ['mnist', 'cifar10']
        for row_i, dset in enumerate(datasets):
            for i in (self.data_dir / dset).iterdir():
                if (col := ext2col.get(os.path.splitext(i)[1])) is not None:
                    col_test_nums[row_i][col].add(
                        int(i.name[:i.name.index('.')]))

        test_nums = [functools.reduce(lambda a, b: a & b, i, i[0])
                     for i in col_test_nums]

        fig_rows = sum(map(len, test_nums))
        fig_cols = len(col_titles) + 1
        fig, axes = plt.subplots(
            fig_rows, fig_cols,
            gridspec_kw={'wspace': 0.05, 'hspace': 0.4},
            squeeze=True,
            figsize=(fig_cols*1.6 + (fig_cols-1)*0.05,
                     fig_rows*1.6 + (fig_rows-1)*0.4))
        row_id = 0

        refimg = None
        def imshow(r, c, img, title, out_score):
            nonlocal refimg
            if c == 0:
                refimg = img
            else:
                assert np.abs(img - refimg).max() < [0.1, 2/255][dset_id]

            if r:
                title = ''
            else:
                title += '\n'
            img = np.squeeze(img, 0)
            img = np.transpose(img, (1, 2, 0))
            if img.shape[2] == 1:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax = axes[r, c]
            ax.imshow(img, interpolation=None)
            ax.axis('off')

            cls = np.argmax(out_score)
            cw = cw_loss_vec(out_score, cls)
            title = f'{title}{class_names[cls]}\n'r'$L_{\operatorname{CW}}$='
            if cw >= 0.01:
                title += f'{cw:.2f}'
            else:
                title += f'{cw:.2e}'
            ax.set_title(title)

        for dset_id, nums in enumerate(test_nums):
            if dset_id == 0:
                class_names = list(map(str, range(10))) # mnist
            else:
                class_names = self.CIFAR10_CLASSES

            for num in nums:
                for col, (title, ext) in enumerate(zip(col_titles, col2ext)):
                    if 'mm1' in ext:
                        adv_score_key = 'adv_out_score_mm'
                    else:
                        adv_score_key = 'adv_out_score'

                    with (self.data_dir / datasets[dset_id] /
                          f'{num}.model{ext}').open('rb') as fin:
                        data = pickle.load(fin)
                    if col == 0:
                        imshow(row_id, 0, data['inp'],
                               'verified robust', data['inp_out_score'])
                    imshow(row_id, col + 1, data['adv_inp'],
                           title, data[adv_score_key])
                row_id += 1

        fig.savefig(str(self.out_dir / 'adv-img.pdf'))

    def _open_outfile(self, name):
        return (self.out_dir / name).open('w')

    @classmethod
    def _write_latex_defs(cls, fout, defs):
        for k, v in defs:
            print(fr'\newcommand{{\{k}}}{{{v}}}', file=fout)

def set_plot_style():
    # see http://www.jesshamrick.com/2016/04/13/reproducible-plots/
    plt.style.use(['seaborn-white', 'seaborn-paper'])
    matplotlib.rcParams.update({
        'font.size': 10,
        'legend.fontsize': 'medium',
        'axes.titlesize': 'large',
        'axes.labelsize': 'large',
        'xtick.labelsize': 'large',
        'ytick.labelsize': 'large',
    })

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('out_dir', help='latex data output dir')
    args = parser.parse_args()
    set_plot_style()

    gp = GenPaperFig(args)
    for i in dir(gp):
        if i.startswith('gen_'):
            print(f'executing {i}() ...')
            getattr(gp, i)()

if __name__ == '__main__':
    main()
