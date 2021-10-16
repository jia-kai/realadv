#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from realadv.utils import torch_as_npy
import torch

import argparse
import pickle

def main():
    parser = argparse.ArgumentParser(
        description='convert model files to be used by check.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input')
    parser.add_argument('output')
    args = parser.parse_args()

    rec = torch.load(args.input, map_location='cpu')
    out = {'input': torch_as_npy(rec['input']),
           'alpha': rec['robust']['alpha'],
           'label': int(rec['label'].item()),
           'eps': rec['eps'],
           }

    with open(args.output, 'wb') as fout:
        pickle.dump(out, fout, protocol=4)    # to be loaded by py3.5


if __name__ == '__main__':
    main()
