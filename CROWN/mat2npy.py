#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from realadv.cvt_model import read_mat_file

import pickle
import argparse
import itertools

def main():
    parser = argparse.ArgumentParser(
        description='convert .mat weights to .npy weights to be read by CROWN',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input')
    parser.add_argument('output')
    args = parser.parse_args()

    wdict: dict = read_mat_file(args.input)

    wlist = []
    for i in itertools.chain(itertools.product(['fc'], '123'), ['softmax']):
        if isinstance(i, tuple):
            i = ''.join(i)
        wlist.append(wdict.pop(f'{i}/weight'))
        wlist.append(wdict.pop(f'{i}/bias').flatten())

    print([i.shape for i in wlist])
    assert all(i.startswith('_') for i in wdict.keys())

    with open(args.output, 'wb') as fout:
        pickle.dump(wlist, fout, protocol=4)    # to be loaded by py3.5


if __name__ == '__main__':
    main()
