#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

import argparse
import sys
import pickle

def get_kern(rng, args):
    kern = rng.normal(
        scale=0.2,
        size=(1, 1, args.kern_size, args.kern_size)).astype(np.float32)
    out_size = args.input_size - args.kern_size + 1
    weight = np.zeros((args.input_size**2, out_size**2), dtype=np.float32)

    for oh in range(out_size):
        for ow in range(out_size):
            for kh in range(args.kern_size):
                for kw in range(args.kern_size):
                    ih = oh + kh
                    iw = ow + kw
                    if max(ih, iw) < args.input_size:
                        ipos = ih * args.input_size + iw
                        opos = oh * out_size + ow
                        weight[ipos, opos] = kern[0, 0, kh, kw]

    return kern, weight

def make_tf_model(args, midsize, weights):
    from tensorflow.contrib.keras.api.keras.models import Sequential
    from tensorflow.contrib.keras.api.keras.layers import (
        Dense, Activation, Flatten)
    model = Sequential()
    model.add(Flatten(input_shape=(args.input_size, args.input_size, 1)))
    model.add(Dense(midsize))
    model.add(Activation('relu'))
    for i in range(args.id_layers):
        model.add(Dense(args.id_size))
        model.add(Activation('relu'))
    model.add(Dense(1))
    model.add(Activation('relu'))
    model.add(Dense(2))

    expect_shapes = [i.shape for i in model.get_weights()]
    get_shapes = [i.shape for i in weights]
    assert expect_shapes == get_shapes, (expect_shapes, get_shapes)
    model.set_weights(weights)
    return model

def gen_id_layers(rng, args, x0, w0, b0):
    x0 = x0.flatten()
    ws = []
    bs = []
    wall = w0
    ball = b0
    for i in range(args.id_layers):
        wi = rng.normal(loc=0.02, scale=0.06,
                        size=(wall.shape[1], args.id_size))
        wi = wi.astype(np.float32)
        wall = wall @ wi
        ball = ball @ wi
        ws.append(wi)

        bi = []
        for j in range(args.id_size):
            wj = wall[:, j]
            bj = ball[j]
            xmin = x0 - np.sign(wj) * args.eps
            bi.append(-(xmin @ wj + bj) + 1e-4)
        bi = np.ascontiguousarray(bi, dtype=np.float32).flatten()
        bs.append(bi)
        ball += bi

    return ws, bs

def get_max_out(args, x0, ws, bs):
    assert len(ws) == len(bs)
    x0 = x0.flatten()
    wall = ws[0]
    ball = bs[0]
    for w, b in zip(ws[1:], bs[1:]):
        wall = wall @ w
        ball = ball @ w + b
    assert wall.shape[1] == 1, [i.shape for i in ws]
    xmax = x0 + np.sign(wall.flatten()) * args.eps
    return float(xmax @ wall + ball)

def main():
    parser = argparse.ArgumentParser(
        description='generate a network and an input that can be '
        'verified by CROWN')
    parser.add_argument('crown_path')
    parser.add_argument('--eps', type=float, default=0.001)
    parser.add_argument('--x0-center', type=float, default=0.1)
    parser.add_argument('--x0-scale', type=float, default=0.01)
    parser.add_argument('--input-size', type=int, default=13)
    parser.add_argument('--kern-size', type=int, default=5)
    parser.add_argument('--id-layers', type=int, default=2,
                        help='number of layers with always-active relus')
    parser.add_argument('--id-size', type=int, default=16)
    parser.add_argument('--seed', type=int, default=92702104)
    parser.add_argument('output', help='output weight file')
    args = parser.parse_args()

    rng = np.random.RandomState(args.seed)

    model_inp = rng.normal(args.x0_center, args.x0_scale,
                           size=(args.input_size, args.input_size, 1))
    model_inp = np.clip(model_inp, args.eps, 1 - args.eps).astype(np.float32)

    # all relu units in the first layer are active
    kern, kern_mm = get_kern(rng, args)
    bias0_v = (-(model_inp.flatten() @ kern_mm).min() +
               np.abs(kern).sum() * args.eps + 1e-6)
    bias0 = np.ascontiguousarray(bias0_v, dtype=np.float32)
    bias0_mm = np.empty(kern_mm.shape[1], dtype=np.float32)
    bias0_mm[:] = bias0_v

    # the relu in the id layers are always active
    id_layer_w, id_layer_b = gen_id_layers(
        rng, args, model_inp, kern_mm, bias0_mm)

    # w1 relu is always inactive
    w1 = rng.normal(
        loc=0.3,
        scale=0.1,
        size=(args.id_size, 1)).astype(np.float32)
    b1 = np.zeros(1, dtype=np.float32)

    w2 = np.zeros((1, 2), dtype=np.float32)
    w2[:, 0] = 1
    b2 = np.zeros(2, dtype=np.float32)
    b2[1] = 1e-7

    sys.path.append(args.crown_path)
    from get_bounds_ours import compute_worst_bound

    def get_bound(b):
        b1[:] = b
        all_w = [kern_mm] + id_layer_w + [w1, w2]
        all_b = [bias0_mm] + id_layer_b + [b1, b2]
        all_weights = []
        for i, j in zip(all_w, all_b):
            all_weights.extend([i, j])
        model = make_tf_model(args, kern_mm.shape[1], all_weights)
        model_pred = np.squeeze(model.predict(model_inp[None]), axis=0)
        return compute_worst_bound(
            [np.ascontiguousarray(x.T) for x in all_w], all_b,
            1, 0,
            model_inp, model_pred, None,
            eps=args.eps
        )

    b1_init = -get_max_out(args, model_inp,
                           [kern_mm] + id_layer_w + [w1],
                           [bias0_mm] + id_layer_b + [b1])
    lower = b1_init - 1e-4
    higher = b1_init + 1e-4
    assert get_bound(lower)[0] > 0
    assert get_bound(higher)[0] <= 0
    while higher - lower > 1e-8:
        mid = (higher + lower) / 2
        bound = get_bound(mid)[0]
        if bound <= 0:
            higher = mid
        else:
            lower = mid

    print('bound init:', get_bound(b1_init))
    bound = get_bound(lower)
    print(b1_init, b1_init - lower, bound)
    assert bound[0] > 0

    other_weights = []
    for i, j in zip(id_layer_w, id_layer_b):
        other_weights.extend([i, j])
    other_weights.extend([w1, b1, w2, b2])
    with open(args.output, 'wb') as fout:
        obj = {'input': model_inp,
               'eps': args.eps,
               'weights': [kern, bias0] + other_weights,
               'weights_mm': [kern_mm, bias0_mm] + other_weights,
               }
        pickle.dump(obj, fout, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    main()
