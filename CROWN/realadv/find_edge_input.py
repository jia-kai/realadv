from .utils import ModelHelper, get_nr_correct, torch_as_npy
from .mip_verify import MIPVerify

import torch
import numpy as np
import pandas as pd

import threading
import argparse
import pprint
import os
from multiprocessing.pool import ThreadPool

def eval_on_test(model, device):
    nr_test_correct = 0
    nr_test_correct_chk = 0
    nr_test_tot = 0
    err = 0
    for inputs, labels in model.make_testset_loader():
        inputs = inputs.to(device)
        labels = labels.to(device)
        o1 = model(inputs)
        o2 = model.forward_chk(inputs)
        err = max(err, (o1 - o2).abs_().max().item())
        nr_test_correct += get_nr_correct(o1, labels)
        nr_test_correct_chk += get_nr_correct(o2, labels)
        nr_test_tot += inputs.size(0)
    print(f'test acc: unstable={nr_test_correct/nr_test_tot*100:.2f}% '
          f'stable={nr_test_correct_chk/nr_test_tot*100:.2f}%; err={err}')

def select_robust_examples(model, device, fpath):
    dt = pd.read_csv(fpath)
    dt['SampleNumber'] -= 1
    dt['PredictedIndex'] -= 1
    testloader = model.make_testset_loader(batch_size=len(dt))
    inputs, labels = next(iter(testloader))
    sel_inputs = []
    sel_labels = []
    all_idx = set()
    for _, row in dt.iterrows():
        idx = row['SampleNumber']
        if idx in all_idx:
            continue
        all_idx.add(idx)
        if (labels[idx] == row['PredictedIndex'] and
                row['SolveStatus'] == 'InfeasibleOrUnbounded'):
            sel_inputs.append(inputs[idx:idx+1])
            sel_labels.append(labels[idx:idx+1])
    sel_inputs = torch.cat(sel_inputs, 0).to(device)
    sel_labels = torch.cat(sel_labels, 0).to(device)
    print(f'robust examples: {sel_inputs.shape}/{len(all_idx)}')
    return sel_inputs, sel_labels

class UndecidableResult(Exception):
    pass


def find_edge_robust(args, mip_verify_inst, model, orig_input, label):
    flog = open(args.output + '.log', 'w')
    lock = threading.RLock()

    def lprint(*msg):
        with lock:
            print(*msg)
            print(*msg, file=flog)
            flog.flush()

    save_state = {
        'input': orig_input,
        'label': label,
        'eps': args.epsilon,
        'model_path': args.model,
        'mat_model_path': args.mat_model,
    }

    max_adv_alpha = float('-inf')
    min_robust_alpha = float('inf')

    def submit_verify_result(v):
        nonlocal max_adv_alpha
        nonlocal min_robust_alpha
        alpha = v['alpha']
        if v['status_known']:
            updated = False
            if not v['robust'] and alpha > max_adv_alpha:
                max_adv_alpha = alpha
                save_state['adv'] = v.copy()
                updated = True
            elif v['robust'] and alpha < min_robust_alpha:
                min_robust_alpha = alpha
                save_state['robust'] = v.copy()
                updated = True
            if updated:
                torch.save(save_state, args.output)
                lprint('output file updated: '
                       f'{min_robust_alpha=} {max_adv_alpha=}')

            # remove huge values from log
            v.pop('PerturbationValue', None)
            v.pop('PerturbedInputValue', None)

        lprint(f'verify for {alpha=}: {pprint.pformat(v)}')
        if not v['status_known']:
            raise UndecidableResult()

    def check_robust(alpha, worker_number=0):
        inp = orig_input * alpha
        scores = torch_as_npy(model.features_chk(inp[np.newaxis]))
        pred = np.argmax(scores)
        if pred != label:
            v = {
                'robust': False,
                'status_known': True,
            }
        else:
            v = mip_verify_inst[worker_number](inp, label)

        v['alpha'] = alpha
        v['model_scores'] = scores
        with lock:
            submit_verify_result(v)
        return v['robust']

    def search_bin():
        left = args.left
        right = args.right
        assert check_robust(right)
        assert not check_robust(left)
        while right - left > args.target_resolution:
            alpha = (left + right) * 0.5
            lprint(f'binary search: {left} {right}: '
                   f'gap={right-left:.2e} mid={alpha}')
            if check_robust(alpha):
                right = alpha
            else:
                left = alpha

    def search_grid(left, right):
        local_lock = threading.Lock()
        robust = []
        non_robust = []
        unknown = []
        available_slots = list(range(args.workers))

        grid = int(min(args.grid,
                       np.ceil((right - left) / args.target_resolution)))

        def work(i):
            with local_lock:
                worker = available_slots.pop()

            k = (i + 1) / (grid + 1)
            x = left * (1 - k) + right * k

            try:
                if check_robust(x, worker):
                    dst = robust
                else:
                    dst = non_robust
            except UndecidableResult:
                dst = unknown

            with local_lock:
                available_slots.append(worker)
                dst.append(x)
                lprint(f'grid: {robust=} {non_robust=} {unknown=}')

        with ThreadPool(args.workers) as pool:
            try:
                pool.map(work, range(grid))
            except KeyboardInterrupt:
                for i in mip_verify_inst:
                    i.stop()
                raise

    def work():
        try:
            return search_bin()
        except UndecidableResult:
            pass

        assert 'robust' in save_state
        if 'adv' in save_state:
            left = save_state['adv']['alpha']
        else:
            left = 0
        search_grid(left, save_state['robust']['alpha'])

    with torch.no_grad():
        try:
            return work()
        finally:
            flog.close()
            for i in mip_verify_inst:
                i.stop()

def main():
    parser = argparse.ArgumentParser(
        description='construct robust inputs that are close to being '
        'non-robust',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('model', help='converted pytorch model')
    parser.add_argument('mat_model', help='original model')
    parser.add_argument('verify_csv',
                        help='MadryLab/relu_stable verification result')
    parser.add_argument('--index', type=int, default=0,
                        help='select the base input index')
    parser.add_argument('-e', '--epsilon', type=float, required=True,
                        help='adv Linf bound')
    parser.add_argument('-o', '--output',
                        help='output file name')
    parser.add_argument('--time-limit', type=int, default=360,
                        help='time limit for MIPVerify')
    parser.add_argument('--left', type=float, default=0,
                        help='lower bound for search')
    parser.add_argument('--right', type=float, default=1,
                        help='upper bound for search')
    parser.add_argument('--grid', type=int, default=16,
                        help='max number of slices in grid search')
    parser.add_argument('--target-resolution', type=int, default=1e-7,
                        help='target resolution of search')
    parser.add_argument('--workers', type=int, default=4,
                        help='number of MIPVerify worker processes')
    args = parser.parse_args()

    if args.output and os.path.exists(args.output + '.finished'):
        print(f'skip finished {args.output}')
        return

    model = ModelHelper.create_with_load(args.model)

    mip_verify_inst = [
        MIPVerify(args.mat_model, (model.input_size, model.input_chl),
                  args.time_limit, args.epsilon)
        for _ in range(args.workers)
    ]

    device = 'cuda'
    model = model.to(device)
    model.use_unstable_conv(True)
    eval_on_test(model, device)

    robust_inputs, robust_labels = select_robust_examples(
        model, device, args.verify_csv)

    rout = model(robust_inputs)
    assert get_nr_correct(rout, robust_labels) == robust_labels.size(0)
    print('error on robust:',
          (rout - model.forward_chk(robust_inputs)).abs_().max().item())

    if args.output:
        find_edge_robust(
            args, mip_verify_inst, model,
            robust_inputs[args.index], robust_labels[args.index])
        with open(args.output + '.finished', 'w') as fout:
            pass

if __name__ == '__main__':
    main()
