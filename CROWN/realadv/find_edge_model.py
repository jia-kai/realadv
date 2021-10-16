from .utils import ModelHelper, torch_as_npy
from .mip_verify import MIPVerify

import numpy as np
import torch

import argparse
import pprint
import traceback
import os
import threading

lprint = None
# print to log, to be initialized in main

def cw_loss_vec(score: np.ndarray, label):
    score = score.copy()
    t = score[label]
    score[label] = float('-inf')
    return t - score.max()

def pformat_verify(v: dict):
    v = v.copy()
    v.pop('PerturbationValue', None)
    v.pop('PerturbedInputValue', None)
    return v

def preprocess(model, device, record):
    """sanity check and return ``(record, robust input, confidence)``"""
    orig_inp: torch.Tensor = record['input'].to(device)
    assert orig_inp.ndim == 3
    orig_inp = orig_inp[np.newaxis]

    assert 'robust' in record
    adv_inp = None
    if 'adv' in record:
        adv_inp = record['adv'].get('PerturbedInputValue', None)
    if adv_inp is None:
        lprint('WARNING adv input not found by verifier, use alpha instead')
        adv_inp = orig_inp * record['adv']['alpha']
    else:
        # cwhn format
        assert adv_inp.shape[0] in [1, 3] and adv_inp.shape[-1] == 1
        adv_inp = torch.from_numpy(
            np.transpose(adv_inp, (3, 0, 2, 1)).astype(np.float32)).to(device)

    assert orig_inp.shape == adv_inp.shape, (orig_inp.shape, adv_inp.shape)
    err = (orig_inp * record['adv']['alpha'] - adv_inp).abs_().max().item()
    if err > record['eps']:
        lprint(f'WARNING Linf too large: Linf-eps={err-record["eps"]:.2e}')

    label = int(record['label'])

    rob_input = orig_inp * record['robust']['alpha']

    out = torch_as_npy(model.features_chk(rob_input)).flatten()
    assert np.argmax(out) == label
    out = torch_as_npy(model.features_chk(adv_inp)).flatten()
    if (cw := cw_loss_vec(out, label)) >= 0:
        lprint(f'WARNING adv is not truely adv: cw={cw}')

    data_range = (torch.clamp(rob_input - record['eps'], 0, 1),
                  torch.clamp(rob_input + record['eps'], 0, 1))

    def clip_inplace(x):
        x = torch.min(x, data_range[1], out=x)
        x = torch.max(x, data_range[0], out=x)
        return x

    adv_inp = clip_inplace(adv_inp)
    out_ref = torch_as_npy(model.features_chk(adv_inp)).flatten()
    out_unstbl = torch_as_npy(model.features(adv_inp)).flatten()
    assert np.argmax(out_ref) == label
    if np.argmax(out_unstbl) != label:
        err = float('inf')
    else:
        err = out_ref[label] - out_unstbl[label]

    cw = cw_loss_vec(out_ref, label)
    lprint(f'robust_alpha={record["robust"]["alpha"]:.3f} '
           f'adv_alpha={record["adv"]["alpha"]:.3f} '
           f'tgt_score_err={err:.3e} cw={cw:.3f} '
           f'time={record["robust"]["SolveTime"]:.3f}')
    return record, rob_input, cw

class ModelFinder:
    """modify classification confidence of a model so it is close to the
    decision boundary of robustness"""

    _mip_verify = None

    def __init__(self, model, args):
        self.model = model
        self.args = args

    def __del__(self):
        self.stop()

    def __call__(self, inp_edge_record, inp, adv_conf):
        label = int(inp_edge_record['label'])
        model = self.model
        mip_verify = self._get_mip_verify(inp_edge_record)

        def run2(tol0, tol1):
            dst = [None, None]
            def worker(tol, i):
                dst[i] = mip_verify[i](inp, label, tolerance=tol)
            t0 = threading.Thread(target=worker, args=(tol0, 0))
            t1 = threading.Thread(target=worker, args=(tol1, 1))
            t0.start()
            t1.start()
            t0.join()
            t1.join()
            return dst

        v0, v1 = run2(0, -adv_conf)
        lprint(f'default: {pformat_verify(v0)}')
        lprint(f'with delta {adv_conf}: {pformat_verify(v1)}')
        assert v0['status_known'] and v1['status_known']
        assert v0['robust'] and not v1['robust']
        assert v1['SolveStatus'] == 'Optimal'

        gap_robust = min(v1['ObjectiveValue'] + self.args.margin, 0)
        gap_adv = v1['ObjectiveValue'] - self.args.margin

        v2, v3 = run2(gap_robust, gap_adv)
        lprint(f'with {gap_robust=}: {pformat_verify(v2)}')
        lprint(f'with {gap_adv=}: {pformat_verify(v3)}')
        assert gap_robust > gap_adv
        assert v2['status_known'] and v3['status_known']
        assert v2['robust'] and not v3['robust']

        if self.args.margin_search > 0:
            def update(v, g):
                nonlocal v2, v3, gap_robust, gap_adv
                lprint(f'search with {g}: {pformat_verify(v)}')
                if not v['status_known']:
                    return 0
                if v['robust'] and g < gap_robust:
                    gap_robust = g
                    v2 = v
                elif not v['robust'] and g > gap_adv:
                    gap_adv = g
                    v3 = v
                return 1

            while gap_robust - gap_adv > self.args.margin_search:
                mid0 = gap_robust * (1/3) + gap_adv * (2/3)
                mid1 = gap_robust * (2/3) + gap_adv * (1/3)
                vmid0, vmid1 = run2(mid0, mid1)
                if update(vmid0, mid0) + update(vmid1, mid1) < 2:
                    lprint('give up margin search due to timeout')
                    break
            lprint(f'gaps: {gap_robust} {gap_adv} {gap_robust-gap_adv:.2e}')

        save_state = {
            'input_record': inp_edge_record,
            'input': inp,
            'gap': gap_robust,
            'adv': v3,
        }
        torch.save(save_state, inp_edge_record['path'] + '.model')

    def _get_mip_verify(self, inp_edge_record):
        if self._mip_verify is None:
            m = self.model
            self._mip_verify = [
                MIPVerify(
                    inp_edge_record['mat_model_path'],
                    (m.input_size, m.input_chl),
                    self.args.time_limit,
                    inp_edge_record['eps'],
                    max_nr_threads=self.args.max_solver_threads,
                )
                for _ in range(2)
            ]
        return self._mip_verify

    def stop(self):
        if self._mip_verify is not None:
            for i in self._mip_verify:
                i.stop()
            del self._mip_verify

def main_impl(args):
    device = 'cuda'
    model = None

    inputs = []
    for fpath in args.edge_input:
        if '.' in os.path.basename(fpath):
            lprint(f'skip file {fpath}')
            continue

        if not os.path.exists(fpath + '.finished'):
            lprint(f'finish mark for {fpath} does not exist, skip')
            continue

        record = torch.load(fpath)
        record['path'] = fpath

        lprint(f'-> processing {fpath} ...')
        if model is None:
            model = (ModelHelper.
                     create_with_load(record['model_path']).
                     to(device).
                     use_unstable_conv(True))
        try:
            result = preprocess(model, device, record)
            inputs.append(result)
        except AssertionError:
            lprint('failed:', traceback.format_exc())

    # sort by confidence
    inputs.sort(key=lambda x: x[-1])
    mf = ModelFinder(model, args)
    try:
        for i in inputs:
            finish_mark = i[0]['path'] + '.model.finished'
            lprint(f'work on {i[0]["path"]}: cw={i[2]}')
            if os.path.exists(finish_mark):
                lprint('already finished, skip')
                continue

            try:
                mf(*i)
                with open(finish_mark, 'w') as fout:
                    pass
            except AssertionError:
                lprint('failed:', traceback.format_exc())
    finally:
        mf.stop()

def run_main_with_lprint(main, args, lprint_setter):
    if args.log_file:
        with open(args.log_file, 'w') as fout:
            def lprint(*msg):
                print(*msg)
                print(*msg, file=fout)
                fout.flush()
            lprint_setter(lprint)
            return main(args)
    else:
        lprint_setter(print)
        return main(args)

def main():
    global lprint
    parser = argparse.ArgumentParser(
        description='try to find a model bias gap to make it close to the '
        'decision boundary of robustness, while also finding an adv input '
        'close to the boundary of input constraints'
    )
    parser.add_argument('edge_input', nargs='+',
                        help='output files from find_edge_input')
    parser.add_argument('--time-limit', type=int, default=360,
                        help='time limit for MIPVerify')
    parser.add_argument('--log-file', help='file to write log messages')
    parser.add_argument('--max-solver-threads', type=int, default=8,
                        help='max number of MIPVerify threads')
    parser.add_argument('--margin', type=float, default=3e-6,
                        help='default margin for solver objective to '
                        'derive gap and near adv')
    parser.add_argument('--margin-search', type=float, default=1e-7,
                        help='search for robustness/adv gap thresholding; set '
                        'to 0 to disable searching and use margin directly')
    args = parser.parse_args()
    np.set_printoptions(precision=2)

    def set_lprint(p):
        global lprint
        lprint = p
    run_main_with_lprint(main_impl, args, set_lprint)
