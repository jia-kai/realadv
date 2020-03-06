from .utils import ModelHelper, torch_as_npy, get_nr_correct, setup_pyx_import
from .find_edge_model import cw_loss_vec, run_main_with_lprint
from .cvt_model import read_mat_file
from .mip_verify import MIPVerify

import torch
import numpy as np
import scipy.io as sio

import argparse
import pickle
import os
import json
import tempfile
import sys
import pprint
import traceback
import functools

with setup_pyx_import():
    from ._attack_fparith import make_bounds_linf

lprint = None

def eval_acc(model, device):
    nr_test_tot = 0
    nr_test_correct = 0
    for inputs, labels in model.make_testset_loader():
        inputs = inputs.to(device)
        labels = labels.to(device)
        nr_test_correct += get_nr_correct(model(inputs), labels)
        nr_test_tot += inputs.size(0)
    return nr_test_correct / nr_test_tot

def check_adv(args, device,
              gap, mat_model_path, model_class, inp, adv_inp, label, eps, *,
              allow_retry=2):
    err = np.abs(inp - adv_inp).max()
    assert 0 <= adv_inp.min() <= adv_inp.max() <= 1
    assert err <= eps, (err, eps, err - eps)

    with tempfile.NamedTemporaryFile() as modified_model:
        if args.no_gap:
            modified_model_name = mat_model_path
        else:
            modified_model_name = modified_model.name
            params = read_mat_file(mat_model_path)
            params['softmax/bias'][0, label] += gap
            sio.savemat(modified_model_name, params)

        mip_verify = MIPVerify(
            modified_model_name,
            (model_class.input_size, model_class.input_chl),
            args.time_limit,
        )
        try:
            model = (model_class.
                     from_mat(modified_model_name).
                     to(device).
                     use_unstable_conv(not args.stable))
            adv_inp_dev = torch.from_numpy(adv_inp).to(device)
            out0 = torch_as_npy(model(adv_inp_dev)).flatten()
            out1 = torch_as_npy(model.features_chk(adv_inp_dev)).flatten()
            test_acc = eval_acc(model, device)

            lprint(f'conv {out0} l={np.argmax(out0)} '
                  f'cw={cw_loss_vec(out0, label):.2e}')
            lprint(f'mm   {out1} l={np.argmax(out1)} '
                  f'cw={cw_loss_vec(out1, label):.2e}')
            lprint(f'test acc: {test_acc*100:.2f}%')

            if args.mm:
                out_adv = out1
            else:
                out_adv = out0

            assert np.argmax(out_adv) != label

            v = mip_verify(inp, label, eps)
            if not (v['status_known'] and v['robust']):
                v.pop('PerturbationValue', None)
                v.pop('PerturbedInputValue', None)
                lprint('verification of original model failed:',
                      pprint.pformat(v))
                if allow_retry > 0 and v['status_known']:
                    try:
                        # try to increase robustness a little bit
                        safe = -cw_loss_vec(out_adv, label)
                        gap += min(-v['ObjectiveValue'],
                                   max(safe - 1e-7, safe * 0.99))
                        gap = float(np.nextafter(gap, float('inf'),
                                                 dtype=np.float32))
                        lprint(f'retrying with new gap {gap} ...')
                        return check_adv(
                            args, device,
                            gap, mat_model_path, model_class, inp, adv_inp,
                            label, eps, allow_retry=allow_retry-1)
                    except:
                        traceback.print_exc()
                return
        finally:
            mip_verify.stop()

    inp_dev = torch.from_numpy(inp).to(device)
    save_state = {
        'inp': inp,
        'adv_inp': adv_inp,
        'label': label,
        'gap': gap,
        'eps': eps,
        'verify': v,
        'test_acc': test_acc,
        'adv_out_score': out0,
        'adv_out_score_mm': out1,
        'inp_out_score': torch_as_npy(model(inp_dev)).flatten(),
        'mat_model_path': mat_model_path,
        'device': device,
        'argv_options': [i for i in sys.argv if i.startswith('-')],
    }
    return save_state

def make_json_state(state):
    ret = {}
    for k, v in state.items():
        if isinstance(v, np.ndarray):
            if v.size > 16 or v.ndim > 1:
                continue
            v = list(map(float, v))
        ret[k] = v
    vs = ret['verify'] = state['verify'].copy()
    for k, v in list(ret['verify'].items()):
        if isinstance(v, np.float64):
            vs[k] = float(v)
        elif isinstance(v, np.int64):
            vs[k] = int(v)
        elif not isinstance(v, (str, bool)):
            del vs[k]
    return ret

def make_data_range(inp: torch.Tensor, eps):
    inp_npy = torch_as_npy(inp)
    assert inp_npy.dtype == np.float32

    imin, imax = make_bounds_linf(inp_npy, eps)
    def chk_eps(v):
        vm = v.max()
        assert vm <= eps, (vm, eps, eps - vm)
    chk_eps((inp_npy - imin).astype(np.float64))
    chk_eps((imax - inp_npy).astype(np.float64))
    chk_eps(inp_npy.astype(np.float64) - imin.astype(np.float64))
    chk_eps(imax.astype(np.float64) - inp_npy.astype(np.float64))
    return (torch.from_numpy(imin).to(inp.device),
            torch.from_numpy(imax).to(inp.device))

def process(args, out_path, edge_model_record):
    device = 'cpu' if args.cpu else 'cuda'
    model = (ModelHelper.
             create_with_load(edge_model_record['input_record']['model_path']).
             to(device).
             use_unstable_conv(not args.stable))
    label = int(edge_model_record['input_record']['label'])
    eps = edge_model_record['input_record']['eps']

    if args.no_gap:
        edge_model_record['gap'] = 0
    else:
        for i in (model.features[-1], model.features_chk[-1]):
            i.bias[label] += edge_model_record['gap']
            i.bias.detach_()

    inp: torch.Tensor = edge_model_record['input'].to(device)
    assert inp.ndim == 4
    adv_inp = edge_model_record['adv']['PerturbedInputValue']
    # cwhn format
    assert adv_inp.shape[0] in [1, 3] and adv_inp.shape[-1] == 1
    adv_inp = torch.from_numpy(
        np.transpose(adv_inp, (3, 0, 2, 1)).astype(np.float32)).to(device)

    assert inp.shape == adv_inp.shape

    data_range = make_data_range(inp, eps)
    def clip_adv(x):
        x = torch.min(x, data_range[1])
        x = torch.max(x, data_range[0], out=x)
        return x

    def cw_loss(x, mm=args.mm):
        if mm:
            out = model.features_chk(x)
        else:
            out = model(x)
        return cw_loss_vec(torch_as_npy(out.flatten()), label)

    label_dev = torch.tensor([label], dtype=torch.long, device=device)
    orig_adv_inp = adv_inp
    adv_inp = clip_adv(adv_inp)
    init_loss = cur_loss = cw_loss(adv_inp)
    init_loss_mm = cw_loss(adv_inp, True)
    lprint(f'adv_loss={cw_loss(adv_inp, False)} adv_loss_mm={init_loss_mm} '
           f'inp_loss={cw_loss(inp)} gap={edge_model_record["gap"]}')

    yield init_loss

    if args.stable:
        blk_stride = 4
        blk_off = 0
    else:
        # winograd: conv(13, 5) -> 9
        blk_stride = 9
        blk_off = 4

    assert model.features[0].weight.shape[2:] == (5, 5)
    for iter_num in range(args.max_iter):
        for i in range(0, inp.shape[2]-blk_off, blk_stride):
            i0 = i + blk_off
            i1 = i + blk_stride
            for j in range(0, inp.shape[3]-blk_off, blk_stride):
                j0 = j + blk_off
                j1 = j + blk_stride
                sub = adv_inp[:, :, i0:i1, j0:j1]
                diff = torch.empty_like(sub).uniform_(-args.pert, args.pert)
                x = adv_inp.clone()
                x[:, :, i0:i1, j0:j1] += diff
                x = clip_adv(x)
                new_loss = cw_loss(x)
                if new_loss < cur_loss:
                    linf = (x - orig_adv_inp).abs_().max().item()
                    lprint(
                        f'#{iter_num}: loss: {cur_loss:.5e} -> {new_loss:.5e} '
                        f'diff={init_loss - new_loss:.2e},'
                        f'{init_loss_mm - new_loss:.2e} Linf={linf:.3e}')
                    cur_loss = new_loss
                    adv_inp = x

    if cur_loss < 0:
        state = check_adv(
            args,
            device,
            float(edge_model_record['gap']),
            edge_model_record['input_record']['mat_model_path'],
            type(model),
            torch_as_npy(inp),
            torch_as_npy(adv_inp),
            int(label),
            float(eps),
        )
        if state is not None:
            with open(out_path, 'wb') as fout:
                pickle.dump(state, fout, protocol=pickle.HIGHEST_PROTOCOL)
            j_state = make_json_state(state)
            with open(out_path + '.json', 'w') as fout:
                json.dump(j_state, fout, indent=2)

    with open(out_path + '.finished', 'w') as fout:
        pass

def main():
    parser = argparse.ArgumentParser(
        description='try attack by random input change'
    )
    parser.add_argument('edge_model', nargs='+',
                        help='model produced by find_edge_model')
    parser.add_argument('--max-iter', type=int, default=1000,
                        help='max number of iterations to find adversarial '
                        'example')
    parser.add_argument('--time-limit', type=int, default=360,
                        help='time limit for MIPVerify')
    parser.add_argument('--pert', type=float, default=2e-7,
                        help='uniform sampling range for searching '
                        'adversarial example')
    parser.add_argument('--stable', action='store_true',
                        help='use stable computation instead of the unstable')
    parser.add_argument('--cpu', action='store_true',
                        help='use CPU runtime library; implies --stable')
    parser.add_argument('--mm', action='store_true',
                        help='use matmul inference; implies --stable')
    parser.add_argument('--no-gap', action='store_true',
                        help='do not apply gap to model bias and use original '
                        'model')
    parser.add_argument('--ignore-finished', action='store_true',
                        help='ignore finished mark and rerun all cases')
    parser.add_argument('--log-file', help='log output file')
    args = parser.parse_args()
    np.set_printoptions(precision=2)

    if args.cpu or args.mm:
        args.stable = True

    def set_lprint(p):
        global lprint
        lprint = p

    output_suffix = (f'nogap{int(args.no_gap)}_stable{int(args.stable)}_'
                     f'mm{int(args.mm)}_cpu{int(args.cpu)}')
    if args.log_file:
        args.log_file += f'_{output_suffix}'

    run_main_with_lprint(functools.partial(main_impl, output_suffix),
                         args, set_lprint)

def main_impl(output_suffix, args):
    all_models = []

    for i in args.edge_model:
        adv_name = i + f'.adv_{output_suffix}'
        if os.path.exists(adv_name + '.finished') and not args.ignore_finished:
            lprint(f'skip existing file {i}')
        else:
            lprint(f'preprocessing on {i}')
            gen = process(args, adv_name, torch.load(i))
            all_models.append((gen, i, next(gen)))

    all_models.sort(key=lambda x: x[-1])
    for gen, name, loss in all_models:
        lprint(f'working on {name} with {loss=}')
        try:
            next(gen)
            assert 0, 'should stop iter'
        except StopIteration:
            pass
