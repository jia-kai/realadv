"""an interface to MIPVerify.jl"""

from .utils import torch_as_npy

import numpy as np
import torch
import h5py
import scipy.io as sio

import signal
import subprocess
import tempfile
import time
import lzma
import multiprocessing
from pathlib import Path

class MIPVerify:
    _subp = None
    _model_tmp_file = None
    default_eps = None

    def __init__(self, model_path, input_size_chl, time_limit, default_eps=None,
                 nr_threads=None, max_nr_threads=8):
        impl_path = str(Path(__file__).resolve().parent / 'mip_verify_impl.jl')
        inp_size, inp_chl = map(str, map(int, input_size_chl))
        time_limit = str(int(time_limit))
        if nr_threads is None:
            nr_threads = min(max_nr_threads, multiprocessing.cpu_count())
        nr_threads = str(int(nr_threads))

        if model_path.endswith('.xz'):
            with lzma.open(model_path, 'rb') as fin:
                self._model_tmp_file = tempfile.NamedTemporaryFile('wb')
                self._model_tmp_file.write(fin.read())
                self._model_tmp_file.flush()
            model_path = self._model_tmp_file.name

        subp = subprocess.Popen(
            ['julia', '--color=yes', impl_path,
             model_path, inp_size, inp_chl, time_limit, nr_threads],
            stdin=subprocess.PIPE)
        self._subp = subp
        self.default_eps = default_eps

    def _close_model_tmp(self):
        if self._model_tmp_file is not None:
            self._model_tmp_file.close()
            del self._model_tmp_file

    def __del__(self):
        self.stop()

    def stop(self):
        self._close_model_tmp()
        if self._subp is not None:
            self._subp.send_signal(signal.SIGKILL)
            del self._subp

    def __call__(self, *args, **kwargs):
        """shorthand for :meth:`untargeted_attack`"""
        return self.untargeted_attack(*args, **kwargs)

    def untargeted_attack(self, inp: np.ndarray, label: int, eps: float = None,
                          tolerance: float = 0):
        if isinstance(inp, torch.Tensor):
            inp = torch_as_npy(inp)
        if eps is None:
            eps = self.default_eps

        if inp.ndim == 4:
            inp = np.squeeze(inp, 0)

        assert inp.ndim == 3
        inp = np.transpose(inp, (1, 2, 0))[np.newaxis]
        label = int(label)
        eps = float(eps)
        tolerance = float(tolerance)

        def work(fname, done_flag):
            sio.savemat(fname, {'img': inp})
            self._subp.stdin.write(f'{fname}\n{label}\n{eps}\n{tolerance}\n'.
                                   encode('utf-8'))
            self._subp.stdin.flush()
            while not Path(done_flag).exists():
                time.sleep(1)
                if (p := self._subp.poll()) is not None:
                    raise RuntimeError(
                        f'verifier process died with exit code {p}')

            with h5py.File(fname, 'r') as f:
                ret = {k: np.array(v) for k, v in f.items()}
            for k, v in list(ret.items()):
                if v.size == 1:
                    ret[k] = v[0]
            ret['SolveStatus'] = (''.join(
                chr(int(i)) for i in ret['SolveStatus']))
            ret['robust'] = ret['SolveStatus'] == 'InfeasibleOrUnbounded'
            ret['status_known'] = ret['SolveStatus'] != 'UserLimit'
            self._close_model_tmp()
            return ret

        with tempfile.NamedTemporaryFile() as ftmp:
            done_flag = ftmp.name + '.done'
            def cleanup():
                try:
                    Path(done_flag).unlink()
                except FileNotFoundError:
                    pass

            cleanup()
            try:
                 return work(ftmp.name, done_flag)
            finally:
                cleanup()
