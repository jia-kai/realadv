import subprocess
import ctypes
import threading
from contextlib import contextmanager
from pathlib import Path

g_base_dir = Path(__file__).resolve().parent.parent
subprocess.check_call(['make', '-C', str(g_base_dir), '-s'])
g_lib = ctypes.CDLL(str(g_base_dir / 'cudnnwrap.so'), ctypes.RTLD_GLOBAL)
g_lock = threading.Lock()

@contextmanager
def enforce_conv_fwd_algo(name: str):
    """set the algorithm to use for fwd convs"""
    assert isinstance(name, str)
    set_algo = g_lib.set_algo
    set_algo.argtypes = [ctypes.c_char_p]
    reset_cnt = g_lib.reset_call_cnt
    reset_cnt.restype = ctypes.c_int

    with g_lock:
        set_algo(name.encode('utf-8'))
        reset_cnt()
        try:
            yield
            assert reset_cnt() >= 1
        finally:
            set_algo(b'')
