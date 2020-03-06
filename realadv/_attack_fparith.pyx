# -*- coding: utf-8 -*-
# cython: language_level=3

import numpy as np
cimport numpy as np

cdef extern from "_attack_fparith.h":
    float lower_bound(float center_f32, double eps_f64)
    float upper_bound(float center_f32, double eps_f64)

def make_bounds_linf(center, double eps):
    cf = center.flatten()
    lower = np.empty_like(cf)
    upper = np.empty_like(cf)
    cdef np.ndarray[np.float32_t, ndim=1] cf_np = cf
    cdef np.ndarray[np.float32_t, ndim=1] lower_np = lower
    cdef np.ndarray[np.float32_t, ndim=1] upper_np = upper
    cdef unsigned i
    cdef unsigned size = cf.shape[0]

    for i in range(size):
        lower_np[i] = lower_bound(cf_np[i], eps)
        upper_np[i] = upper_bound(cf_np[i], eps)
    return lower.reshape(center.shape), upper.reshape(center.shape)
