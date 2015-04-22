import numpy_wrapper
orig_any = any # We'll shadow `any` with numpy's version
from numpy_wrapper import *
# Functions shadowing builtins don't get imported by *
from numpy_wrapper import round, max, min, abs, complex, any
import numpy_wrapper as np
import gnumpy as gnp
from gnumpy import garray, is_garray, gpu_float32

def merge_numpy_gnumpy(np_fun, gnp_fun):
    def np_or_gnp_fun(*args, **kwargs):
        if orig_any(map(gnp.is_garray, args)):
            return gnp_fun(*args, **kwargs)
        else:
            return np_fun(*args, **kwargs)
    return np_or_gnp_fun

gnumpy_wrapped_funs = ['dot', 'outer', 'concatenate', 'where', 'nonzero',
                       'eye', 'tensordot', 'all', 'any', 'sum', 'mean', 'max',
                       'min', 'abs', 'exp', 'isinf', 'isnan', 'log', 'log1p',
                       'negative', 'sign', 'sqrt', 'tanh']

gdict = globals()
for name in gnumpy_wrapped_funs:
    gdict[name] = merge_numpy_gnumpy(np.__dict__[name], gnp.__dict__[name])

def array(x, dtype=None, **kwargs):
    if gnp.is_garray(x):
        if dtype is gpu_float32:
            return x
        else:
            return np.array(gnp.as_numpy_array(x), dtype=dtype, **kwargs)
    else:
        if dtype is gpu_float32:
            return gnp.as_garray(np.array(x, **kwargs))
        else:
            return np.array(x, dtype=dtype, **kwargs)
