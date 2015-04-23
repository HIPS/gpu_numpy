"""Test framework for checking that numpy and gpu_numpy give equivalent results."""
import itertools as it
import numpy.random as npr
import numpy as onp
import gpu_numpy as gnp

def combo_check(fun_name, array_argnums, *args, **kwargs):
    # Tests all combinations of args given.
    args = list(args)
    kwarg_key_vals = [[(key, val) for val in kwargs[key]] for key in kwargs]
    num_args = len(args)
    for args_and_kwargs in it.product(*(args + kwarg_key_vals)):
        cur_args = args_and_kwargs[:num_args]
        cur_kwargs = dict(args_and_kwargs[num_args:])
        check_np_equivalence(fun_name, cur_args, cur_kwargs, array_argnums)
        print ".",

def check_np_equivalence(fun_name, args, kwargs, array_argnums):
    ans_truth = getfun(onp, fun_name)(*args, **kwargs) # Original numpy, regular args
    gnp_fun = getfun(gnp, fun_name)
    for dtypes in it.product(*[[gnp.float64, gnp.gpu_float32]]*len(array_argnums)):
        new_args = list(args)
        for dtype, argnum in zip(dtypes, array_argnums):
            if isinstance(args[argnum], onp.ndarray):
                new_args[argnum] = gnp.array(args[argnum], dtype=dtype)

        try:
            ans_test = gnp_fun(*new_args, **kwargs)
            if isinstance(ans_test, gnp.garray):
                ans_test = gnp.array(ans_test, dtype=gnp.float64)

            assert type(ans_truth) is type(ans_test), \
                "Type mismatch! \nTruth:  {0}, \nResult: {1}".format(type(ans_truth), type(ans_test))
            assert onp.allclose(ans_truth, ans_test, atol=1e-5), \
                "Value mismatch.\nTruth:  {0}, \nResult: {1}".format(ans_truth, ans_test)
        except:
            print "Test failed, dtypes were {0} args were {1}, kwargs were {2}".format(
                dtypes, args, kwargs)
            raise

def getfun(module, fun_name):
    if isinstance(fun_name, str):
        return getattr(module, fun_name)
    else:
        # Assume it's actually a function
        return fun_name

def stat_check(fun_name):
    # Tests fun_namections that compute statistics, like sum, mean, etc
    x = 3.5
    A = npr.randn()
    B = npr.randn(3)
    C = npr.randn(2, 3)
    D = npr.randn(1, 3)
    combo_check(fun_name, (0,), [x, A])
    combo_check(fun_name, (0,), [B, C, D], axis=[None, 0], keepdims=[True, False])
    combo_check(fun_name, (0,), [C, D], axis=[None, 0, 1], keepdims=[True, False])

def unary_ufunc_check(fun_name, lims=[-2, 2]):
    scalar_int = transform(lims, 1)
    scalar = transform(lims, 0.4)
    vector = transform(lims, npr.rand(2))
    mat    = transform(lims, npr.rand(3, 2))
    mat2   = transform(lims, npr.rand(1, 2))
    combo_check(fun_name, (0,), [scalar_int, scalar, vector, mat, mat2])

def binary_ufunc_check(fun_name, lims_A=[-2, 2], lims_B=[-2, 2]):
    T_A = lambda x : transform(lims_A, x)
    T_B = lambda x : transform(lims_B, x)
    scalar_int = 1
    scalar = 0.6
    vector = npr.rand(2)
    mat    = npr.rand(3, 2)
    mat2   = npr.rand(1, 2)
    combo_check(fun_name, (0, 1), [T_A(scalar), T_A(scalar_int), T_A(vector), T_A(mat), T_A(mat2)],
                             [T_B(scalar), T_B(scalar_int), T_B(vector), T_B(mat), T_B(mat2)])

def transform(lims, x):
    return x * (lims[1] - lims[0]) + lims[0]
