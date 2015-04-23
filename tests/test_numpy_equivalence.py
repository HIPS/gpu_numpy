import numpy.random as npr
from test_utils import combo_check, stat_check, unary_ufunc_check, binary_ufunc_check
npr.seed(0)

# Array statistics functions
def test_max():  stat_check('max')
def test_all():  stat_check('all')
def test_any():  stat_check('any')
def test_max():  stat_check('max')
def test_mean(): stat_check('mean')
def test_min():  stat_check('min')
def test_sum():  stat_check('sum')
def test_prod(): stat_check('prod')
def test_var():  stat_check('var')
def test_std():  stat_check('std')

# Unary ufunc tests

def test_sin():     unary_ufunc_check('sin') 
def test_abs():     unary_ufunc_check('abs', lims=[0.1, 4.0])
def test_absolute():unary_ufunc_check('absolute', lims=[0.1, 4.0])
def test_arccosh(): unary_ufunc_check('arccosh', lims=[1.1, 4.0])
def test_arcsinh(): unary_ufunc_check('arcsinh', lims=[-0.9, 0.9])
def test_arctanh(): unary_ufunc_check('arctanh', lims=[-0.9, 0.9])
def test_ceil():    unary_ufunc_check('ceil', lims=[-1.5, 1.5])
def test_cos():     unary_ufunc_check('cos')
def test_cosh():    unary_ufunc_check('cosh')
def test_deg2rad(): unary_ufunc_check('deg2rad')
def test_degrees(): unary_ufunc_check('degrees')
def test_exp():     unary_ufunc_check('exp')
def test_exp2():    unary_ufunc_check('exp2')
def test_expm1():   unary_ufunc_check('expm1')
def test_fabs():    unary_ufunc_check('fabs')
def test_floor():   unary_ufunc_check('floor', lims=[-1.5, 1.5])
def test_log():     unary_ufunc_check('log',   lims=[0.2, 2.0])
def test_log10():   unary_ufunc_check('log10', lims=[0.2, 2.0])
def test_log1p():   unary_ufunc_check('log1p', lims=[0.2, 2.0])
def test_log2():    unary_ufunc_check('log2',  lims=[0.2, 2.0])
def test_rad2deg(): unary_ufunc_check('rad2deg')
def test_radians(): unary_ufunc_check('radians')
def test_sign():    unary_ufunc_check('sign')
def test_sin():     unary_ufunc_check('sin')
def test_sinh():    unary_ufunc_check('sinh')
def test_sqrt():    unary_ufunc_check('sqrt', lims=[1.0, 3.0])
def test_square():  unary_ufunc_check('square')
def test_tan():     unary_ufunc_check('tan', lims=[-1.1, 1.1])
def test_tanh():    unary_ufunc_check('tanh')
def test_real():    unary_ufunc_check('real')
def test_real_ic(): unary_ufunc_check('real_if_close')
def test_imag():    unary_ufunc_check('imag')
def test_conj():    unary_ufunc_check('conj')
def test_angle():   unary_ufunc_check('angle')

# Binary ufunc tests

def test_add(): binary_ufunc_check('add')
def test_logaddexp(): binary_ufunc_check('logaddexp')
def test_logaddexp2(): binary_ufunc_check('logaddexp2')
def test_remainder(): binary_ufunc_check_no_same_args('remainder', lims_A=[-0.9, 0.9], lims_B=[0.7, 1.9])
def test_mod(): binary_ufunc_check_no_same_args('mod',    lims_B=[0.8, 2.1])
def test_mod_neg(): binary_ufunc_check_no_same_args('mod',    lims_B=[-0.3, -2.0])

def test_op_mul(): binary_ufunc_check(op.mul)
def test_op_add(): binary_ufunc_check(op.add)
def test_op_sub(): binary_ufunc_check(op.sub)
def test_op_mod(): binary_ufunc_check_no_same_args(op.mod, lims_B=[0.3, 2.0])
def test_op_mod_neg(): binary_ufunc_check_no_same_args(op.mod, lims_B=[-0.3, -2.0])
def test_op_div(): binary_ufunc_check(op.div, lims_B=[0.5, 2.0])
def test_op_pow(): binary_ufunc_check(op.pow, lims_A=[0.7, 2.0])



# Misc tests

def test_transpose(): combo_check('transpose', [0],
                                  [npr.randn(2, 3, 4)],
                                  axes = [None, [0, 1, 2], [0, 2, 1],
                                                [2, 0, 1], [2, 1, 0],
                                                [1, 0, 2], [1, 2, 0]])

R = npr.randn
def test_dot(): combo_check('dot', [0, 1],
                            [1.5, R(3), R(2, 3)],
                            [0.3, R(3), R(3, 4)])
def test_tensordot_1(): combo_check('tensordot', [0, 1],
                                    [R(1, 3), R(2, 3, 2)],
                                    [R(3),    R(3, 1),    R(3, 4, 2)],
                                    axes=[ [(1,), (0,)] ])
def test_tensordot_2(): combo_check('tensordot', [0, 1],
                                    [R(3),    R(3, 1),    R(3, 4, 2)],
                                    [R(1, 3), R(2, 3, 2)],
                                    axes=[ [(0,), (1,)] ])
def test_tensordot_3(): combo_check('tensordot', [0, 1],
                                    [R(2, 3),    R(2, 3, 4)],
                                    [R(1, 2, 3), R(2, 2, 3, 4)],
                                    axes=[ [(0, 1), (1, 2)] ,  [(1, 0), (2, 1)] ])
def test_tensordot_4(): combo_check('tensordot', [0, 1],
                                    [R(2, 2), R(4, 2, 2)],
                                    [R(2, 2), R(2, 2, 4)],
                                    axes=[1, 2])

# Need custom tests because gradient is undefined when arguments are identical.
def test_maximum(): combo_check('maximum', [0, 1],
                               [R(1), R(1,4), R(3, 4)],
                               [R(1), R(1,4), R(3, 4)])

def test_minimum(): combo_check('minimum', [0, 1],
                               [R(1), R(1,4), R(3, 4)],
                               [R(1), R(1,4), R(3, 4)])

def test_sort():       combo_check('sort', [0], [R(1), R(7)])
def test_msort():     combo_check('msort', [0], [R(1), R(7)])
def test_partition(): combo_check('partition', [0], [R(7), R(14)], kth=[0, 3, 6])
