from __future__ import absolute_import
import types
import numpy as np


def wrap_namespace(old, new):
    unchanged_types =  set([types.FloatType, types.IntType, types.NoneType, types.TypeType])
    function_types = set([np.ufunc, types.FunctionType, types.BuiltinFunctionType])
    for name, obj in old.iteritems():
        if type(obj) in function_types:
            new[name] = primitive(obj)
        elif type(obj) in unchanged_types:
            new[name] = obj

wrap_namespace(np.__dict__, globals())



def cpu_or_gpu(cpufun, gpufun):
    def newfun(*args, **kwargs):
        self.fun = fun
    newfun.__name__ = cpufun.__name__
    newfun.__doc__ = cpufun.__doc__