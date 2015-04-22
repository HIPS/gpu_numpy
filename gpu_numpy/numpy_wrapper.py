from __future__ import absolute_import
import types
import numpy as np

def wrap_namespace(old, new):
    for name, obj in old.iteritems():
        new[name] = obj

wrap_namespace(np.__dict__, globals())
