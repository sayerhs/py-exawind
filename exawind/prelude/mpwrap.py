# -*- coding: utf-8 -*-

"""\
Multiprocessing module wrappers
-------------------------------

Allows class methods to be used in multiprocessing calls by registering pickle
option for class methods. The calling site just needs to import this module to
trigger the pickle setup. No additional action is necessary.

"""

import multiprocessing
import types
from six.moves import copyreg as copyreg

def _reduce_method(meth):
    return (getattr, (meth.__self__, meth.__func__.__name__))

copyreg.pickle(types.MethodType, _reduce_method)
