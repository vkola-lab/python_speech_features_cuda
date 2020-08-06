"""
Created on Sun Aug  2 23:53:58 2020

@author: cxue2
"""

from ._env import env

# import numpy
import numpy as np

# import cupy if available
if env.is_cupy_available:
    import cupy as cp


def _env_consistency_check(arr):

    # backend check
    if env.is_cupy_available and env.backend is cp and type(arr) is cp.core.core.ndarray:
        pass
    
    elif env.backend is np and type(arr) is np.ndarray:
        pass
        
    else:
        msg = 'The input array is {} while the backend is set to be <{}>.'.format(type(arr), env.backend.__name__)
        raise TypeError(msg)

        
    # dtype check
    if arr.dtype.type is env.dtype:
        pass
        
    else:
        msg = 'The dtype of the input array is <{}> while the environment dtype is set to be <{}>.'.format(arr.dtype.type.__name__, env.dtype.__name__)
        raise TypeError(msg)