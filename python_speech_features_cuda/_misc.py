"""
Created on Sun Aug  2 23:53:58 2020

@author: cxue2
"""

from ._env import env


def _env_consistency_check(arr):

    # backend check
    if env.backend is env.cp and type(arr) is env.cp.core.core.ndarray:
        pass
    
    elif env.backend is env.np and type(arr) is env.np.ndarray:
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