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
    

_err_msg_0 = 'The backend type and the data type of input array must ' + \
             'exactly match the pakage environment settings.'


def _reshape(arr, copy=False):
    '''
    Reshape the input array.

    Parameters
    ----------
    arr : array_like of shape ([B0, ..., Bn,] L)
        Input array.
    copy : boolean, optional
        Copy input array if True.

    Returns
    -------
    array_like of shape ([B0 * ... * Bn,] L)
        A reshaped copy of the input array. If the input array is 1D, an extra
        dimension will be appended to the left.
    tuple
        The original shape of the input array.
    '''
    
    # input array
    arr_ = env.backend.array(arr, copy=True) if copy else arr
    
    # reshape
    arr_ = arr_.reshape(-1, arr_.shape[-1])
    
    return arr_


def _env_consistency_check(arr):

    # backend check
    if env.is_cupy_available and env.backend is cp and type(arr) is cp.core.core.ndarray:
        flg = True
    
    elif env.backend is np and type(arr) is np.ndarray:
        flg = True
        
    else:
        flg = False
    
    # return immediately if backend check fails
    if not flg: return flg
        
    # dtype check
    if arr.dtype.type is env.dtype:
        flg = True
        
    else:
        flg = False
    
    return flg