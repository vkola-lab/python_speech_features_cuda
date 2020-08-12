"""
Created on Tue Aug 11 17:02:08 2020

@author: cxue2
"""

from .._env import env

if env.is_numba_available:
    from ._jit import _jit_sum
    from ._jit import _jit_mul
    from ._jit import _jit_rplzro_log
    from ._jit import _jit_rplzro 
else:
    _jit_sum        = None
    _jit_mul        = None
    _jit_rplzro_log = None
    _jit_rplzro     = None


def sum(in_):
    
    # call default function
    if not env.use_numba: return env.backend.sum(in_, axis=-1)

    # reshape
    shp = in_.shape
    in_ = in_.reshape(-1, shp[-1])
     
    # run sum
    out = _jit_sum(in_, in_.shape[0], in_.shape[1])
    
    return out.reshape(shp[:-1])


def mul(inm, inv, inplace=True):
        
    # call numba jit
    if not env.use_numba: return inm * inv

    # reshape
    shp = inm.shape
    inm = inm.reshape(-1, shp[-1])
     
    # run mul
    out = _jit_mul(inm, inv, inplace, inm.shape[0], inm.shape[1])
    
    return out.reshape(shp)


def rplzro(in_, eps, inplace=True):
    
    # call default function
    if not env.use_numba: return env.backend.where(in_ == 0, eps, in_)
        
    # reshape
    shp = in_.shape
    in_ = in_.reshape(-1)
    
    # run
    out = _jit_rplzro(in_, eps, in_.shape[0], inplace)
    
    return out.reshape(shp)


def rplzro_log(in_, eps, inplace=True):
    
    # call default function
    if not env.use_numba:
        tmp = env.backend.where(in_ == 0, eps, in_)
        tmp = env.backend.log(tmp)
        return tmp
        
    # reshape
    shp = in_.shape
    in_ = in_.reshape(-1)
    
    # run
    out = _jit_rplzro_log(in_, env.np.log(eps), len(in_), inplace)
    
    return out.reshape(shp)
        
        

