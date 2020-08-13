"""
Created on Mon Aug 10 17:06:13 2020

@author: cxue2
"""

from .. import env

# numba backend
nb = env.nb
np = env.np


@nb.njit(parallel=True, cache=True, fastmath=True)
def _jit_preemp_frmsig(in_, out, cnt, n_frm, frm_len, frm_stp, preemph, win):

    # offset for i
    off = cnt * n_frm
    
    for i in nb.prange(n_frm):        
        for j in range(frm_len):
            
            # index for input signal
            idx = frm_stp * i + j
            
            if 0 < idx < len(in_):
                out[off+i,j] = (in_[idx] - preemph * in_[idx-1]) * win[j]
            elif idx != 0:
                out[off+i,j] = 0.
            else:
                out[off,0] = in_[0] * win[0]
                
                
@nb.njit(parallel=True, cache=True, fastmath=True)
def _jit_powdiv(in_, out, nfft, fastmath=True):
    
    for i in nb.prange(in_.shape[0]):
        for j in range(in_.shape[1]):
            re, im = in_[i,j].real, in_[i,j].imag
            out[i,j] = (re * re + im * im) / nfft
            
            
@nb.njit(parallel=True, cache=True, fastmath=True)
def _jit_sum(in_, dim_i, dim_j):
    
    out = np.empty(dim_i, dtype=in_.dtype)
    
    for i in nb.prange(dim_i):
        tmp = 0.
        for j in range(dim_j):
            tmp += in_[i,j]
        out[i] = tmp
        
    return out


@nb.njit(parallel=True, cache=True, fastmath=True)
def _jit_mul(inm, inv, inplace, dim_i, dim_j):
    
    if inplace: 
        out = inm
    else:
        out = np.empty((dim_i, dim_j), dtype=inm.dtype)
        
    for i in nb.prange(dim_i):
        for j in range(dim_j):
            out[i,j] = inm[i,j] * inv[j]
            
    return out


@nb.njit(parallel=True, cache=True, fastmath=True)
def _jit_rplzro(in_, eps, dim, inplace):
    
    if inplace: 
        out = in_
    else:
        out = np.empty((dim,), dtype=in_.dtype)
    
    for i in nb.prange(dim):
        if in_[i] == 0:
            out[i] = eps
            
    return out


@nb.njit(parallel=True, cache=True, fastmath=True)
def _jit_rplzro_log(in_, log_eps, dim, inplace):
    
    if inplace: 
        out = in_
    else:
        out = np.empty((dim,), dtype=in_.dtype)
    
    for i in nb.prange(dim):
        if in_[i] == 0:
            out[i] = log_eps
        else:
            out[i] = np.log(in_[i])
            
    return out