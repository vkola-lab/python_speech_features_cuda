"""
Created on Mon Aug 10 17:06:13 2020

@author: cxue2
"""

from .._env import env

# numba backend
nb = env.nb


@nb.njit(parallel=True, cache=True)
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
                out[off+i,j] = 0
            else:
                out[off,0] = in_[0] * win[0]
                
                
@nb.njit(parallel=True, cache=True)
def _jit_powdiv(in_, out, nfft):
    
    for i in nb.prange(in_.shape[0]):
        for j in range(in_.shape[1]):
            re, im = in_[i,j].real, in_[i,j].imag
            out[i,j] = (re * re + im * im) / nfft