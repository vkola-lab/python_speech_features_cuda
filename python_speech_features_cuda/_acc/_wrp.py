"""
Created on Mon Aug 10 18:26:54 2020

@author: cxue2
"""

from .. import _env_consistency_check
from .. import env

from .. import preemphasis
from .. import framesig
from .. import powspec

from . import fft
from . import _jit_preemp_frmsig
from . import _jit_powdiv

from math import ceil


def preemp_frmsig_powspc(in_, frm_len, frm_stp, preemph, win, nfft):
    
    # if either numba or pyfftw is missing or disabled, call default functions
    if not (env.use_numba and env.use_pyfftw):
        tmp = preemphasis(in_, preemph)
        tmp = framesig(tmp, frm_len, frm_stp, win)
        tmp = powspec(tmp, nfft)
        return tmp
    
    assert nfft is None or nfft >= frm_len, 'nfft must be greater than or equal to frame length.'
    _env_consistency_check(in_)
    
    # validate window function if given
    if win is not None:
        _env_consistency_check(win)
        assert len(win.shape) == 1, 'winfunc must be an 1-D array.'
        assert len(win) == frm_len, 'winfunc length shall be consistent with frame length.'
    
    # defualt window function (all 1s)
    else:
        win = env.np.ones(frm_len, dtype=env.dtype)
        
    # reshape input array
    shp_in_ = in_.shape
    
    if len(shp_in_) == 1:
        in_ = in_[env.np.newaxis,:]
        
    elif len(shp_in_) > 2:
        in_ = in_.reshape(-1, shp_in_[-1])
    
    # number of sequences and sequence length
    n_seq, seq_len = in_.shape
    
    # calculate number of frames
    if seq_len > frm_len:
        n_frm = 1 + ceil((seq_len - frm_len) / frm_stp)
        
    else:
        n_frm = 1
    
    # initialize fft obj
    nfft = nfft or frm_len
    obj = fft._fftw_obj((n_seq * n_frm, frm_len), nfft)
    
    # apply pre-emphasis, framing and window function all together
    for cnt in range(n_seq):
        _jit_preemp_frmsig(in_[cnt], obj.input_array, cnt, n_frm, frm_len, frm_stp, preemph, win)
    
    # run fft
    obj()
    
    # calculate power spectrum
    out = env.np.empty(obj.output_array.shape, dtype=env.dtype)
    _jit_powdiv(obj.output_array, out, nfft)
    
    # reshape back
    out = out.reshape(shp_in_[:-1] + (n_frm, out.shape[-1]))
    
    return out