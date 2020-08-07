"""
Created on Sun Aug  2 21:37:17 2020

@author: cxue2
"""

from ._misc import _env_consistency_check
from ._env  import env

import numpy as np
             

def framesig(sig, frame_len, frame_step, winfunc=None):
    '''
    Frame signals into overlapping frames.

    Parameters
    ----------
    sig : array_like of shape ([B0, ..., Bn,] signal_len)
        Input signals.
    frame_len : int
        Length of each frame measured in samples.
    frame_step : int
        Number of samples after the start of the previous frame that the next frame should begin.
    winfunc : array_like, optional
        The analysis window to apply to each frame. The default is None.

    Returns
    -------
    rsl : array_like of shape ([B0, ..., Bn,] #_of_frames, frame_len)
        Frames.
    '''
    
    _env_consistency_check(sig)
    
    # check for winfunc    
    if winfunc is not None:
        _env_consistency_check(winfunc)
        assert len(winfunc.shape) == 1, 'winfunc must be an 1-D array.'
        assert winfunc.shape[0] == frame_len, 'winfunc length shall be consistent with frame length.'
    
    # calculate number of frames
    if sig.shape[-1] > frame_len:
        n_frm = 1 + int(np.ceil((sig.shape[-1] - frame_len) / frame_step))
        
    else:
        n_frm = 1
    
    # strided split
    shp = sig.shape[:-1] + (n_frm, frame_len)
    std = sig.strides[:-1] + (sig.strides[-1] * frame_step, sig.strides[-1])
    tmp = env.backend.lib.stride_tricks.as_strided(sig, shape=shp, strides=std)
    
    # copy is necessary since stride_tricks returns a view
    # If the user accidentally operates the array inplace, the result will be unexpected.
    tmp = env.backend.copy(tmp)
    
    # assign 0 to overflowed bytes
    idx = (n_frm - 1) * frame_step + frame_len - sig.shape[-1]
    tmp[...,-1,-idx:] = 0
    
    # apply winfunc
    if winfunc is not None:
        tmp = tmp * winfunc
        
    return tmp


def magspec(sig, nfft=None):
    '''
    Compute the magnitude spectrum of the input signals.

    Parameters
    ----------
    sig : array_like of shape ([B0, ..., Bn,] signal_len)
        Input signals.
    nfft : int
        The FFT length to use. Default is None. Please check
        https://numpy.org/doc/stable/reference/generated/numpy.fft.fft.html
        for details.

    Returns
    -------
    array_like of shape ([B0, ..., Bn,] nfft//2+1)
        The magnitude spectrum of the input signals.
    '''

    # compute power spectrum (un-normalized)
    nfft = nfft or sig.shape[-1]
    tmp = powspec(sig, nfft) * nfft
    
    # square root to get magnitude spectrum
    tmp = env.backend.sqrt(tmp)
    
    return tmp


def powspec(sig, nfft=None):
    '''
    Compute the power spectrum of the input signals. 

    Parameters
    ----------
    sig : array_like of shape ([B0, ..., Bn,] signal_len)
        Input signals.
    nfft : int
        The FFT length to use. Default is None. Please check
        https://numpy.org/doc/stable/reference/generated/numpy.fft.fft.html
        for details.

    Returns
    -------
    array_like of shape ([B0, ..., Bn,] nfft//2+1)
        The power spectrum of the input signals.
    '''
    
    _env_consistency_check(sig)
    
    # apply FFT
    nfft = nfft or sig.shape[-1]
    tmp = env.backend.fft.rfft(sig, nfft)
    
    # compute power spectrum (un-normalized)
    tmp = env.backend.real(tmp * tmp.conj())
    
    # cast dtype back since fft may change dtype
    tmp = tmp.astype(env.dtype, copy=False)
    
    # divided by length to get power
    tmp = tmp / nfft
    
    return tmp


def logpowspec(sig, nfft=None, norm=True):
    '''
    Compute the log power spectrum of the input signals.

    Parameters
    ----------
    sig : array_like of shape ([B0, ..., Bn,] signal_len)
        Input signals.
    nfft : int, optional
        The FFT length to use. Default is None. Please check
        https://numpy.org/doc/stable/reference/generated/numpy.fft.fft.html
        for details.
    norm : boolean, optional
        If norm, the log power spectrum is normalised so that the max value 
        (across all frames) is 0.

    Returns
    -------
    array_like of shape ([B0, ..., Bn,] nfft//2+1)
        The log power spectrum of the input signals.
    '''

    # compute power spectrum
    nfft = nfft or sig.shape[-1]
    tmp = powspec(sig, nfft)
    
    # eliminate zeros for numerical stability
    eps = env.backend.finfo(env.dtype).eps
    tmp = env.backend.where(tmp == 0, eps, tmp)
    
    # apply log
    tmp = 10 * env.backend.log10(tmp)
    
    # normalize
    if norm: tmp = tmp - env.backend.max(tmp)
    
    return tmp


def preemphasis(sig, coeff=0.97):
    '''
    Perform preemphasis on the input signals.

    Parameters
    ----------
    sig : array_like of shape ([B0, ..., Bn,] signal_len)
        Input signals.
    coeff : int, optional
        The preemphasis coefficient. 0 is no filter, default is 0.97.

    Returns
    -------
    array_like of shape ([B0, ..., Bn,] signal_len)
        The filtered signals.
    '''
    
    _env_consistency_check(sig)
    
    # result placeholder
    tmp = env.backend.copy(sig)
    
    # apply preemphasis filter
    tmp[...,1:] = sig[...,1:] - coeff * sig[...,:-1]
    
    return tmp