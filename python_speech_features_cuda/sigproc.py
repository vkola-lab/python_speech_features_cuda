"""
Created on Sun Aug  2 21:37:17 2020

@author: cxue2
"""

from ._misc import _reshape
from ._misc import _env_consistency_check
from ._misc import _err_msg_0
from ._env  import env
             

def framesig(sig, frame_len, frame_step, winfunc=None):
    '''
    Frame signals into overlapping frames.

    Parameters
    ----------
    sig : array_like of shape ([B0, ..., Bn,] L)
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
    
    assert _env_consistency_check(sig), _err_msg_0
    
    # check for winfunc
    assert winfunc is None or _env_consistency_check(winfunc), 'Invalidk winfunc type.'
    
    if winfunc is not None:
        assert len(winfunc.shape) == 1, 'winfunc must be an 1-D array.'
        assert winfunc.shape[0] == frame_len, 'winfunc length shall be consistent with frame length.'
        
    # reshape input array
    shp = sig.shape
    tmp = _reshape(sig)
    
    # number of frames after strided split
    n_row = (shp[-1] - frame_len) // frame_step + 1
    
    # # need for zero-padding?
    # cap = frame_step * n_row + frame_len - frame_step
    # pad = 0 if cap == shp[-1] else 1
    # n_row += pad
    # dif = cap + pad * frame_step - shp[-1]
    
    # result placeholder
    rsl_shp = (n_row * len(tmp), frame_len)
    rsl = env.backend.empty(rsl_shp, dtype=env.dtype)
    
    # offset in bytes
    off = tmp.strides[-1]
    
    # name alias for strided split function
    fnc = env.backend.lib.stride_tricks.as_strided
    
    # main loop
    for i, seq in enumerate(tmp):
        rsl[i*n_row:(i+1)*n_row,:] = fnc(seq, shape=(n_row, frame_len), 
                                         strides=(frame_step*off, off))
        
        # # zero-padding
        # if pad: rsl[(i+1)*n_row-1,-dif:] = 0
    
    # apply winfunc
    if winfunc is not None:
        rsl *= winfunc
        
    # reshape back
    rsl = rsl.reshape(shp[:-1] + (n_row, frame_len))
        
    return rsl


def magspec(sig, nfft=None):
    '''
    Compute the magnitude spectrum of the input signals.

    Parameters
    ----------
    sig : array_like of shape ([B0, ..., Bn,] L)
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
    sig : array_like of shape ([B0, ..., Bn,] L)
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
    
    assert _env_consistency_check(sig), _err_msg_0
    
    # copy and reshape input array
    shp = sig.shape
    tmp = _reshape(sig)
    
    # apply FFT
    nfft = nfft or shp[-1]
    tmp = env.backend.fft.rfft(tmp, nfft)
    
    # compute power spectrum (un-normalized)
    tmp = env.backend.real(tmp * tmp.conj())
    
    # cast dtype back since fft may change dtype
    tmp = tmp.astype(env.dtype, copy=False)
    
    # divided by length to get power
    tmp = tmp / nfft
    
    # reshape back
    tmp = tmp.reshape(shp[:-1] + (nfft//2+1,))
    
    return tmp


def logpowspec(sig, nfft=None, norm=True):
    '''
    Compute the log power spectrum of the input signals.

    Parameters
    ----------
    sig : array_like of shape ([B0, ..., Bn,] L)
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
    tmp[tmp <= eps] = eps
    
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
    sig : array_like of shape ([B0, ..., Bn,] L)
        Input signals.
    coeff : int, optional
        The preemphasis coefficient. 0 is no filter, default is 0.97.

    Returns
    -------
    array_like of shape ([B0, ..., Bn,] L)
        The filtered signals.
    '''
    
    assert _env_consistency_check(sig), _err_msg_0
    
    # copy and reshape input array
    shp = sig.shape
    tmp = _reshape(sig, copy=True)
    
    # apply preemphasis filter
    tmp[:,1:] = tmp[:,1:] - coeff * tmp[:,:-1]
    
    # reshape back
    tmp = tmp.reshape(shp)
    
    return tmp