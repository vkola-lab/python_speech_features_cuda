"""
Created on Fri Aug  7 18:44:00 2020

@author: cxue2
"""

from ._env  import env
from ._buf  import buf

import numpy as np


def _mel_filterbank(samplerate, nfilt, nfft, lowfreq, highfreq):
    '''
    Compute a Mel-filterbank. If a filterbank has been computered before, the
    buffered result will be returned instead.

    Parameters
    ----------
    (Check the description of mfcc() for details)

    Returns
    -------
    array_like of shape (nfilt, nfft//2+1)
        Mel-filterbank.
    '''
    
    # look up buffer
    key = (samplerate, nfilt, nfft, lowfreq, highfreq)
    try:
        return buf['bnk'][key]
    except KeyError:
        pass
    
    # validate high frequence
    lowfreq = lowfreq or 0
    highfreq = highfreq or samplerate / 2
    assert highfreq <= samplerate / 2, 'highfreq is greater than samplerate/2'
    
    # compute points evenly spaced in mels
    mel_l = _hz2mel(lowfreq)
    mel_h = _hz2mel(highfreq)
    mel_pts = np.linspace(mel_l, mel_h, nfilt + 2)
    
    # convert from Hz to fft bin number
    idc = np.floor((nfft + 1) * _mel2hz(mel_pts) / samplerate).astype(np.int)

    # mel filter bank placeholder
    bnk = np.zeros((nfilt, nfft//2+1))
    
    # construct mel filter bank
    for m in range(1, nfilt + 1):
        
        f_m_l = idc[m-1]  # left
        f_m_c = idc[m]    # center
        f_m_r = idc[m+1]  # right
    
        for k in range(f_m_l, f_m_c):
            bnk[m-1,k] = (k - idc[m-1]) / (idc[m] - idc[m-1])
            
        for k in range(f_m_c, f_m_r):
            bnk[m-1,k] = (idc[m+1] - k) / (idc[m+1] - idc[m])
            
    # convert data type to be consistent with package environment
    bnk = env.backend.asarray(bnk, dtype=env.dtype)
            
    # save to buffer
    buf['bnk'][key] = bnk
    
    return bnk


def _dct_mat_type_2(nfilt, numcep):
    '''
    Compute a type-2 DCT matrix. If a matrix has been computered before, the
    buffered result will be returned instead.

    Parameters
    ----------
    nfilt : int
        DCT length.
    numcep : int, optional
        Length to return. 

    Returns
    -------
    array_like of shape (nfilt, nfilt)
        Type-2 DCT matrix.
    '''
    
    # look up buffer
    key = (nfilt, numcep)
    try:
        return buf['dct_mat'][key]
    except KeyError:
        pass
    
    # placeholder for dct matrix
    mat = np.zeros((nfilt, nfilt))
    
    # construct dct matrix
    for k in range(nfilt):
        mat[k,:] = np.pi * k * (2 * np.arange(nfilt) + 1) / (2 * nfilt)
        mat[k,:] = 2 * np.cos(mat[k,:])
        
    # truncate
    mat = np.copy(mat[:numcep,:])
        
    # convert data type to be consistent with package environment
    mat = env.backend.asarray(mat, dtype=env.dtype)
    
    # save to buffer
    buf['dct_mat'][key] = mat
        
    return mat


def _dct_scl_type_2(nfilt, numcep):
    '''
    Compute a type-2 DCT scaling vector. If a vector has been computered
    before, the buffered result will be returned instead.

    Parameters
    ----------
    nfilt : int
        DCT length.
    numcep : int, optional
        Length to return. 

    Returns
    -------
    array_like of shape (nfilt,)
        Type-2 DCT scaling vector.
    '''
    
    # look up buffer
    key = (nfilt, numcep)
    try:
        return buf['dct_scl'][key]
    except KeyError:
        pass
    
    # placeholder for dct scale
    vec = np.zeros((nfilt,))  # for orthogonal transformation
    
    # construct dct scale
    vec[0]  = np.sqrt(1 / 4 / nfilt)
    vec[1:] = np.sqrt(1 / 2 / nfilt)
    
    # truncate
    vec = np.copy(vec[:numcep])
    
    # convert data type to be consistent with package environment
    vec = env.backend.asarray(vec, dtype=env.dtype)
    
    # save to buffer
    buf['dct_scl'][key] = vec
    
    return vec


def _lifter(numcep, ceplifter):
    '''
    Compute a lifter. If a lifter has been computered before, the buffered
    result will be returned instead.

    Parameters
    ----------
    (Check the description of mfcc() for details)

    Returns
    -------
    array_like of shape (numcep,)
        Lifter vector.
    '''
    
    # look up buffer
    key = (numcep, ceplifter)
    try:
        return buf['lft'][key]
    except KeyError:
        pass
    
    # construct cepstrum lifter
    vec = 1 + (ceplifter / 2) * np.sin(np.pi * np.arange(numcep) / ceplifter)
    
    # convert data type to be consistent with package environment
    vec = env.backend.asarray(vec, dtype=env.dtype)
    
    # save to buffer
    buf['lft'][key] = vec
    
    return vec


def _hz2mel(hz_):
    
    return 2595. * np.log10(1. + hz_ / 700.)


def _mel2hz(mel):
    
    return 700. * (10. ** (mel / 2595.0) - 1.)