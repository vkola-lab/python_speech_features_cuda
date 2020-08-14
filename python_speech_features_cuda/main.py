#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 07:29:34 2020

@author: cxue2
"""

from ._env import _env_consistency_check
from ._env import env

from ._aux import _mel_filterbank
from ._aux import _dct_mat_type_2
from ._aux import _dct_scl_type_2
from ._aux import _lifter

from .sigproc import framesig
from .sigproc import powspec
from .sigproc import preemphasis

from . import _acc 


def mfcc(sig, samplerate=16000, winlen=.025, winstep=.01, numcep=13, nfilt=26,
         nfft=None, lowfreq=None, highfreq=None, preemph=.97, ceplifter=22,
         appendEnergy=True, winfunc=None):
    '''
    Extract MFCC features from the input signals.

    Parameters
    ----------
    sig : array_like of shape ([B0, ..., Bn,] signal_length)
        Input signals.
    samplerate : float, optional
        Sampling rate of the signal (in Hz). The default is 16000.
    winlen : float, optional
        Length of the frame (in seconds). The default is .025.
    winstep : float, optional
        Step between successive frames (in seconds). The default is .01.
    numcep : int, optional
        Number of cepstral coefficients to return. They are counted from the
        0th. The default is 13.
    nfilt : int, optional
        Number of filters in the Mel-filterbank. The default is 26.
    nfft : int, optional
        FFT length. If None is given, the length is assumed to be equal to the
        number of samples in a frame that is int(round(samplerate * winlen).
        The default is None.
    lowfreq : float, optional
        Lower bound of Mel-filterbank (in Hz). If None is given, the value will
        be assumed to be 0. The default is None.
    highfreq : float, optional
        Upper bound of Mel-filterbank (in Hz). If None is given, the value will
        be assumed to be samplerate / 2.  The default is None.
    preemph : float, optional
        Parameter for pre-emphasis filtering. Setting this value to 0 is
        equivalent to no filtering. The default is .97.
    ceplifter : float, optional
        Parameter for the liftering applied to the final cepstral coefficients.
        0 is equivalent to no liftering. The default is 22.
    appendEnergy : boolean, optional
        If the value is true, the 0th cepstral coefficient is replaced with the
        log of the total frame energy. The default is True.
    winfunc : array_like of shape (frame_length,), optional
        The analysis window applied to each frame. The default is None.

    Returns
    -------
    array_like of shape ([B0, ..., Bn,] #_of_frames, numcep)
        MFCC features of the input signals.
    '''

    # compute log mel filter bank spectrogram
    tmp, eng = logfbank(sig, samplerate, winlen, winstep, nfilt, nfft,
                        lowfreq, highfreq, preemph, appendEnergy, winfunc)
    
    # DCT and truncate
    tmp = tmp @ _dct_mat_type_2(nfilt, numcep).T
    tmp = _acc.mul(tmp, _dct_scl_type_2(nfilt, numcep))
    
    # apply lifter
    tmp = lifter(tmp, ceplifter)
    
    # replace first cepstral coefficient with log of frame energy
    if appendEnergy: tmp[...,0] = eng
    
    return tmp


def fbank(sig, samplerate=16000, winlen=.025, winstep=.01, nfilt=26, 
          nfft=None, lowfreq=None, highfreq=None, preemph=.97,
          calcEnergy=True, winfunc=None):
    '''
    Compute Mel-filterbank energies from the input signals.

    Parameters
    ----------
    calcEnergy : boolean, optional
        If the value is true, total energies for each frame will be calculated.
    (Check the description of mfcc() for details)        

    Returns
    -------
    array_like of shape ([B0, ..., Bn,] #_of_frames, nfilt)
        Mel-filterbank energies for each frame.
    array_like of shape ([B0, ..., Bn,] #_of_frames)
        Total energies for each frame. None if calcEnergy is false.
    '''
    
    # convert seconds to number of samples
    frm_len = round(samplerate * winlen)
    frm_stp = round(samplerate * winstep)
    
    # FFT length
    nfft = nfft or frm_len
    
    # pre-emphasis, framing and power-spectra all together
    tmp = _acc.preemp_frmsig_powspc(sig, frm_len, frm_stp, preemph, winfunc, nfft)
    
    # total energy
    eng = _acc.sum(tmp) if calcEnergy else None
    
    # apply mel filter bank
    bnk = _mel_filterbank(samplerate, nfilt, nfft, lowfreq, highfreq)
    tmp = tmp @ bnk.T
    
    return tmp, eng


def logfbank(sig, samplerate=16000, winlen=.025, winstep=.01, nfilt=26,
             nfft=None, lowfreq=None, highfreq=None, preemph=.97,
             calcEnergy=True, winfunc=None):
    '''
    Compute log Mel-filterbank energies from the input signals.

    Parameters
    ----------
    calcEnergy : boolean, optional
        If the value is true, total energies for each frame will be calculated.
    (Check the description of mfcc() for details)

    Returns
    -------
    array_like of shape ([B0, ..., Bn,] #_of_frames, nfilt)
        Log Mel-filterbank energies for each frame.
    array_like of shape ([B0, ..., Bn,] #_of_frames)
        Log total energies for each frame. None if calcEnergy is false.
    '''
    
    # compute mel filter bank spectrogram
    tmp, eng = fbank(sig, samplerate, winlen, winstep, nfilt, nfft, lowfreq,
                     highfreq, preemph, calcEnergy, winfunc)
    
    # compute log
    eps = env.backend.finfo(env.dtype).eps
    tmp = _acc.rplzro_log(tmp, eps, inplace=True)
    eng = _acc.rplzro_log(eng, eps, inplace=True) if eng is not None else None
    
    return tmp, eng


def ssc(sig, samplerate=16000, winlen=.025, winstep=.01, nfilt=26,
        nfft=None, lowfreq=None, highfreq=None, preemph=.97, winfunc=None):
    '''
    Compute Spectral Subband Centroid features from the input signals.

    Parameters
    ----------
    (Check the description of mfcc() for details)

    Returns
    -------
    array_like of shape ([B0, ..., Bn,] #_of_frames, nfilt)
        Spectral Subband Centroid features for each frame.
    '''
    
    # convert seconds to number of samples
    frm_len = round(samplerate * winlen)
    frm_stp = round(samplerate * winstep)
    
    # fft length
    nfft = nfft or frm_len
        
    # pre-emphasis, framing and power-spectra all together
    psp = _acc.preemp_frmsig_powspc(sig, frm_len, frm_stp, preemph, winfunc, nfft)
    
    # compute denominator
    bnk = _mel_filterbank(samplerate, nfilt, nfft, lowfreq, highfreq)
    dnm = psp @ bnk.T
    
    # eliminate zeros for denominator
    eps = env.backend.finfo(env.dtype).eps
    dnm = env.backend.where(dnm == 0, eps, dnm)
    
    # the last step
    vec = env.backend.linspace(1, samplerate/2, psp.shape[-1], dtype=env.dtype)
    tmp = (psp * vec) @ bnk.T / dnm
    
    return tmp


def hz2mel(hz_):
    '''
    Convert values in Hz to Mel.

    Parameters
    ----------
    hz_ : array_like of any shape
        Values in Hz.

    Returns
    -------
    array_like of shape same to the input
        Values in Mel.
    '''
    
    _env_consistency_check(hz_)
    
    return 2595. * env.backend.log10(1. + hz_ / 700.)


def mel2hz(mel):
    '''
    Convert values in Mel to Hz.

    Parameters
    ----------
    hz_ : array_like of any shape
        Values in Mel.

    Returns
    -------
    array_like of shape same to the input
        Values in Hz.
    '''
    
    _env_consistency_check(mel)
    
    return 700. * (10. ** (mel / 2595.0) - 1.)


def get_filterbanks(samplerate=16000, nfilt=26, nfft=512, lowfreq=None, highfreq=None):
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
    
    return _mel_filterbank(samplerate, nfilt, nfft, lowfreq, highfreq)


def lifter(cep, ceplifter=22):
    '''
    Apply a lifter the the matrix of cepstra. This has the effect of increasing
    the magnitude of the high frequency DCT coefficients.

    Parameters
    ----------
    cep : array_like of shape ([B0, ..., Bn,] #_of_frames, numcep)
        Cepstral coefficients for each frame.
    ceplifter : float, optional
        Parameter for the liftering applied to the final cepstral coefficients.
        0 is equivalent to no liftering. The default is 22.

    Returns
    -------
    array_like of shape ([B0, ..., Bn,] #_of_frames, numcep)
        Cepstral coefficients after liftering.
    '''
    
    _env_consistency_check(cep)
    
    if ceplifter > 0:
        return _acc.mul(cep, _lifter(cep.shape[-1], ceplifter))
        
    else:
        return cep       


def delta(fea, n=2):
    '''
    Compute delta features.

    Parameters
    ----------
    fea : array_like of shape ([B0, ..., Bn,] #_of_frames, feature_length)
        Input features.
    n : int, optional
        Number of neighbor frames. The default is 2.

    Returns
    -------
    array_like of shape ([B0, ..., Bn,] #_of_frames, feature_length)
        Delta features.
    '''
    
    _env_consistency_check(fea)
    assert type(n) is int and n > 0, 'n must be an integer greater than 0.'
    
    # delta feature placeholder
    dlt = env.backend.zeros_like(fea)
    
    # computation loop
    for i in range(-n, n+1):
        
        if i < 0:
            dlt[...,-i:,:] += fea[...,:i,:] * i
            
        elif i > 0:
            dlt[...,:-i,:] += fea[...,i:,:] * i
    
    # divide by 2 times the square sum of 1, ..., n
    dlt /= 2 * sum([i * i for i in range(1, n+1)])
           
    return dlt