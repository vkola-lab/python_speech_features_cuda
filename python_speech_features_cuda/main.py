#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 07:29:34 2020

@author: cxue2
"""

from ._misc import _env_consistency_check
from ._env  import env
from ._buf  import buf

from .sigproc import framesig
from .sigproc import powspec
from .sigproc import preemphasis

import numpy as np


def mfcc(sig, samplerate=16000, winlen=.025, winstep=.01, numcep=13, nfilt=26,
         nfft=None, lowfreq=0, highfreq=None, preemph=.97, ceplifter=22,
         appendEnergy=True, winfunc=None):

    # compute log mel filter bank spectrogram
    tmp, eng = logfbank(sig, samplerate, winlen, winstep, nfilt, nfft,
                        lowfreq, highfreq, preemph, winfunc)
    
    # DCT
    tmp = tmp @ _dct_mat_type_2(nfilt).T
    tmp = tmp * _dct_scl_type_2(nfilt)
    
    # truncate cepstral coefficients
    tmp = tmp[...,:numcep]
    
    # apply lifter
    tmp = lifter(tmp)
    
    # replace first cepstral coefficient with log of frame energy
    if appendEnergy: tmp[...,0] = eng
    
    return tmp


def fbank(sig, samplerate=16000, winlen=.025, winstep=.01, nfilt=26, 
          nfft=None, lowfreq=0, highfreq=None, preemph=.97, winfunc=None):
    
    # preemphasis
    tmp = preemphasis(sig, coeff=preemph)
    
    # split signals into frames
    frm_len = int(np.round(samplerate * winlen))
    frm_stp = int(np.round(samplerate * winstep))
    tmp = framesig(tmp, frm_len, frm_stp, winfunc=winfunc)
    
    # compute power spectrum
    nfft = nfft or tmp.shape[-1]
    tmp = powspec(tmp, nfft=nfft)
    
    # total energy
    eng = env.backend.sum(tmp, axis=-1)
    
    # apply mel filter bank
    bnk = get_filterbanks(samplerate, nfilt, nfft, lowfreq, highfreq)
    tmp = tmp @ bnk.T
    
    return tmp, eng


def logfbank(sig, samplerate=16000, winlen=.025, winstep=.01, nfilt=26,
             nfft=None, lowfreq=0, highfreq=None, preemph=.97, winfunc=None):
    
    # compute mel filter bank spectrogram
    tmp, eng = fbank(sig, samplerate, winlen, winstep, nfilt, nfft, lowfreq,
                     highfreq, preemph, winfunc)
    
    # numerical stability for log
    eps = env.backend.finfo(env.dtype).eps
    tmp = env.backend.where(tmp == 0, eps, tmp)
    eng = env.backend.where(eng == 0, eps, eng)
    
    return env.backend.log(tmp), env.backend.log(eng)


def ssc(sig, samplerate=16000, winlen=.025, winstep=.01, nfilt=26,
        nfft=None, lowfreq=0, highfreq=None, preemph=.97, winfunc=None):
    
    highfreq = highfreq or samplerate/2
    
    # preemphasis
    tmp = preemphasis(sig, preemph)
    
    # split signals into frames
    frm_len = int(np.round(samplerate * winlen))
    frm_stp = int(np.round(samplerate * winstep))
    tmp = framesig(tmp, frm_len, frm_stp, winfunc=winfunc)
    
    # calculate power spectrum
    psp = powspec(tmp, nfft)
    eps = env.backend.finfo(env.dtype).eps
    psp = env.backend.where(psp == 0, eps, psp)
    
    # apply mel filter bank
    bnk = get_filterbanks(samplerate, nfilt, nfft, lowfreq, highfreq)
    fea = psp @ bnk.T
    
    vec = env.backend.linspace(1, samplerate/2, psp.shape[-1], dtype=env.dtype)
    tmp = (psp * vec) @ bnk.T / fea
    
    return tmp


def hz2mel(hz_):
    
    _env_consistency_check(hz_)
    
    return 2595. * env.backend.log10(1. + hz_ / 700.)


def _hz2mel(hz_):
    
    return 2595. * np.log10(1. + hz_ / 700.)


def mel2hz(mel):
    
    _env_consistency_check(mel)
    
    return 700. * (10. ** (mel / 2595.0) - 1.)


def _mel2hz(mel):
    
    return 700. * (10. ** (mel / 2595.0) - 1.)


def get_filterbanks(samplerate=16000, nfilt=26, nfft=512, lowfreq=0, highfreq=None):
    
    # look up buffer
    key = (samplerate, nfilt, nfft, lowfreq, highfreq)
    try:
        return buf.hmp['bnk'][key]
    except KeyError:
        pass
    
    # validate high frequence
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
    buf.hmp['bnk'][key] = bnk
    
    return bnk


def lifter(cep, ceplifter=22):
    
    _env_consistency_check(cep)
    
    return cep * _lifter(cep.shape[-1], ceplifter) if ceplifter > 0 else cep        


def delta(fea, n=2):
    
    assert np.issubdtype(type(n), np.int) and n > 0, 'n must be an integer greater than 0.'
    
    # delta feature placeholder
    dlt = env.backend.zeros_like(fea)
    
    # computation loop
    for i in range(-n, n+1):
        
        if i < 0:
            dlt[...,-i:,:] += fea[...,:i,:] * i
            
        elif i > 0:
            dlt[...,:-i,:] += fea[...,i:,:] * i
        
        else:
            continue
    
    # divide by 2 times the square sum of 1, ..., n
    dlt /= 2 * np.sum(np.arange(1, n+1) ** 2, dtype=env.dtype)
           
    return dlt
    

def _lifter(numcep, ceplifter):
    
    # look up buffer
    key = (numcep, ceplifter)
    try:
        return buf.hmp['lft'][key]
    except KeyError:
        pass
    
    # construct cepstrum lifter
    vec = 1 + (ceplifter / 2) * np.sin(np.pi * np.arange(numcep) / ceplifter)
    
    # convert data type to be consistent with package environment
    vec = env.backend.asarray(vec, dtype=env.dtype)
    
    # save to buffer
    buf.hmp['lft'][key] = vec
    
    return vec


def _dct_mat_type_2(nfilt):
    
    # look up buffer
    try:
        return buf.hmp['dct_mat'][nfilt]
    except KeyError:
        pass
    
    # placeholder for dct matrix
    mat = np.zeros((nfilt, nfilt))
    
    # construct dct matrix
    for k in range(nfilt):
        mat[k,:] = np.pi * k * (2 * np.arange(nfilt) + 1) / (2 * nfilt)
        mat[k,:] = 2 * np.cos(mat[k,:])
        
    # convert data type to be consistent with package environment
    mat = env.backend.asarray(mat, dtype=env.dtype)
    
    # save to buffer
    buf.hmp['dct_mat'][nfilt] = mat
        
    return mat


def _dct_scl_type_2(nfilt):
    
    # look up buffer
    try:
        return buf.hmp['dct_scl'][nfilt]
    except KeyError:
        pass
    
    # placeholder for dct scale
    vec = np.zeros((nfilt,))  # for orthogonal transformation
    
    # construct dct scale
    vec[0]  = np.sqrt(1 / 4 / nfilt)
    vec[1:] = np.sqrt(1 / 2 / nfilt)
    
    # convert data type to be consistent with package environment
    vec = env.backend.asarray(vec, dtype=env.dtype)
    
    # save to buffer
    buf.hmp['dct_scl'][nfilt] = vec
    
    return vec