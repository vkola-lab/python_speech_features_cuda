"""
Created on Sat Aug  8 07:10:34 2020

@author: cxue2
"""
# import os
# os.environ['NUMBA_NUM_THREADS'] = '6'

import numba as nb
from numba import float64, void
from math import ceil


def _dct_mat_type_2(nfilt):

    # placeholder for dct matrix
    mat = np.zeros((nfilt, nfilt))
    
    # construct dct matrix
    for k in range(nfilt):
        mat[k,:] = np.pi * k * (2 * np.arange(nfilt) + 1) / (2 * nfilt)
        mat[k,:] = 2 * np.cos(mat[k,:])
        
    return mat


def _dct_scl_type_2(nfilt):
    
    # placeholder for dct scale
    vec = np.zeros((nfilt,))  # for orthogonal transformation
    
    # construct dct scale
    vec[0]  = np.sqrt(1 / 4 / nfilt)
    vec[1:] = np.sqrt(1 / 2 / nfilt)
    
    return vec

_par = True

@nb.njit(parallel=True, cache=True)
def _preemp_frmsig(in_, out, frm_stp, preemph, win):

    for i in range(out.shape[0]):
        for j in nb.prange(out.shape[1]): 
            for k in range(out.shape[2]):
                
                # index for input signal
                idx = frm_stp * j + k
                
                if 0 < idx < in_.shape[1]:
                    out[i,j,k] = (in_[i,idx] - preemph * in_[i,idx-1]) * win[k]
                elif idx != 0:
                    out[i,j,k] = 0
                else:
                    out[i,0,0] = in_[i,0] * win[0]


@nb.njit(parallel=_par, cache=True)
def _jit_preemp_frmsig(sig, out, frm_stp, preemph, win):
    
    bnd_i, bnd_j = out.shape
    
    for i in nb.prange(bnd_i): 
        for j in range(bnd_j):
            
            # index for input signal
            idx = frm_stp * i + j
            
            if 0 < idx < len(sig):
                out[i,j] = (sig[idx] - preemph * sig[idx-1]) * win[j]
            elif idx != 0:
                out[i,j] = 0
            else:
                out[0,0] = sig[0] * win[0]
                
                
@nb.njit(parallel=_par, cache=True)
def _jit_pow_div(in_, out, nfft):
    
    for i in nb.prange(in_.shape[0]):
        for j in range(in_.shape[1]):
            re, im = in_[i,j].real, in_[i,j].imag
            out[i,j] = (re * re + im * im) / nfft
            
            
# @nb.njit(parallel=_par, cache=True)
# def _jit_matmul(a, b):
    
#     # operand dimensions
#     I, K = a.shape
#     _, J = b.shape
    
#     blk_siz_x, blk_siz_y = 4, 4
#     blk_cnt_x = ceil(K / blk_siz_x)
#     blk_cnt_y = ceil(I / blk_siz_y)
    
#     # allocate output matrix
#     out = np.zeros((I, J))
    
#     for blk_idx_x in nb.prange(blk_cnt_x):
#         for blk_idx_y in nb.prange(blk_cnt_y):
            
#             blk_off_x = blk_siz_x * blk_idx_x
#             blk_off_y = blk_siz_y * blk_idx_y
    
#             for i_ in range(blk_siz_y):
#                 for j_ in range(blk_siz_x):
                    
#                     i = blk_off_y + i_
#                     j = blk_off_x + j_
                    
#                     if i >= I or j >= J: continue
                    
#                     for k in range(K):
#                         out[i,j] += a[i,k] * b[k,j]
            
#     return out

@nb.njit(parallel=_par, cache=True, fastmath=True)
def _jit_matmul_ele(a, b):
    
    # operand dimensions
    I, K = a.shape
    _, J = b.shape
    
    out = np.empty((I,J), dtype=a.dtype)
    
    for i in nb.prange(I):
        for j in range(J):
            tmp = 0
            for k in range(K):
                tmp += a[i,k] * b[k,j]
            out[i,j] = tmp
       

@nb.njit(parallel=_par, cache=True)
def _jit_matmul_vec(a, b):
    
    # operand dimensions
    I, K = a.shape
    _, J = b.shape
    
    out = np.empty((I,J), dtype=a.dtype)
    
    for i in nb.prange(I):
        out[i] = np.dot(a[i,:], b[:,i])
            
    return out

@nb.njit(parallel=False, cache=True)
def _jit_matmul_mat(a, b):
    return np.dot(a, b)


# for testing purpose only
if __name__ == '__main__':

    import numpy as np
    from math import ceil
    import python_speech_features as psf
    import python_speech_features_cuda as psfc
    from timeit import default_timer
    # import pyfftw
    
    psfc.env.dtype = np.float32
    
    # pyfftw.config.NUM_THREADS = 12
    # pyfftw.config.PLANNER_EFFORT = 'FFTW_MEASURE'
    
    psfc.env.backend = np
    
    dct_mat = _dct_mat_type_2(13)
    dct_scl = _dct_scl_type_2(13)
    
    frm_len = 400
    frm_stp = 160
    preemph = .97
    nfft = 512
    
    win = np.ones(frm_len, dtype=psfc.env.dtype)
    
    sig = np.random.rand(2, 500000).astype(psfc.env.dtype)
    sig_ = sig[1,:]
    
    # calculate number of frames
    if len(sig) > frm_len:
        n_frm = 1 + ceil((len(sig) - frm_len) / frm_stp)
        
    else:
        n_frm = 1
    
    # test loop
    n_iter = 100
    
    # %timeit out = np.empty((n_frm, frm_len), dtype=np.float64); \\
    #         _jit_preemp_frmsig(sig, out, frm_stp, preemph, win)
    
    beg = default_timer()
    for _ in range(n_iter):
        out_psf = psf.sigproc.preemphasis(sig_, preemph)
        out_psf = psf.sigproc.framesig(out_psf, frm_len, frm_stp)
        out_psf = psf.sigproc.powspec(out_psf, nfft)
        
        # out_psf = fftpack.dct(out_psf)
        # out_psf = psf.mfcc(sig, nfft=nfft)
    end = default_timer()
    tim_psf = (end - beg) / n_iter
    print('psf:\t{}'.format(tim_psf))
    
    n_iter = 100
    
    beg = default_timer()
    for _ in range(n_iter):
        out = psfc._acc._preemp_frmsig_powspc(sig_, frm_len, frm_stp, preemph, None, nfft)
        # out = np.empty((1, n_frm, frm_len), dtype=np.float64)
        # _preemp_frmsig(sig_, out, frm_stp, preemph, win)
        # out = np.empty((n_frm, frm_len), dtype=np.float64)
        # _jit_preemp_frmsig(sig, out, frm_stp, preemph, win)
        # tmp = pyfftw.interfaces.numpy_fft.rfft(out, nfft)
        # tmp = np.fft.rfft(out, nfft)
        # out = np.empty_like(tmp, dtype=np.float64)
        # _jit_pow_div(tmp, out, nfft)
        # tmp = np.empty_like(out, dtype=np.float64)
        # _jit_dct(out, tmp, dct_mat, dct_scl)
    end = default_timer()
    tim = (end - beg) / n_iter
    print('psf:\t{}'.format(tim))
    
    print('speedup:\t{:.2f}'.format(tim_psf / tim))
    print('allclose:\t{}'.format(np.allclose(out, out_psf)))
    
    # _preemp_frmsig.parallel_diagnostics(level=4)
    
    # a = np.random.rand(2048, 2048)
    # b = np.random.rand(2048, 2048)
    
    # c = np.random.rand(1, 2048)
    # d = np.random.rand(1, 2048)
    
    
    # a = pyfftw.empty_aligned((3124, 512), dtype='complex128')
    # a[:] = np.random.randn(3124, 512) + 1j*np.zeros((3124, 512))
    # fft_dat = np.random.rand(3124, 512)
    # fft_cmp = fft_dat.astype(np.complex128)
    # fft_mat = np.random.rand(512, 512).astype(np.complex128)
    
    # t = np.random.rand(3124, 400)
    
    # a = pyfftw.empty_aligned((3124, 512), dtype='float64')
    # b = pyfftw.empty_aligned((3124, 257), dtype='complex128')
    
    # fft_object = pyfftw.FFTW(a, b, flags=['FFTW_MEASURE'], threads=12)
    # a[:] = np.random.rand(3124, 512)
    
    
    