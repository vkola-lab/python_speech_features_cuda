"""
Created on Sun Aug  9 15:38:41 2020

@author: cxue2
"""

from .._env import env
from .._buf import buf


class _FFT():    
    
    def __call__(self, arr, nfft):
        
        # call FFTW
        if env.use_pyfftw:
            obj = self._fftw_obj(arr.shape, nfft)
            obj.input_array[...,:arr.shape[-1]] = arr[:]  # copy to placeholder
            return obj()
        
        # call either numpy or cupy's fft function
        else:
            return env.np.fft.rfft(arr, nfft)


    def _fftw_obj(self, shp, nfft):
        '''
        Generate FFTW object for Fast Fourier Transform. A buffered one will be
        returned if it has already been created.
    
        Parameters
        ----------
        shp : tuple
            Input shape.
        nfft : int
            FFT length.
    
        Returns
        -------
        pyfftw.pyfftw.FFTW
            FFTW Object.
        '''
        
        # look up buffer
        key = shp + (nfft,)
        try:
            return buf['fft_obj'][key]
        except KeyError:
            pass
        
        if env.dtype is env.np.float32:
            in_ = env.fw.empty_aligned(shp[:-1] + (nfft,), dtype='float32')
            out = env.fw.empty_aligned(shp[:-1] + (nfft//2+1,), dtype='complex64')
        
        else:
            in_ = env.fw.empty_aligned(shp[:-1] + (nfft,), dtype='float64')
            out = env.fw.empty_aligned(shp[:-1] + (nfft//2+1,), dtype='complex128')
            
        # construct fft object
        obj = env.fw.FFTW(in_, out, flags=['FFTW_MEASURE'], threads=env.n_threads)
        
        # reset input placeholder to 0s
        obj.input_array[:] = 0
            
        # save to buffer
        buf['fft_obj'][key] = obj
        
        return obj
    

fft = _FFT()