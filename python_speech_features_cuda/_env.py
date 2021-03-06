"""
Created on Fri Jul 31 17:10:55 2020

@author: cxue2
"""

from types import ModuleType


# control package environment variables   
class _Env:
    
    def __init__(self):        
        
        self._backend = None
        self._np = None  # for numpy package
        self._cp = None  # for cupy package
        self._nb = None  # for numba package
        self._fw = None  # for pyfftw package
        self._dtype = None
        
        self._use_nb = True
        self._use_fw = True
        
        from multiprocessing import cpu_count
        from math import ceil
        self._n_threads = ceil(cpu_count() / 2)
        
        # import numpy (must be available)
        import numpy as np
        self._np = np
        
        # is cupy available
        try:
            import cupy as cp
            self._cp = cp
            
        except ImportError:
            pass
            
        # is numba available
        try:
            # set number of threads for numba
            import os
            os.environ['NUMBA_NUM_THREADS'] = str(self._n_threads)
            
            import numba
            self._nb = numba
            
        except ImportError:
            pass
            
        # is pyfftw available
        try:
            import pyfftw
            self._fw = pyfftw
            
        except ImportError:
            pass

        # assign environment variable: backend            
        self._backend = self._cp or self._np
        
        # assign environment variable: dtype
        self._dtype = self._np.float64
            
    
    @property
    def backend(self):
        
        return self._backend
    
    
    @backend.setter
    def backend(self, be):
        
        # assertions
        assert isinstance(be, ModuleType), 'The backend needs to be a module.'
        assert be.__name__ in ('cupy', 'numpy'), 'Only numpy or cupy can be assigned.'
        
        # set
        self._backend = be
        
    
    @property
    def dtype(self):
        
        return self._dtype
    
    
    @dtype.setter
    def dtype(self, dt):
        
        # assertions
        assert dt in (self.np.float32, self.np.float64), \
            'Only numpy.float32 or numpy.float64 is supported.'
        
        # set
        self._dtype = dt
        
    
    @property
    def np(self):
        
        return self._np
    
    
    @property
    def cp(self):
        
        return self._cp
    
    
    @property
    def nb(self):
        
        return self._nb
    
    
    @property
    def fw(self):
        
        return self._fw
        
    
    @property
    def is_cupy_available(self):
        
        return self._cp is not None
    
    
    @property
    def is_numba_available(self):
        
        return self._nb is not None
    
    
    @property
    def is_pyfftw_available(self):
        
        return self._fw is not None
    
    
    @property
    def use_numba(self):
        
        return self._use_nb and self.is_numba_available and self.backend.__name__ == 'numpy'
    
    
    @use_numba.setter
    def use_numba(self, val):
        
        assert type(val) is bool
        self._use_nb = val
    
    
    @property
    def use_pyfftw(self):
        
        return self._use_fw and self.is_pyfftw_available and self.backend.__name__ == 'numpy'
    
    
    @use_pyfftw.setter
    def use_pyfftw(self, val):
        
        assert type(val) is bool
        self._use_fw = val
    
    
    @property
    def n_threads(self):
        
        return self._n_threads
    
        
    @property
    def flags(self):
        ''' For package buffer. '''
        
        flg = 0
        
        if self.backend.__name__ == 'cupy': flg += 1
        flg <<= 1
        
        if self.use_numba: flg += 1
        flg <<= 1
        
        if self.use_pyfftw: flg += 1
        flg <<= 1
        
        if self.dtype is self.np.float64: flg += 1
        
        return flg
 
    
env = _Env()


def _env_consistency_check(arr):

    # backend check
    if env.backend is env.cp and type(arr) is env.cp.core.core.ndarray:
        pass
    
    elif env.backend is env.np and type(arr) is env.np.ndarray:
        pass
        
    else:
        msg = 'The input array is {} while the backend is set to be <{}>.'.format(type(arr), env.backend.__name__)
        raise TypeError(msg)
        
    # dtype check
    if arr.dtype.type is env.dtype:
        pass
        
    else:
        msg = 'The dtype of the input array is <{}> while the environment dtype is set to be <{}>.'.format(arr.dtype.type.__name__, env.dtype.__name__)
        raise TypeError(msg)    