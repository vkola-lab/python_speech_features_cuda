"""
Created on Fri Jul 31 17:10:55 2020

@author: cxue2
"""

import os
from types import ModuleType

# control package environment variables   
class _Env:
    
    def __init__(self):
        
        self._backend = None
        self._dtype = None
        self._is_cupy_available = None
        
        # import numpy
        import numpy as np
        
        # is cupy available
        try:
            import cupy as cp
            self._is_cupy_available = True
            
        except ImportError:
            self._is_cupy_available = False 
        
        # assign environment variable: backend
        try:
            assert os.environ['PSF_BACKEND'] in ('cupy', 'numpy')
            if os.environ['PSF_BACKEND'] == 'numpy': self._is_cupy_available = False
        
        except KeyError:  # if os.environ['PSF_BACKEND'] is not set
            os.environ['PSF_BACKEND'] = 'cupy' if self._is_cupy_available else 'numpy'
            
        self._backend = cp if self._is_cupy_available else np
        
        # assign environment variable: dtype
        try:
            assert os.environ['PSF_DTYPE'] in ('float32', 'float64')
            self._dtype = np.float32 if os.environ['PSF_DTYPE'] == 'float32' else np.float64
            
        except KeyError:  # if os.environ['PSF_DTYPE'] is not set
            os.environ['PSF_DTYPE'] = 'float32'
            self._dtype = np.float32
            
    
    @property
    def backend(self):
        
        return self._backend
    
    
    @backend.setter
    def backend(self, be):
        
        # assertions
        assert isinstance(be, ModuleType), 'The backend to assign needs to be a module.'
        assert be.__name__ in ('cupy', 'numpy'), 'Only numpy or cupy can be assigned.'
        
        # set
        self._backend = be
        os.environ['PSF_BACKEND'] = be.__name__
        
    
    @property
    def dtype(self):
        
        return self._dtype
    
    
    @dtype.setter
    def dtype(self, dt):
        
        # assertions
        assert type(dt) is type, 'dtype needs to be a type object.'
        assert dt.__name__ in ('float32', 'float64'), 'Only float32 or float64 is supported.'
        
        # set
        self._dtype = dt
        os.environ['PSF_DTYPE'] = dt.__name__
        
    
    @property
    def is_cupy_available(self):
        
        return self._is_cupy_available
        
    
env = _Env()     