"""
Created on Sun Aug  2 06:54:47 2020

@author: cxue2
"""

from ._env import env

# package global buffer class
class _Buf:
    
    def __init__(self):
        
        self._hmp_np_32 = {'bnk': dict(), 'dct_mat': dict(), 'dct_scl': dict(), 'lft': dict()}
        self._hmp_np_64 = {'bnk': dict(), 'dct_mat': dict(), 'dct_scl': dict(), 'lft': dict()}
        self._hmp_cp_32 = {'bnk': dict(), 'dct_mat': dict(), 'dct_scl': dict(), 'lft': dict()}
        self._hmp_cp_64 = {'bnk': dict(), 'dct_mat': dict(), 'dct_scl': dict(), 'lft': dict()}    
    
    
    @property
    def hmp(self):
        
        if env.backend.__name__ == 'cupy' and env.dtype.__name__ == 'float32':
            return self._hmp_cp_32
        
        elif env.backend.__name__ == 'cupy' and env.dtype.__name__ == 'float64':
            return self._hmp_cp_64
        
        elif env.backend.__name__ == 'numpy' and env.dtype.__name__ == 'float32':
            return self._hmp_np_32
        
        elif env.backend.__name__ == 'numpy' and env.dtype.__name__ == 'float64':
            return self._hmp_np_64
    
        
    def reset(self):
        
        self.__init__()

        
buf = _Buf()
        
        