"""
Created on Sun Aug  2 06:54:47 2020

@author: cxue2
"""

from ._env import env

# package buffer class
class _Buf:
    
    def __init__(self):
        
        self._hmp = {}
        
        
    def __getitem__(self, key):
        
        # get environment flags as key
        key_env = env.flags
        
        # initialize hash map
        if key_env not in self._hmp:
            self._hmp[key_env] = {'bnk': {}, 'fft_obj': {}, 'dct_mat': {},
                                  'dct_scl': {}, 'lft': {}}
        
        return self._hmp[key_env][key]
     

    def reset(self):
        
        self.__init__()

        
buf = _Buf()
        
        