from __future__ import print_function
import numpy as np
from .Maps import IdentityMap

class Model(np.ndarray):

    def __new__(cls, input_array, mapping=None):
        assert isinstance(mapping, IdentityMap), 'mapping must be a SimPEG.Mapping'
        assert isinstance(input_array, np.ndarray), 'input_array must be a numpy array'
        assert len(input_array.shape) == 1, 'input_array must be a 1D vector'
        obj = np.asarray(input_array).view(cls)
        obj._mapping = mapping
        if not obj.size == mapping.nP:
            raise Exception('Incorrect size for array.')
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self._mapping = getattr(obj, '_mapping', None)

    @property
    def mapping(self):
        return self._mapping

    @property
    def transform(self):
        if getattr(self, '_transform', None) is None:
            self._transform = self.mapping * self.view(np.ndarray)
        return self._transform

    @property
    def transformDeriv(self):
        if getattr(self, '_transformDeriv', None) is None:
            self._transformDeriv = self.mapping.deriv(self.view(np.ndarray))
        return self._transformDeriv
