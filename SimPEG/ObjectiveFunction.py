from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division

import numpy as np
import scipy.sparse as sp
from six import integer_types
import warnings

from . import Utils
from .Tests import checkDerivative
from . import Maps
from . import Props

__all__ = [
    'BaseObjectiveFunction', 'ComboObjectiveFunction', 'L2ObjectiveFunction'
]


class BaseObjectiveFunction(Props.BaseSimPEG):

    counter = None
    debug = False

    mapPair = Maps.IdentityMap  #: Base class of expected maps
    _mapping = None  #: An IdentityMap instance.
    _hasFields = False  #: should we have the option to store fields

    _nP = None

    def __init__(self, nP=None, **kwargs):
        if nP is not None:
            self._nP = nP
        Utils.setKwargs(self, **kwargs)

    def __call__(self, x, f=None):
        raise NotImplementedError(
            '__call__ has not been implemented for {} yet'.format(
                self.__class__.__name__
            )
        )

    @property
    def nP(self):
        if self._nP is not None:
            return self._nP
        if getattr(self, 'mapping', None) is not None:
            return self.mapping.nP
        return '*'


    @property
    def mapping(self):
        if getattr(self, '_mapping') is None:
            if getattr(self, '_nP') is not None:
                self._mapping = self.mapPair(nP=self.nP)
            else:
                self._mapping = self.mapPair()
        return self._mapping

    @mapping.setter
    def mapping(self, value):
        assert isinstance(value, self.mapPair), (
            'mapping must be an instance of a {}, not a {}'
        ).format(self.mapPair, value.__class__.__name__)
        self._mapping = value


    @Utils.timeIt
    def __call__(self, x, f=None):
        raise NotImplementedError(
            "The method __call__ has not been implemented for {}".format(
                self.__class__.__name__
            )
        )

    @Utils.timeIt
    def deriv(self, x, **kwargs):
        raise NotImplementedError(
            "The method deriv has not been implemented for {}".format(
                self.__class__.__name__
            )
        )

    @Utils.timeIt
    def deriv2(self, x, v=None, **kwargs):
        raise NotImplementedError(
            "The method _deriv2 has not been implemented for {}".format(
                self.__class__.__name__
            )
        )

    def _test_deriv(self, x=None, num=4, plotIt=False, **kwargs):
        print('Testing {0!s} Deriv'.format(self.__class__.__name__))
        if x is None:
            if self.nP == '*':
                x = np.random.randn(np.random.randint(1e2, high=1e3))
            else:
                x = np.random.randn(self.nP)

        return checkDerivative(
            lambda m: [self(m), self.deriv(m)], x, num=num, plotIt=plotIt,
            **kwargs
        )

    def _test_deriv2(self, x=None, num=4, plotIt=False, **kwargs):
        print('Testing {0!s} Deriv2'.format(self.__class__.__name__))
        if x is None:
            if self.nP == '*':
                x = np.random.randn(np.random.randint(1e2, high=1e3))
            else:
                x = np.random.randn(self.nP)

        v = np.random.rand(len(x))
        return checkDerivative(
            lambda m: [self.deriv(m).dot(v), self.deriv2(m, v=v)], x, num=num,
            plotIt=plotIt, **kwargs
        )

    def test(self, x=None, num=4, plotIt=False, **kwargs):
        deriv = self._test_deriv(x=x, num=num, **kwargs)
        deriv2 = self._test_deriv2(x=x, num=num, plotIt=False, **kwargs)
        return (deriv & deriv2)

    __numpy_ufunc__ = True

    def __add__(self, objfct2):

        if isinstance(objfct2, Utils.Zero):
            return self

        if not isinstance(objfct2, BaseObjectiveFunction):
            raise Exception(
                "Cannot add type {} to an objective function. Only "
                "ObjectiveFunctions can be added together".format(
                    objfct2.__class__.__name__
                )
            )

        if self.__class__.__name__ != 'ComboObjectiveFunction': #not isinstance(self, ComboObjectiveFunction):
            self = 1 * self

        if objfct2.__class__.__name__ != 'ComboObjectiveFunction': #not isinstance(objfct2, ComboObjectiveFunction):
            objfct2 = 1 * objfct2

        objfctlist = self.objfcts + objfct2.objfcts
        multipliers = self._multipliers + objfct2._multipliers

        return ComboObjectiveFunction(
            objfcts=objfctlist, multipliers=multipliers
        )

    def __radd__(self, objfct2):
        return self+objfct2

    def __mul__(self, multiplier):
        return ComboObjectiveFunction([self], [multiplier])

    def __rmul__(self, multiplier):
        return self * multiplier

    def __div__(self, denominator):
        return self.__mul__(1./denominator)

    def __truediv__(self, denominator):
        return self.__mul__(1./denominator)

    def __rdiv__(self, denominator):
        return self.__mul__(1./denominator)


class ComboObjectiveFunction(BaseObjectiveFunction):

    _multiplier_types = (float, None, Utils.Zero) + integer_types # Directive

    def __init__(self, objfcts=[], multipliers=None, **kwargs):

        if multipliers is None:
            multipliers = len(objfcts)*[1]

        self._nP = '*'

        assert(len(objfcts)==len(multipliers)),(
            "Must have the same number of Objective Functions and Multipliers "
            "not {} and {}".format(len(objfcts),len(multipliers))
            )

        def validate_list(objfctlist, multipliers):
            for fct, mult in zip(objfctlist, multipliers):
                assert (
                    isinstance(fct, BaseObjectiveFunction)
                ) , (
                    "Unrecognized objective function type {} in objfcts. "
                    "All entries in objfcts must inherit from "
                    "ObjectiveFunction".format(fct.__class__.__name__)
                )

                assert(type(mult) in self._multiplier_types), (
                    "Objective Functions can only be multiplied by a "
                    "float, or a properties.Float, not a {}, {}".format(
                        type(mult), mult
                    )
                )

                if fct.nP != '*':
                    if self._nP != '*':
                        assert self._nP == fct.nP, (
                            "Objective Functions must all have the same "
                            "nP={}, not {}".format(self.nP, [f.nP for f in objfcts])
                        )
                    else:
                        self._nP = fct.nP

        validate_list(objfcts, multipliers)

        self.objfcts = objfcts
        self.__multipliers = multipliers

        super(ComboObjectiveFunction, self).__init__(**kwargs)

    def __len__(self):
        return len(self._multipliers)

    def __getitem__(self, key):
        return self._multipliers[key], self.objfcts[key]

    @property
    def __len__(self):
        return self.objfcts.__len__

    @property
    def _multipliers(self):
        return self.__multipliers

    def __call__(self, m, f=None):

        fct = 0.
        for i, phi in enumerate(self):
            multiplier, objfct = phi
            if multiplier == 0.: # don't evaluate the fct
                continue
            else:
                if f is not None and objfct._hasFields:
                    fct += multiplier * objfct(m, f=f[i])
                else:
                    fct += multiplier * objfct(m)
        return fct

    def deriv(self, m, f=None):
        g = Utils.Zero()
        for i, phi in enumerate(self):
            multiplier, objfct = phi
            if multiplier == 0.: # don't evaluate the fct
                continue
            else:
                if f is not None and objfct._hasFields:
                    g += multiplier * objfct.deriv(m, f=f[i])
                else:
                    g += multiplier * objfct.deriv(m)
        return g

    def deriv2(self, m, v=None, f=None):

        H = Utils.Zero()
        for i, phi in enumerate(self):
            multiplier, objfct = phi
            if multiplier == 0.: # don't evaluate the fct
                continue
            else:
                if f is not None and objfct._hasFields:
                    objfct_H = objfct.deriv2(m, v, f=f[i])
                else:
                    objfct_H = objfct.deriv2(m, v)
                H = H + multiplier * objfct_H
        return H

    # This assumes all objective functions have a W.
    # The base class currently does not.
    @property
    def W(self):
        W = []
        for mult, fct in self:
            curW = np.sqrt(mult) * fct.W
            if not isinstance(curW, Utils.Zero):
                W.append(curW)
        return sp.vstack(W)


class L2ObjectiveFunction(BaseObjectiveFunction):

    def __init__(self, W=None, **kwargs):

        super(L2ObjectiveFunction, self).__init__(**kwargs)
        if W is not None:
            if self.nP == '*':
                self._nP = W.shape[1]
            else:
                assert(W.shape[1]) == self.nP, (
                    'nP must be the same as W.shape[0], not {}'.format(self.nP)
                )
        self._W = W

    @property
    def W(self):
        if getattr(self, '_W', None) is None:
            self._W = Utils.Identity()
        return self._W

    def __call__(self, m):
        r = self.W * (self.mapping * m)
        return 0.5 * r.dot(r)

    def deriv(self, m):
        return (
            self.mapping.deriv(m).T *
            (self.W.T * (self.W * (self.mapping * m)))
        )

    def deriv2(self, m, v=None):
        if v is not None:
            return (
                self.mapping.deriv(m).T * (
                    self.W.T * (self.W * (self.mapping.deriv(m) * v))
                )
            )
        W = self.W * self.mapping.deriv(m)
        return self.W.T * self.W
