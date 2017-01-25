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

    _nP = None

    def __init__(self, **kwargs):
        Utils.setKwargs(self, **kwargs)

    def __call__(self, x, **kwargs):
        return self._eval(x, **kwargs)

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
        assert issubclass(value, self.mapPair), (
            'mapping must be an instance of a {}, not a {}'
        ).format(self.mapPair, value.__class__.__name__)


    @Utils.timeIt
    def _eval(self, x, **kwargs):
        raise NotImplementedError(
            "The method _eval has not been implemented for {}".format(
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
    def deriv2(self, x, **kwargs):
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

        return checkDerivative(
            lambda m: [self.deriv(m), self.deriv2(m)], x, num=num,
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

        if not isinstance(self, ComboObjectiveFunction):
            self = 1 * self

        if not isinstance(objfct2, ComboObjectiveFunction):
            objfct2 = 1 * objfct2

        objfctlist = self.objfcts + objfct2.objfcts
        multipliers = self.multipliers + objfct2.multipliers

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

        self._nP = '*'

        if multipliers is None:
            multipliers = len(objfcts)*[1]

        def validate_list(objfctlist, multipliers):
            for fct, mult in zip(objfctlist, multipliers):
                if isinstance(fct, list):
                    validate_list(fct, multipliers)
                else:
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
                        if self.nP != '*':
                            assert self.nP == fct.nP, (
                                "Objective Functions must all have the same "
                                "nP, not {}".format([f.nP for f in objfcts])
                            )
                        else:
                            self._nP = fct.nP

        validate_list(objfcts, multipliers)

        self.objfcts = objfcts
        self._multipliers = multipliers

        super(ComboObjectiveFunction, self).__init__(**kwargs)

    @property
    def multipliers(self):
        return self._multipliers

    @multipliers.setter
    def multipliers(self, value):
        self._multipliers = value


    def _eval(self, x, **kwargs):

        def eval_list(objfcts, multipliers, f):
            for multiplier, objfct in zip(multipliers, objfcts):
                if isinstance(multiplier, Utils.Zero) or multiplier == 0.: # don't evaluate the fct
                    pass
                elif isinstance(objfct, list):
                    f += eval_list(objfct, multiplier, f)
                else:
                    f += multiplier * objfct(x, **kwargs)
            return f

        f = eval_list(self.objfcts, self.multipliers, 0.0)

        return f

    def deriv(self, x, **kwargs):
        def eval_listDerivs(objfcts, multipliers, g):
            for multiplier, objfct in zip(multipliers, objfcts):
                if isinstance(multiplier, Utils.Zero) or multiplier == 0.: # don't evaluate the fct
                    pass
                elif isinstance(objfct, list):
                    g += eval_listDerivs(objfct, multiplier, g)
                else:
                    g += multiplier * objfct.deriv(x, **kwargs)
            return g

        g = eval_listDerivs(self.objfcts, self.multipliers, Utils.Zero())
        return g

    def deriv2(self, x, **kwargs):

        def eval_list2Derivs(objfcts, multipliers, H):
            for multiplier, objfct in zip(multipliers, objfcts):
                if isinstance(multiplier, Utils.Zero) or multiplier == 0.: # don't evaluate the fct
                    pass
                elif isinstance(objfct, list):
                    H = H + eval_list2Derivs(objfct, multiplier, H)

                else:
                    objfct_H = objfct.deriv2(x, **kwargs)
                    if isinstance(objfct_H, Utils.Identity):
                        if self.nP != '*':
                            objfct_H = sp.eye(self.nP)  # if we need a shape
                        elif objfct.nP != '*':
                            objfct_H = sp.eye(objfct.nP)  # if we need a shape
                        else:
                            raise Exception('need to set nP on objective functions to get H')
                    H = H + multiplier * objfct_H
            return H

        H = eval_list2Derivs(self.objfcts, self.multipliers, Utils.Zero())
        return H

        # H = Utils.Zero()
        # for multpliter, objfct in zip(self.multipliers, self.objfcts):
        #     objfct_H = objfct.deriv2(x, **kwargs)
        #     if isinstance(objfct_H, Utils.Zero):
        #         pass
        #
        #     else:
        #         H += multpliter * objfct_H
        # return H

    # This assumes all objective functions have a W.
    # The base class currently does not.
    @property
    def W(self):
        W = []
        for mult, fct in zip(self.multipliers, self.objfcts):
            curW = mult * fct.W
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
            if self._nP != '*' and self._nP is not None:
                self._W = sp.eye(self.nP)
            else:
                self._W = Utils.Identity()
        return self._W

    def _eval(self, m):
        r = self.W * m
        return 0.5 * r.dot(r)

    def deriv(self, m):
        return self.W.T * (self.W * m)

    def deriv2(self, m):
        return self.W.T * self.W
