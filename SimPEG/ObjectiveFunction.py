from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

from properties import Float
import numpy as np
import scipy.sparse as sp
from six import integer_types
import warnings

from . import Utils
from .Tests import checkDerivative
from . import Maps


class BaseObjectiveFunction(object):

    counter = None
    debug = False

    def __init__(self, nP=None, **kwargs):
        self._nP = nP
        Utils.setKwargs(self, **kwargs)

    def __call__(self, x, **kwargs):
        return self._eval(x, **kwargs)

    @property
    def nP(self):
        if getattr(self, '_nP', None) is None:
            self._nP = '*'
        return self._nP

    def _eval(self, x, **kwargs):
        raise NotImplementedError(
            "The method _eval has not been implemented for {}".format(
                self.__class__.__name__
            )
        )

    def deriv(self, x, **kwargs):
        raise NotImplementedError(
            "The method deriv has not been implemented for {}".format(
                self.__class__.__name__
            )
        )

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
            lambda m: [self(m), self.deriv(m)], x, num=num, plotIt=plotIt
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
            plotIt=plotIt
        )

    def test(self, x=None, num=4, plotIt=False, **kwargs):
        deriv = self._test_deriv(x=x, num=num, **kwargs)
        deriv2 = self._test_deriv2(x=x, num=num, plotIt=False, **kwargs)
        return (deriv & deriv2)

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

        if isinstance(self, ComboObjectiveFunction):
            if isinstance(objfct2, ComboObjectiveFunction):
                objfctlist = self.objfcts + objfct2.objfcts
                multipliers = self.multipliers + objfct2.multipliers
            elif isinstance(objfct2, ObjectiveFunction):
                objfctlist = self.objfcts.append(objfct2)
                multipliers = self.multipliers.append(1)
        else:
            if isinstance(objfct2, ComboObjectiveFunction):
                objfctlist = [self] + objfct2.objfcts
                multipliers = [1] + objfct2.multipliers
            else:
                objfctlist = [self, objfct2]
                multipliers = None

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


class ComboObjectiveFunction(BaseObjectiveFunction):

    _multiplier_types = (float, Float, None, Utils.Zero) + integer_types # Directive

    def __init__(self, objfcts=[], multipliers=None, **kwargs):

        self.objfcts = []
        nP = '*'

        for fct in objfcts:
            assert isinstance(fct, BaseObjectiveFunction), (
                "Unrecognized objective function type {} in objfcts. All "
                "entries in objfcts must inherit from  ObjectiveFunction"
            )

            # ensure all objective functions have the same nP
            if fct.nP != '*':
                if nP != '*':
                    assert nP == fct.nP, (
                        "Objective Functions must all have the same nP, not "
                        "{}".format([f.nP for f in objfcts])
                    )
                else:
                    nP = fct.nP
            self.objfcts.append(fct)

        if multipliers is None:
            multipliers = len(self.objfcts)*[1]
        else:
            for mult in multipliers:
                assert(type(mult) in self._multiplier_types), (
                    "Objective Functions can only be multiplied by a float, or"
                    " a properties.Float, not a {}, {}".format(type(mult), mult)
                )
            assert len(multipliers) == len(self.objfcts), (
                "Length of multipliers ({}) must be the same as the length of "
                "objfcts ({})".format(len(multipliers), len(self.objfcts))
            )
        self.multipliers = multipliers

        super(ComboObjectiveFunction, self).__init__(**kwargs)
        self._nP = nP

    def _eval(self, x, **kwargs):
        f = Utils.Zero()
        for multpliter, objfct in zip(self.multipliers, self.objfcts):
            f += multpliter * objfct(x, **kwargs)
        return f

    def deriv(self, x, **kwargs):
        g = Utils.Zero()
        for multpliter, objfct in zip(self.multipliers, self.objfcts):
            g += multpliter * objfct.deriv(x, **kwargs)
        return g

    def deriv2(self, x, **kwargs):
        H = Utils.Zero()
        for multpliter, objfct in zip(self.multipliers, self.objfcts):
            objfct_H = objfct.deriv2(x, **kwargs)
            if isinstance(objfct_H, Utils.Zero):
                pass
            elif isinstance(objfct_H, Utils.Identity) and self.nP != '*':
                H += multpliter * sp.eye(self.nP)  # if we need a shape
            else:
                H += multpliter * objfct_H
        return H

    @property
    def W(self):
        W = []
        for mult, fct in zip(self.multipliers, self.objfcts):
            curW = mult * fct.W
            if not isinstance(curW, Utils.Zero):
                W.append(curW)
        return sp.vstack(W)


class L2ObjectiveFunction(BaseObjectiveFunction):

    def __init__(self, nP=None, **kwargs):

        super(L2ObjectiveFunction, self).__init__(nP=nP, **kwargs)

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
