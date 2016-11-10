from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

from properties import Float
import numpy as np

from . import Utils
from .Tests import checkDerivative


class ObjectiveFunction(object):

    _eval = None
    _deriv = None
    _deriv2 = None

    def __init__(self, **kwargs):

        Utils.setKwargs(self, **kwargs)

    def eval(self, x, **kwargs):
        if getattr(self, '_eval', None) is None:
            raise NotImplementedError
        return self._eval(x, **kwargs)

    __call__ = eval

    def deriv(self, x, **kwargs):
        if getattr(self, '_deriv', None) is None:
            raise NotImplementedError
        return self._deriv(x, **kwargs)

    def deriv2(self, x, **kwargs):
        if getattr(self, '_deriv2', None) is None:
            raise NotImplementedError
        return self._deriv2(x, **kwargs)

    def _test_deriv(self, x=None, num=4, plotIt=False, **kwargs):
        print('Testing {0!s} Deriv'.format(self.__class__.__name__))
        if x is None:
            if getattr(self, 'nP', None) is not None:
                x = np.random.randn(self.nP)
            else:
                x = np.random.randn(np.random.randint(1e2, high=1e3))

        return checkDerivative(
            lambda m: [self(m), self.deriv(m)], x, num=num, plotIt=plotIt
        )

    def _test_deriv2(self, x=None, num=4, plotIt=False, **kwargs):
        print('Testing {0!s} Deriv2'.format(self.__class__.__name__))
        if x is None:
            if getattr(self, 'nP', None) is not None:
                x = np.random.randn(self.nP)
            else:
                x = np.random.randn(np.random.randint(1e2, high=1e3))

        return checkDerivative(
            lambda m: [self.deriv(m), self.deriv2(m)], x, num=num,
            plotIt=plotIt
        )

    def test(self, x=None, num=4, plotIt=False, **kwargs):
        deriv = self._test_deriv(x=x, num=num, **kwargs)
        deriv2 = self._test_deriv2(x=x, num=num, plotIt=False, **kwargs)
        return (deriv & deriv2)

    def __add__(self, objfct2):
        if not isinstance(objfct2, ObjectiveFunction):
            raise Exception(
                "Cannot add type {} to an objective function. Only "
                "ObjectiveFunctions can be added together".format(
                    objfct2.__class__.__name__
                )
            )

        if isinstance(self, ComboObjectiveFunction):
            if isinstance(objfct2, ComboObjectiveFunction):
                objfctlist = self.objfcts + objfct2
                multipliers = self._multipliers + objfct2._multipliers
            elif isinstance(objfct2, ObjectiveFunction):
                objfctlist = self.objfcts.append(objfct2)
                multipliers = self._multipliers.append(1)
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
        return self*multiplier


class ComboObjectiveFunction(ObjectiveFunction):

    _multiplier_types = [float, Float, None, int, long] # Directive

    def __init__(self, objfcts, multipliers=None, **kwargs):

        self.objfcts = []
        for fct in objfcts:
            assert isinstance(fct, ObjectiveFunction), (
                "Unrecognized objective function type {} in objfcts. All "
                "entries in objfcts must inherit from  ObjectiveFunction"
            )
            self.objfcts.append(fct)

        if multipliers is None:
            multipliers = len(self.objfcts)*[1]
        else:
            for mult in multipliers:
                assert(type(mult) in self._multiplier_types), (
                    "Objective Functions can only be multiplied by a float, or"
                    " a properties.Float, not a {}".format(type(mult))
                )
            assert len(multipliers) == len(self.objfcts), (
                "Length of multipliers ({}) must be the same as the length of "
                "objfcts ({})".format(len(multipliers), len(self.objfcts))
            )
        self.multipliers = multipliers

        super(ComboObjectiveFunction, self).__init__(**kwargs)

    def _eval(self, x, **kwargs):
        f = Utils.Zero()
        for multpliter, objfct in zip(self.multipliers, self.objfcts):
            f += multpliter * objfct(x, **kwargs)
        return f

    def _deriv(self, x, **kwargs):
        g = Utils.Zero()
        for multpliter, objfct in zip(self.multipliers, self.objfcts):
            g += multpliter * objfct.deriv(x, **kwargs)
        return g

    def _deriv2(self, x, **kwargs):
        H = Utils.Zero()
        for multpliter, objfct in zip(self.multipliers, self.objfcts):
            H += multpliter * objfct.deriv2(x, **kwargs)
        return H

