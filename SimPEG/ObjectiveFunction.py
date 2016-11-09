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
        print('Testing {0!s} Deriv'.format(self.__class__))
        if x is None:
            if getattr(self, 'nP', None) is not None:
                x = np.random.randn(self.nP)
            else:
                x = np.random.randn(np.random.randint(1e2, high=1e3))

        return checkDerivative(
            lambda m: [self(m), self.deriv(m)], x, num=num, plotIt=plotIt
        )

    def _test_deriv2(self, x=None, num=4, plotIt=False, **kwargs):
        print('Testing {0!s} Deriv2'.format(self.__class__))
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
        if issubclass(ObjectiveFunction, type(objfct2)):

            if isinstance(self, ComboObjectiveFunction):

                return ComboObjectiveFunction(self+[objfct2], )
            return ObjectiveFunction(
                _eval=fct, _deriv=fct_deriv, _deriv2=fct_deriv2
            )

        raise NotImplementedError

    def __radd__(self, objfct2):
        return self+objfct2

    def __mul__(self, scalar):
        if not (isinstance(scalar, Float) or isinstance(scalar, float)):
            raise Exception(
                "Objective Functions can only be multiplied by a float, or a"
                "properties.Float, not a {}".format(type(scalar))
            )
        return ObjectiveFunction(
            _eval=lambda x, **kwargs: scalar*self(x, **kwargs),
            _deriv=lambda x, **kwargs: scalar*self.deriv(x, **kwargs),
            _deriv2=lambda x, **kwargs: scalar*self.deriv2(x, **kwargs)
        )

    def __rmul__(self, scalar):
        return self*scalar


class ComboObjectiveFunction(ObjectiveFunction):

    _multiplier_types = [float, Float, None]

    def __init__(self, objfcts, multipliers=None, **kwargs):

        self.objfcts = []
        for fct in objfcts:
            assert isinstance(fct, ObjectiveFunction), (
                "Unrecognized objective function type {} in objfcts. All "
                "entries in objfcts must inherit from  ObjectiveFunction"
            )
            self.objfcts += fct

        if multipliers is None:
            self.multipliers = len(self.objfcts)*[1]
        else:
            for mult in multipliers:
                assert(type(mult) in self._multiplier_types), (
                    "Objective Functions can only be multiplied by a float, or"
                    " a properties.Float, not a {}".format(type(scalar))
                )
            assert len(multipliers) == len(self.objfcts), (
                "Length of multipliers ({}) must be the same as the length of "
                "objfcts ({})".format(len(multipliers), len(self.objfcts))
            )

        super(ComboObjectiveFunction, self).__init__(self, **kwargs)

    def _eval(self, **kwargs):
        NotImplementedError

    def _deriv(self, **kwargs):
        NotImplementedError

    def _deriv2(self, **kwargs):
        NotImplementedError

