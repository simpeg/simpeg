from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division

import numpy as np
import scipy.sparse as sp
from six import integer_types
import warnings
import dask
import dask.array as da
from . import Utils
from .Tests import checkDerivative
from . import Maps
from . import Props


__all__ = [
    'BaseObjectiveFunction', 'ComboObjectiveFunction', 'L2ObjectiveFunction'
]


class BaseObjectiveFunction(Props.BaseSimPEG):
    """
    Base Objective Function

    Inherit this to build your own objective function. If building a
    regularization, have a look at
    :class:`SimPEG.Regularization.BaseRegularization` as there are additional
    methods and properties tailored to regularization of a model. Similarly,
    for building a data misfit, see :class:`SimPEG.DataMisfit.BaseDataMisfit`.
    """

    counter = None
    debug = False

    mapPair = Maps.IdentityMap  #: Base class of expected maps
    _mapping = None  #: An IdentityMap instance.
    _hasFields = False  #: should we have the option to store fields

    _nP = None  #: number of parameters

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
        """
        Number of model parameters expected.
        """
        if self._nP is not None:
            return self._nP
        if getattr(self, 'mapping', None) is not None:
            return self.mapping.nP
        return '*'

    @property
    def _nC_residual(self):
        """
        Shape of the residual
        """
        if getattr(self, 'mapping', None) is not None:
            return self.mapping.shape[0]
        else:
            return self.nP

    @property
    def mapping(self):
        """
        A `SimPEG.Maps` instance
        """
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
        """
        First derivative of the objective function with respect to the model
        """
        raise NotImplementedError(
            "The method deriv has not been implemented for {}".format(
                self.__class__.__name__
            )
        )

    @Utils.timeIt
    def deriv2(self, x, v=None, **kwargs):
        """
        Second derivative of the objective function with respect to the model
        """
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

        v = x + 0.1*np.random.rand(len(x))
        return checkDerivative(
            lambda m: [self.deriv(m).dot(v), self.deriv2(m, v=v)],
            x, num=num, expectedOrder=1, plotIt=plotIt, **kwargs
        )

    def test(self, x=None, num=4, plotIt=False, **kwargs):
        """
        Run a convergence test on both the first and second derivatives - they
        should be second order!
        """
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
    """
    A composite objective function that consists of multiple objective
    functions. Objective functions are stored in a list, and multipliers
    are stored in a parallel list.

    .. code::python

        import SimPEG.ObjectiveFunction
        phi1 = ObjectiveFunction.L2ObjectiveFunction(nP=10)
        phi2 = ObjectiveFunction.L2ObjectiveFunction(nP=10)

        phi = 2*phi1 + 3*phi2

    is equivalent to

        .. code::python

            import SimPEG.ObjectiveFunction
            phi1 = ObjectiveFunction.L2ObjectiveFunction(nP=10)
            phi2 = ObjectiveFunction.L2ObjectiveFunction(nP=10)

            phi = ObjectiveFunction.ComboObjectiveFunction(
                [phi1, phi2], [2, 3]
            )

    """
    _multiplier_types = (float, None, Utils.Zero, np.float64) + integer_types # Directive
    _multipliers = None

    def __init__(self, objfcts=[], multipliers=None, **kwargs):

        if multipliers is None:
            multipliers = len(objfcts)*[1]

        self._nP = '*'

        assert(len(objfcts)==len(multipliers)), (
            "Must have the same number of Objective Functions and Multipliers "
            "not {} and {}".format(len(objfcts), len(multipliers))
        )

        def validate_list(objfctlist, multipliers):
            """
            ensure that the number of parameters expected by each objective
            function is the same, ensure that if multpliers are supplied, that
            list matches the length of the objective function list
            """
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
        self._multipliers = multipliers

        super(ComboObjectiveFunction, self).__init__(**kwargs)

    def __len__(self):
        return len(self.multipliers)

    def __getitem__(self, key):
        return self.multipliers[key], self.objfcts[key]

    @property
    def __len__(self):
        return self.objfcts.__len__

    @property
    def multipliers(self):
        return self._multipliers

    @multipliers.setter
    def multipliers(self, value):
        for val in value:
            assert type(val) in self._multiplier_types, (
                'Multiplier must be in type {} not {}'.format(
                    self._multiplier_types, type(val)
                )
            )

        assert len(value) == len(self.objfcts), (
            'the length of multipliers should be the same as the number of'
            ' objective functions ({}), not {}'.format(
                len(self.objfcts, len(value))
            )
        )

        self._multipliers = value

    def __call__(self, m, f=None):

        fct = []
        for i, phi in enumerate(self):
            multiplier, objfct = phi
            if multiplier == 0.: # don't evaluate the fct
                continue
            else:
                if f is not None and objfct._hasFields:
                    fct += [multiplier * objfct(m, f=f[i])]
                else:
                    fct += [multiplier * objfct(m)]

        stack = da.vstack(fct)

        return da.sum(stack, axis=0).compute()


    def deriv(self, m, f=None):
        """
        First derivative of the composite objective function is the sum of the
        derivatives of each objective function in the list, weighted by their
        respective multplier.

        :param numpy.ndarray m: model
        :param SimPEG.Fields f: Fields object (if applicable)
        """

        # @dask.delayed
        # def rowSum(arr):
        #     sumIt = 0
        #     for i in range(len(arr)):
        #         sumIt += arr[i]
        #     return sumIt

        g = []
        for i, phi in enumerate(self):
            multiplier, objfct = phi
            if multiplier == 0.: # don't evaluate the fct
                continue
            else:
                if f is not None and objfct._hasFields:
                    g += [multiplier * objfct.deriv(m, f=f[i])]
                else:
                    g += [multiplier * objfct.deriv(m)]

        stack = da.vstack(g)

        return da.sum(stack, axis=0).compute()

    def deriv2(self, m, v=None, f=None):
        """
        Second derivative of the composite objective function is the sum of the
        second derivatives of each objective function in the list, weighted by
        their respective multplier.

        :param numpy.ndarray m: model
        :param numpy.ndarray v: vector we are multiplying by
        :param SimPEG.Fields f: Fields object (if applicable)
        """
        # @dask.delayed
        # def rowSum(arr):
        #     sumIt = 0
        #     for i in range(len(arr)):
        #         sumIt += arr[i]
        #     return sumIt

        H = []
        for i, phi in enumerate(self):
            multiplier, objfct = phi
            if multiplier == 0.: # don't evaluate the fct
                continue
            else:
                if f is not None and objfct._hasFields:

                    H += [multiplier * objfct.deriv2(m, v, f=f[i])]
                else:
                    H += [multiplier * objfct.deriv2(m, v)]

        if isinstance(H[0], dask.array.Array):

            stack = da.vstack(H)

            return da.sum(stack, axis=0).compute()

        else:
            sumIt = 0
            for i in range(len(H)):
                sumIt += H[i]
            return sumIt

    # This assumes all objective functions have a W.
    # The base class currently does not.
    @property
    def W(self):
        """
        W matrix for the full objective function. Includes multiplying by the
        square root of alpha.
        """
        W = []
        for mult, fct in self:
            curW = np.sqrt(mult) * fct.W
            if not isinstance(curW, Utils.Zero):
                W.append(curW)
        return sp.vstack(W)


class L2ObjectiveFunction(BaseObjectiveFunction):
    """
    An L2-Objective Function

    .. math::

        \phi = \frac{1}{2}||\mathbf{W} \mathbf{m}||^2
    """
    def __init__(self, W=None, **kwargs):

        super(L2ObjectiveFunction, self).__init__(**kwargs)
        if W is not None:
            if self.nP == '*':
                self._nP = W.shape[1]
        self._W = W

    @property
    def W(self):
        """
        Weighting matrix. The default if not sepcified is an identity.
        """
        if getattr(self, '_W', None) is None:
            if self._nC_residual != '*':
                self._W = sp.eye(self._nC_residual)
            else:
                self._W = Utils.Identity()
        return self._W

    def __call__(self, m):
        r = self.W * (self.mapping * m)
        return 0.5 * r.dot(r)

    def deriv(self, m):
        """
        First derivative with respect to the model

        :param numpy.ndarray m: model
        """
        return (
            self.mapping.deriv(m).T *
            (self.W.T * (self.W * (self.mapping * m)))
        )

    def deriv2(self, m, v=None):
        """
        Second derivative with respect to the model

        :param numpy.ndarray m: model
        :param numpy.ndarray v: vector to multiply by
        """
        if v is not None:
            return (
                self.mapping.deriv(m).T * (
                    self.W.T * (self.W * (self.mapping.deriv(m) * v))
                )
            )
        W = self.W * self.mapping.deriv(m)
        return W.T * W
