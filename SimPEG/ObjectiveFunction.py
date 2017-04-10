from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division

import numpy as np
import scipy.sparse as sp
from six import integer_types, string_types
import warnings

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

        v = x + 0.01*np.random.rand(len(x))
        return checkDerivative(
            lambda m: [self.deriv(m).dot(v), self.deriv2(m, v=v)],
                x, num=num, expectedOrder=1,
                plotIt=plotIt, **kwargs
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


# class DynamicDescriptorMixin(object):

#     def __getattribute__(self, name):
#         value = object.__getattribute__(self, name)
#         if hasattr(value, '__get__'):
#             value = value.__get__(self, self.__class__)
#         return value

#     def __setattr__(self, name, value):
#         try:
#             obj = object.__getattribute__(self, name)
#         except AttributeError:
#             pass
#         else:
#             if hasattr(obj, '__set__'):
#                 return obj.__set__(self, value)
#         return object.__setattr__(self, name, value)


class ExposedProperty(object):

    def __init__(self, objfcts, prop, val=None, **kwargs):
        # only add functions with that property
        fctlist = [
            fct for fct in objfcts if getattr(fct, prop, None) is not None
        ]
        # print(
        #     'exposing {prop} for {fcts}'.format(
        #         prop=prop, fcts=[fct.__class__.__name__ for fct in objfcts]
        #     )
        # )

        self.prop = prop
        self.objfcts = fctlist

        if val is not None:
            self.val = self.__set__(None, val=val)  # go through setter
        else:
            self.val = val  # skip setter

    def __get__(self, obj, objtype=None):
        return self.val

    def __set__(self, obj, val):
        [setattr(fct, self.prop, val) for fct in self.objfcts] # propagate change
        self.val = val


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
    _multiplier_types = (float, None, Utils.Zero) + integer_types  # Directive
    _exposed = None  # Properties of lower objective functions that are exposed

    def __init__(self, objfcts=[], multipliers=None, **kwargs):

        self._nP = '*'
        self._exposed = {}

        if multipliers is None:
            multipliers = len(objfcts)*[1]

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
        """
        First derivative of the composite objective function is the sum of the
        derivatives of each objective function in the list, weighted by their
        respective multplier.

        :param numpy.ndarray m: model
        :param SimPEG.Fields f: Fields object (if applicable)
        """
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
        """
        Second derivative of the composite objective function is the sum of the
        second derivatives of each objective function in the list, weighted by
        their respective multplier.

        :param numpy.ndarray m: model
        :param numpy.ndarray v: vector we are multiplying by
        :param SimPEG.Fields f: Fields object (if applicable)
        """
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

    def expose(self, properties):
        # if 'all', exposes all top level properties in the objective function
        # list
        if isinstance(properties, string_types) and properties.lower() == 'all':
            prop_set = []
            for objfct in self.objfcts:
                prop_set += [
                    prop for prop in dir(objfct)
                    if prop[0] != '_' and  # only expose top level properties
                    isinstance(
                        getattr(type(objfct), prop, None), property
                    ) and
                    (
                        # don't try and over-write things like nP
                        # which are properties on this class
                        getattr(type(self), prop, None) is None or
                        prop in self._exposed.keys()
                    )
                ]
            properties = list(set(prop_set))

        # if a single string provided, turn it into a list
        if isinstance(properties, string_types):
            properties = [properties]

        # work with dicts if a list provided
        if isinstance(properties, list):
            properties = dict(zip(properties, len(properties)*[None]))

        # go through the properties list and expose them
        for prop, val in properties.items(): # skip if already in self._exposed
            if getattr(type(self), prop, None) is not None:
                raise Exception(
                    "can't expose {} as it is a property on the combo "
                    "objective function".format(prop)
                )
            else:
                new_prop = ExposedProperty(self.objfcts, prop, val=val)
                self._exposed[prop] = new_prop
                setattr(self, prop, new_prop)

    def __setattr__(self, name, value):
        try:
            obj = object.__getattribute__(self, name)
        except AttributeError:
            pass
        else:
            exposed = object.__getattribute__(self, '_exposed')
            if exposed is not None and name in exposed.keys():
                return exposed[name].__set__(self, value)
        return object.__setattr__(self, name, value)

    def __getattribute__(self, name):
        exposed = object.__getattribute__(self, '_exposed')
        if exposed is not None and name in exposed.keys():
            return exposed[name].__get__(self, self.__class__)
        return object.__getattribute__(self, name)


class L2ObjectiveFunction(BaseObjectiveFunction):
    """
    An L2-Objective Function

    .. math::

        \phi = \frac{1}{2}||\mathbf{W} \mathbf{m}||^2
    """

    mapPair = Maps.IdentityMap  #: Base class of expected maps

    def __init__(self, W=None, **kwargs):

        super(L2ObjectiveFunction, self).__init__(**kwargs)
        if W is not None:
            if self.nP == '*':
                self._nP = W.shape[1]
        self._W = W

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
        return self.W.T * self.W
