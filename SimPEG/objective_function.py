import numpy as np
import scipy.sparse as sp

from discretize.tests import check_derivative

from .maps import IdentityMap
from .props import BaseSimPEG
from .utils import timeIt, Zero, Identity

__all__ = ["BaseObjectiveFunction", "ComboObjectiveFunction", "L2ObjectiveFunction"]


class BaseObjectiveFunction(BaseSimPEG):
    """
    Base Objective Function

    Inherit this to build your own objective function. If building a
    regularization, have a look at
    :class:`SimPEG.regularization.BaseRegularization` as there are additional
    methods and properties tailored to regularization of a model. Similarly,
    for building a data misfit, see :class:`SimPEG.DataMisfit.BaseDataMisfit`.
    """

    map_class = IdentityMap  #: Base class of expected maps

    def __init__(
        self,
        nP=None,
        mapping=None,
        has_fields=False,
        counter=None,
        debug=False,
    ):
        self._nP = nP
        if mapping is None:
            self._mapping = mapping
        else:
            self.mapping = mapping
        self.counter = counter
        self.debug = debug
        self.has_fields = has_fields

    def __call__(self, x, f=None):
        """
        Evaluate the objective functions for a given model
        """
        raise NotImplementedError(
            "__call__ has not been implemented for {} yet".format(
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
        if getattr(self, "mapping", None) is not None:
            return self.mapping.nP
        return "*"

    @property
    def _nC_residual(self):
        """
        Shape of the residual
        """
        if getattr(self, "mapping", None) is not None:
            return self.mapping.shape[0]
        else:
            return self.nP

    @property
    def mapping(self):
        """
        A `SimPEG.Maps` instance
        """
        if self._mapping is None:
            if self._nP is not None:
                self._mapping = self.map_class(nP=self.nP)
            else:
                self._mapping = self.map_class()
        return self._mapping

    @mapping.setter
    def mapping(self, value):
        if not isinstance(value, self.map_class):
            raise TypeError(
                f"Invalid mapping of class '{value.__class__.__name__}'. "
                f"It must be an instance of {self.map_class.__name__}"
            )
        self._mapping = value

    @timeIt
    def deriv(self, x, **kwargs):
        """
        First derivative of the objective function with respect to the model
        """
        raise NotImplementedError(
            "The method deriv has not been implemented for {}".format(
                self.__class__.__name__
            )
        )

    @timeIt
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
        print("Testing {0!s} Deriv".format(self.__class__.__name__))
        if x is None:
            if self.nP == "*":
                x = np.random.randn(np.random.randint(1e2, high=1e3))
            else:
                x = np.random.randn(self.nP)

        return check_derivative(
            lambda m: [self(m), self.deriv(m)], x, num=num, plotIt=plotIt, **kwargs
        )

    def _test_deriv2(self, x=None, num=4, plotIt=False, **kwargs):
        print("Testing {0!s} Deriv2".format(self.__class__.__name__))
        if x is None:
            if self.nP == "*":
                x = np.random.randn(np.random.randint(1e2, high=1e3))
            else:
                x = np.random.randn(self.nP)

        v = x + 0.1 * np.random.rand(len(x))
        expectedOrder = kwargs.pop("expectedOrder", 1)
        return check_derivative(
            lambda m: [self.deriv(m).dot(v), self.deriv2(m, v=v)],
            x,
            num=num,
            expectedOrder=expectedOrder,
            plotIt=plotIt,
            **kwargs,
        )

    def test(self, x=None, num=4, **kwargs):
        """
        Run a convergence test on both the first and second derivatives - they
        should be second order!
        """
        deriv = self._test_deriv(x=x, num=num, **kwargs)
        deriv2 = self._test_deriv2(x=x, num=num, plotIt=False, **kwargs)
        return deriv & deriv2

    __numpy_ufunc__ = True

    def __add__(self, other):
        if isinstance(other, Zero):
            return self
        if not isinstance(other, BaseObjectiveFunction):
            raise TypeError(
                f"Cannot add type '{other.__class__.__name__}' to an objective "
                "function. Only ObjectiveFunctions can be added together."
            )
        objective_functions, multipliers = [], []
        for instance in (self, other):
            if isinstance(instance, ComboObjectiveFunction) and instance._unpack_on_add:
                objective_functions += instance.objfcts
                multipliers += instance.multipliers
            else:
                objective_functions.append(instance)
                multipliers.append(1)
        combo = ComboObjectiveFunction(
            objfcts=objective_functions, multipliers=multipliers
        )
        return combo

    def __radd__(self, other):
        return self + other

    def __mul__(self, multiplier):
        return ComboObjectiveFunction(objfcts=[self], multipliers=[multiplier])

    def __rmul__(self, multiplier):
        return self * multiplier

    def __div__(self, denominator):
        return self * (1.0 / denominator)

    def __truediv__(self, denominator):
        return self * (1.0 / denominator)

    def __rdiv__(self, denominator):
        return self * (1.0 / denominator)


class ComboObjectiveFunction(BaseObjectiveFunction):
    """
    Composite for multiple objective functions

    A composite class for multiple objective functions. Each objective function
    is accompanied by a multiplier. Both objective functions and multipliers
    are stored in a list.

    Parameters
    ----------
    objfcts : list or None, optional
        List containing the objective functions that will live inside the
        composite class. If ``None``, an empty list will be created.
    multipliers : list or None, optional
        List containing the multipliers for its respective objective function
        in ``objfcts``.  If ``None``, a list full of ones with the same length
        as ``objfcts`` will be created.
    unpack_on_add : bool, optional
        Weather to unpack the multiple objective functions when adding them to
        another objective function, or to add them as a whole.

    Examples
    --------
    Build a simple combo objective function:

    >>> objective_fun_a = L2ObjectiveFunction(nP=3)
    >>> objective_fun_b = L2ObjectiveFunction(nP=3)
    >>> combo = ComboObjectiveFunction([objective_fun_a, objective_fun_b], [1, 0.5])
    >>> print(len(combo))
    2
    >>> print(combo.multipliers)
    [1, 0.5]

    Combo objective functions are also created after adding two objective functions:

    >>> combo = 2 * objective_fun_a + 3.5 * objective_fun_b
    >>> print(len(combo))
    2
    >>> print(combo.multipliers)
    [2, 3.5]

    We could add two combo objective functions as well:

    >>> objective_fun_c = L2ObjectiveFunction(nP=3)
    >>> objective_fun_d = L2ObjectiveFunction(nP=3)
    >>> combo_1 = 4.3 * objective_fun_a + 3 * objective_fun_b
    >>> combo_2 = 1.5 * objective_fun_c + 0.5 * objective_fun_d
    >>> combo = combo_1 + combo_2
    >>> print(len(combo))
    4
    >>> print(combo.multipliers)
    [4.3, 3, 1.5, 0.5]

    We can choose to not unpack the objective functions when creating the
    combo. For example:

    >>> objective_fun_a = L2ObjectiveFunction(nP=3)
    >>> objective_fun_b = L2ObjectiveFunction(nP=3)
    >>> objective_fun_c = L2ObjectiveFunction(nP=3)
    >>>
    >>> # Create a ComboObjectiveFunction that won't unpack
    >>> combo_1 = ComboObjectiveFunction(
    ...     objfcts=[objective_fun_a, objective_fun_b],
    ...     multipliers=[0.1, 1.2],
    ...     unpack_on_add=False,
    ... )
    >>> combo_2 = combo_1 + objective_fun_c
    >>> print(len(combo_2))
    2
    """

    _multiplier_types = (float, None, Zero, np.float64, int, np.integer)

    def __init__(self, objfcts=None, multipliers=None, unpack_on_add=True):
        # Define default lists if None
        if objfcts is None:
            objfcts = []
        if multipliers is None:
            multipliers = len(objfcts) * [1]

        # Validate inputs
        self._check_length_objective_funcs_multipliers(objfcts, multipliers)
        self._validate_objective_functions(objfcts)
        self._validate_multipliers(multipliers)

        # Get number of parameters (nP) from objective functions
        number_of_parameters = [f.nP for f in objfcts if f.nP != "*"]
        if number_of_parameters:
            nP = number_of_parameters[0]
        else:
            nP = None

        super().__init__(nP=nP)
        self.objfcts = objfcts
        self._multipliers = multipliers
        self._unpack_on_add = unpack_on_add

    def __len__(self):
        return len(self.multipliers)

    def __getitem__(self, key):
        return self.multipliers[key], self.objfcts[key]

    @property
    def multipliers(self):
        """
        Multipliers for each objective function
        """
        return self._multipliers

    @multipliers.setter
    def multipliers(self, value):
        """
        Set multipliers attribute after checking if they are valid
        """
        self._validate_multipliers(value)
        self._check_length_objective_funcs_multipliers(self.objfcts, value)
        self._multipliers = value

    def __call__(self, m, f=None):
        """
        Evaluate the objective functions for a given model
        """
        fct = 0.0
        for i, phi in enumerate(self):
            multiplier, objfct = phi
            if multiplier == 0.0:  # don't evaluate the fct
                continue
            if f is not None and objfct.has_fields:
                objective_func_value = objfct(m, f=f[i])
            else:
                objective_func_value = objfct(m)
            fct += multiplier * objective_func_value
        return fct

    def deriv(self, m, f=None):
        """
        First derivative of the composite objective function is the sum of the
        derivatives of each objective function in the list, weighted by their
        respective multplier.

        :param numpy.ndarray m: model
        :param SimPEG.Fields f: Fields object (if applicable)
        """
        g = Zero()
        for i, phi in enumerate(self):
            multiplier, objfct = phi
            if multiplier == 0.0:  # don't evaluate the fct
                continue
            if f is not None and objfct.has_fields:
                aux = objfct.deriv(m, f=f[i])
            else:
                aux = objfct.deriv(m)
            if not isinstance(aux, Zero):
                g += multiplier * aux
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
        H = Zero()
        for i, phi in enumerate(self):
            multiplier, objfct = phi
            if multiplier == 0.0:  # don't evaluate the fct
                continue
            if f is not None and objfct.has_fields:
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
            if not isinstance(curW, Zero):
                W.append(curW)
        return sp.vstack(W)

    def get_functions_of_type(self, fun_class) -> list:
        """
        Find an objective function type from a ComboObjectiveFunction class.
        """
        target = []
        if isinstance(self, fun_class):
            target += [self]
        else:
            for fct in self.objfcts:
                if isinstance(fct, ComboObjectiveFunction):
                    target += [fct.get_functions_of_type(fun_class)]
                elif isinstance(fct, fun_class):
                    target += [fct]

        return [fun for fun in target if fun]

    def _validate_objective_functions(self, objective_functions):
        """
        Validate objective functions

        Check if the objective functions have the right types, and if
        they all have the same number of parameters.
        """
        for function in objective_functions:
            if not isinstance(function, BaseObjectiveFunction):
                raise TypeError(
                    "Unrecognized objective function type "
                    f"{function.__class__.__name__} in 'objfcts'. "
                    "All objective functions must inherit from BaseObjectiveFunction."
                )
        number_of_parameters = [f.nP for f in objective_functions if f.nP != "*"]
        if number_of_parameters:
            all_equal = all(np.equal(number_of_parameters, number_of_parameters[0]))
            if not all_equal:
                np_list = [f.nP for f in objective_functions]
                raise ValueError(
                    f"Invalid number of parameters '{np_list}' found in "
                    "objective functions. Except for the ones with '*', they all "
                    "must have the same number of parameters."
                )

    def _validate_multipliers(self, multipliers):
        """
        Validate multipliers

        Check if the multipliers have the right types.
        """
        for multiplier in multipliers:
            if type(multiplier) not in self._multiplier_types:
                valid_types = ", ".join(str(t) for t in self._multiplier_types)
                raise TypeError(
                    f"Invalid multiplier '{multiplier}' of type '{type(multiplier)}'. "
                    "Objective functions can only be multiplied by " + valid_types
                )

    def _check_length_objective_funcs_multipliers(
        self, objective_functions, multipliers
    ):
        """
        Check if objective functions and multipliers have the same length
        """
        if len(objective_functions) != len(multipliers):
            raise ValueError(
                "Inconsistent number of elements between objective functions "
                f"('{len(objective_functions)}') and multipliers "
                f"('{len(multipliers)}'). They must have the same number of parameters."
            )


class L2ObjectiveFunction(BaseObjectiveFunction):
    r"""
    An L2-Objective Function

    .. math::

        \phi = \frac{1}{2}||\mathbf{W} \mathbf{m}||^2
    """

    def __init__(
        self,
        nP=None,
        mapping=None,
        W=None,
        has_fields=False,
        counter=None,
        debug=False,
    ):
        # Check if nP and shape of W are consistent
        if W is not None and nP is not None and nP != W.shape[1]:
            raise ValueError(
                f"Number of parameters nP ('{nP}') doesn't match the number of "
                f"rows ('{W.shape[1]}') of the weights matrix W."
            )
        super().__init__(
            nP=nP,
            mapping=mapping,
            has_fields=has_fields,
            debug=debug,
            counter=counter,
        )
        if W is not None and self.nP == "*":
            self._nP = W.shape[1]
        self._W = W

    @property
    def W(self):
        """
        Weighting matrix. The default if not specified is an identity.
        """
        if getattr(self, "_W", None) is None:
            if self._nC_residual != "*":
                self._W = sp.eye(self._nC_residual)
            else:
                self._W = Identity()
        return self._W

    def __call__(self, m):
        """
        Evaluate the objective functions for a given model
        """
        r = self.W * (self.mapping * m)
        return 0.5 * r.dot(r)

    def deriv(self, m):
        """
        First derivative with respect to the model

        :param numpy.ndarray m: model
        """
        return self.mapping.deriv(m).T * (self.W.T * (self.W * (self.mapping * m)))

    def deriv2(self, m, v=None):
        """
        Second derivative with respect to the model

        :param numpy.ndarray m: model
        :param numpy.ndarray v: vector to multiply by
        """
        if v is not None:
            return self.mapping.deriv(m).T * (
                self.W.T * (self.W * (self.mapping.deriv(m) * v))
            )
        W = self.W * self.mapping.deriv(m)
        return W.T * W
