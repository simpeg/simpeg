from __future__ import annotations
from abc import ABC, abstractmethod

import numbers
import numpy as np

from discretize.tests import check_derivative

from .utils import Zero
from .typing import RandomSeed

__all__ = ["BaseObjectiveFunction", "ComboObjectiveFunction"]

VALID_MULTIPLIERS = (numbers.Number, Zero)


def _need_to_pass_fields(objective_function) -> bool:
    """
    Determine if we need to pass fields to the given objective function
    """
    has_fields = getattr(objective_function, "has_fields", False)
    is_combo = isinstance(objective_function, ComboObjectiveFunction)
    return has_fields or is_combo


class BaseObjectiveFunction(ABC):
    """
    Base class for creating objective functions.

    .. important::
        If building a regularization function within SimPEG, please inherit
        :py:class:`simpeg.regularization.BaseRegularization`, as this class has
        additional functionality related to regularization.
        And if building a data misfit function, please inherit
        :py:class:`simpeg.data_misfit.BaseDataMisfit`.
    """

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, model, f=None) -> float:
        """
        Evaluate the objective function for a given model.

        Parameters
        ----------
        x : (nP) numpy.ndarray
            A vector representing a set of model parameters.
        f : simpeg.fields.Fields, optional
            Field object (if applicable).
        """
        pass

    @abstractmethod
    def deriv(self, model):
        r"""Gradient of the objective function evaluated for the model provided.

        Where :math:`\phi (\mathbf{m})` is the objective function, this method
        evaluates and returns the derivative with respect to the model
        parameters; i.e. the gradient:

        .. math::

            \frac{\partial \phi}{\partial \mathbf{m}}

        Parameters
        ----------
        m : (n_param, ) numpy.ndarray
            The model for which the gradient is evaluated.

        Returns
        -------
        (n_param, ) numpy.ndarray
            The gradient of the objective function evaluated for the model provided.
        """
        pass

    @abstractmethod
    def deriv2(self, model, v=None):
        r"""Hessian of the objective function evaluated for the model provided.

        Where :math:`\phi (\mathbf{m})` is the objective function, this method
        returns the second-derivative (Hessian) with respect to the model
        parameters:

        .. math::

            \frac{\partial^2 \phi}{\partial \mathbf{m}^2}

        or the second-derivative (Hessian) multiplied by a vector :math:`(\mathbf{v})`:

        .. math::

            \frac{\partial^2 \phi}{\partial \mathbf{m}^2} \, \mathbf{v}

        Parameters
        ----------
        m : (n_param, ) numpy.ndarray
            The model for which the Hessian is evaluated.
        v : None or (n_param, ) numpy.ndarray, optional
            A vector.

        Returns
        -------
        (n_param, n_param) scipy.sparse.csr_matrix or (n_param, ) numpy.ndarray
            If the input argument *v* is ``None``, the Hessian of the objective
            function for the model provided is returned. If *v* is not ``None``,
            the Hessian multiplied by the vector provided is returned.
        """
        pass

    @property
    @abstractmethod
    def nP(self):
        """
        Number of parameters expected in models.
        """
        pass

    def __add__(self, other):
        if isinstance(other, Zero):
            return self
        if not isinstance(other, BaseObjectiveFunction):
            raise TypeError(
                f"Cannot add type '{other.__class__.__name__}' to an objective "
                "function. Only 'BaseObjectiveFunction's can be added together."
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

    def _test_deriv(
        self,
        x=None,
        num=4,
        plotIt=False,
        random_seed: RandomSeed | None = None,
        **kwargs,
    ):
        print("Testing {0!s} Deriv".format(self.__class__.__name__))
        if x is None:
            rng = np.random.default_rng(seed=random_seed)
            n_params = rng.integers(low=100, high=1_000) if self.nP == "*" else self.nP
            x = rng.standard_normal(size=n_params)
        return check_derivative(
            lambda m: [self(m), self.deriv(m)], x, num=num, plotIt=plotIt, **kwargs
        )

    def _test_deriv2(
        self,
        x=None,
        num=4,
        plotIt=False,
        random_seed: RandomSeed | None = None,
        **kwargs,
    ):
        print("Testing {0!s} Deriv2".format(self.__class__.__name__))
        rng = np.random.default_rng(seed=random_seed)
        if x is None:
            n_params = rng.integers(low=100, high=1_000) if self.nP == "*" else self.nP
            x = rng.standard_normal(size=n_params)

        v = x + 0.1 * rng.uniform(size=len(x))
        expectedOrder = kwargs.pop("expectedOrder", 1)
        return check_derivative(
            lambda m: [self.deriv(m).dot(v), self.deriv2(m, v=v)],
            x,
            num=num,
            expectedOrder=expectedOrder,
            plotIt=plotIt,
            **kwargs,
        )

    def test_derivatives(
        self, x=None, num=4, random_seed: RandomSeed | None = None, **kwargs
    ):
        """Run a convergence test on both the first and second derivatives.

        They should be second order!

        Parameters
        ----------
        x : None or (n_param, ) numpy.ndarray, optional
            The evaluation point for the Taylor expansion.
        num : int
            The number of iterations in the convergence test.
        random_seed : :class:`~simpeg.typing.RandomSeed` or None, optional
            Random seed used for generating a random array for ``x`` if it's
            None, and the ``v`` array for testing the second derivatives. It
            can either be an int, a predefined Numpy random number generator,
            or any valid input to ``numpy.random.default_rng``.

        Returns
        -------
        bool
            ``True`` if both tests pass. ``False`` if either test fails.
        """
        deriv = self._test_deriv(x=x, num=num, random_seed=random_seed, **kwargs)
        deriv2 = self._test_deriv2(
            x=x, num=num, plotIt=False, random_seed=random_seed, **kwargs
        )
        return deriv & deriv2


class ComboObjectiveFunction(BaseObjectiveFunction):
    r"""Composite for multiple objective functions.

    This class allows the creation of an objective function :math:`\phi` which is the sum
    of a list of other objective functions :math:`\phi_i`. Each objective function has associated with it
    a multiplier :math:`c_i` such that

    .. math::
        \phi = \sum_{i = 1}^N c_i \phi_i

    Parameters
    ----------
    objfcts : None or list of simpeg.objective_function.BaseObjectiveFunction, optional
        List containing the objective functions that will live inside the
        composite class. If ``None``, an empty list will be created.
    multipliers : None or list of int, optional
        List containing the multipliers for each objective function
        in ``objfcts``.  If ``None``, a list full of ones with the same length
        as ``objfcts`` will be created.
    unpack_on_add : bool
        Whether to unpack the multiple objective functions when adding them to
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

    def __init__(
        self,
        objfcts: list[BaseObjectiveFunction] | None = None,
        multipliers=None,
        unpack_on_add=True,
    ):
        # Define default lists if None
        if objfcts is None:
            objfcts = []
        if multipliers is None:
            multipliers = [1.0 for _ in range(len(objfcts))]

        # Validate inputs
        _check_length_objective_funcs_multipliers(objfcts, multipliers)
        _validate_objective_functions(objfcts)
        for multiplier in multipliers:
            _validate_multiplier(multiplier)

        # Get number of parameters (nP) from objective functions
        number_of_parameters = [f.nP for f in objfcts if f.nP != "*"]
        if number_of_parameters:
            nP = number_of_parameters[0]
        else:
            nP = None

        self._nP = nP
        self.objfcts = objfcts
        self._multipliers = multipliers
        self._unpack_on_add = unpack_on_add

    def __len__(self):
        return len(self.multipliers)

    def __getitem__(self, key):
        return self.multipliers[key], self.objfcts[key]

    @property
    def nP(self):
        """
        Number of parameters
        """
        return self._nP

    @property
    def multipliers(self) -> list[float]:
        r"""Multipliers for the objective functions.

        For a composite objective function :math:`\phi`, that is, a weighted sum of
        objective functions :math:`\phi_i` with multipliers :math:`c_i` such that

        .. math::
            \phi = \sum_{i = 1}^N c_i \phi_i,

        this method returns the multipliers :math:`c_i` in
        the same order of the ``objfcts``.

        Returns
        -------
        list of float
            Multipliers for the objective functions.
        """
        return self._multipliers

    @multipliers.setter
    def multipliers(self, value):
        """Set multipliers attribute after checking if they are valid."""
        for multiplier in value:
            _validate_multiplier(multiplier)
        _check_length_objective_funcs_multipliers(self.objfcts, value)
        self._multipliers = value

    def __call__(self, m, f=None):
        """Evaluate the objective functions for a given model."""
        fct = 0.0
        for i, phi in enumerate(self):
            multiplier, objfct = phi
            if multiplier == 0.0:  # don't evaluate the fct
                continue
            if f is not None and _need_to_pass_fields(objfct):
                objective_func_value = objfct(m, f=f[i])
            else:
                objective_func_value = objfct(m)
            fct += multiplier * objective_func_value
        return fct

    def deriv(self, m, f=None):
        # Docstring inherited from BaseObjectiveFunction
        g = Zero()
        for i, phi in enumerate(self):
            multiplier, objfct = phi
            if multiplier == 0.0:  # don't evaluate the fct
                continue
            if f is not None and _need_to_pass_fields(objfct):
                aux = objfct.deriv(m, f=f[i])
            else:
                aux = objfct.deriv(m)
            if not isinstance(aux, Zero):
                g += multiplier * aux
        return g

    def deriv2(self, m, v=None, f=None):
        # Docstring inherited from BaseObjectiveFunction
        H = Zero()
        for i, phi in enumerate(self):
            multiplier, objfct = phi
            if multiplier == 0.0:  # don't evaluate the fct
                continue
            if f is not None and _need_to_pass_fields(objfct):
                objfct_H = objfct.deriv2(m, v, f=f[i])
            else:
                objfct_H = objfct.deriv2(m, v)
            H = H + multiplier * objfct_H
        return H

    def get_functions_of_type(self, fun_class) -> list:
        """Return objective functions of a given type(s).

        Parameters
        ----------
        fun_class : class or list of classes
            Objective function class or list of objective function classes to return.

        Returns
        -------
        list of simpeg.objective_function.BaseObjectiveFunction
            Objective functions of a given type(s).
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


def _validate_objective_functions(objective_functions):
    """
    Validate objective functions.

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


def _validate_multiplier(multiplier):
    """
    Validate multiplier.

    Check if the multiplier is of a valid type.
    """
    if not isinstance(multiplier, VALID_MULTIPLIERS) or isinstance(multiplier, bool):
        raise TypeError(
            f"Invalid multiplier '{multiplier}' of type '{type(multiplier)}'. "
            "Objective functions can only be multiplied by scalar numbers."
        )


def _check_length_objective_funcs_multipliers(objective_functions, multipliers):
    """
    Check if objective functions and multipliers have the same length.
    """
    if len(objective_functions) != len(multipliers):
        raise ValueError(
            "Inconsistent number of elements between objective functions "
            f"('{len(objective_functions)}') and multipliers "
            f"('{len(multipliers)}'). They must have the same number of parameters."
        )
