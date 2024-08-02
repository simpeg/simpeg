"""
Maps that transform physical properties from one space to another.
"""

import warnings
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator
from scipy.constants import mu_0
from scipy.special import expit, logit
from discretize.utils import mkvc, sdiag, rotation_matrix_from_normals

from ._base import IdentityMap

from ..utils import validate_integer, validate_direction, validate_float, validate_type


class ExpMap(IdentityMap):
    r"""Mapping that computes the natural exponentials of the model parameters.

    Where :math:`\mathbf{m}` is a set of model parameters, ``ExpMap`` creates
    a mapping :math:`\mathbf{u}(\mathbf{m})` that computes the natural exponential
    of every element in :math:`\mathbf{m}`; i.e.:

    .. math::
        \mathbf{u}(\mathbf{m}) = exp(\mathbf{m})

    ``ExpMap`` is commonly used when working with physical properties whose values
    span many orders of magnitude (e.g. the electrical conductivity :math:`\sigma`).
    By using ``ExpMap``, we can invert for a model that represents the natural log
    of a set of physical property values, i.e. when :math:`m = log(\sigma)`

    Parameters
    ----------
    mesh : discretize.BaseMesh
        The number of parameters accepted by the mapping is set to equal the number
        of mesh cells.
    nP : int
        Set the number of parameters accepted by the mapping directly. Used if the
        number of parameters is known. Used generally when the number of parameters
        is not equal to the number of cells in a mesh.
    """

    def __init__(self, mesh=None, nP=None, **kwargs):
        super().__init__(mesh=mesh, nP=nP, **kwargs)

    def _transform(self, m):
        return np.exp(mkvc(m))

    def inverse(self, D):
        r"""Apply the inverse of the exponential mapping to an array.

        For the exponential mapping :math:`\mathbf{u}(\mathbf{m})`, the
        inverse mapping on a variable :math:`\mathbf{x}` is performed by taking
        the natural logarithms of elements, i.e.:

        .. math::
            \mathbf{m} = \mathbf{u}^{-1}(\mathbf{x}) = log(\mathbf{x})

        Parameters
        ----------
        D : numpy.ndarray
            A set of input values

        Returns
        -------
        numpy.ndarray
            A :class:`numpy.ndarray` containing result of applying the
            inverse mapping to the elements in *D*; which in this case
            is the natural logarithm.
        """
        return np.log(mkvc(D))

    def deriv(self, m, v=None):
        r"""Derivative of mapping with respect to the input parameters.

        For a mapping :math:`\mathbf{u}(\mathbf{m})` that computes the natural
        exponential function for each parameter in the model :math:`\mathbf{m}`,
        i.e.:

        .. math::
            \mathbf{u}(\mathbf{m}) = exp(\mathbf{m}),

        the derivative of the mapping with respect to the model is a diagonal
        matrix of the form:

        .. math::
            \frac{\partial \mathbf{u}}{\partial \mathbf{m}}
            = \textrm{diag} \big ( exp(\mathbf{m}) \big )

        Parameters
        ----------
        m : (nP) numpy.ndarray
            A vector representing a set of model parameters
        v : (nP) numpy.ndarray
            If not ``None``, the method returns the derivative times the vector *v*

        Returns
        -------
        scipy.sparse.csr_matrix
            Derivative of the mapping with respect to the model parameters. If the
            input argument *v* is not ``None``, the method returns the derivative times
            the vector *v*.
        """
        deriv = sdiag(np.exp(mkvc(m)))
        if v is not None:
            return deriv * v
        return deriv

    @property
    def is_linear(self):
        return False


class ReciprocalMap(IdentityMap):
    r"""Mapping that computes the reciprocals of the model parameters.

    Where :math:`\mathbf{m}` is a set of model parameters, ``ReciprocalMap``
    creates a mapping :math:`\mathbf{u}(\mathbf{m})` that computes the
    reciprocal of every element in :math:`\mathbf{m}`;
    i.e.:

    .. math::
        \mathbf{u}(\mathbf{m}) = \mathbf{m}^{-1}

    Parameters
    ----------
    mesh : discretize.BaseMesh
        The number of parameters accepted by the mapping is set to equal the number
        of mesh cells.
    nP : int
        Set the number of parameters accepted by the mapping directly. Used if the
        number of parameters is known. Used generally when the number of parameters
        is not equal to the number of cells in a mesh.
    """

    def __init__(self, mesh=None, nP=None, **kwargs):
        super().__init__(mesh=mesh, nP=nP, **kwargs)

    def _transform(self, m):
        return 1.0 / mkvc(m)

    def inverse(self, D):
        r"""Apply the inverse of the reciprocal mapping to an array.

        For the reciprocal mapping :math:`\mathbf{u}(\mathbf{m})`,
        the inverse mapping on a variable :math:`\mathbf{x}` is itself a
        reciprocal mapping, i.e.:

        .. math::
            \mathbf{m} = \mathbf{u}^{-1}(\mathbf{x}) = \mathbf{x}^{-1}

        Parameters
        ----------
        D : numpy.ndarray
            A set of input values

        Returns
        -------
        numpy.ndarray
            A :class:`numpy.ndarray` containing result of applying the
            inverse mapping to the elements in *D*; which in this case
            is just a reciprocal mapping.
        """
        return 1.0 / mkvc(D)

    def deriv(self, m, v=None):
        r"""Derivative of mapping with respect to the input parameters.

        For a mapping that computes the reciprocal for each
        parameter in the model :math:`\mathbf{m}`, i.e.:

        .. math::
            \mathbf{u}(\mathbf{m}) = \mathbf{m}^{-1}

        the derivative of the mapping with respect to the model is a diagonal
        matrix of the form:

        .. math::
            \frac{\partial \mathbf{u}}{\partial \mathbf{m}}
            = \textrm{diag} \big ( -\mathbf{m}^{-2} \big )

        Parameters
        ----------
        m : (nP) numpy.ndarray
            A vector representing a set of model parameters
        v : (nP) numpy.ndarray
            If not ``None``, the method returns the derivative times the vector *v*

        Returns
        -------
        scipy.sparse.csr_matrix
            Derivative of the mapping with respect to the model parameters. If the
            input argument *v* is not ``None``, the method returns the derivative times
            the vector *v*.
        """
        deriv = sdiag(-mkvc(m) ** (-2))
        if v is not None:
            return deriv * v
        return deriv

    @property
    def is_linear(self):
        return False


class LogMap(IdentityMap):
    r"""Mapping that computes the natural logarithm of the model parameters.

    Where :math:`\mathbf{m}` is a set of model parameters, ``LogMap``
    creates a mapping :math:`\mathbf{u}(\mathbf{m})` that computes the
    natural logarithm of every element in
    :math:`\mathbf{m}`; i.e.:

    .. math::
        \mathbf{u}(\mathbf{m}) = \textrm{log}(\mathbf{m})

    Parameters
    ----------
    mesh : discretize.BaseMesh
        The number of parameters accepted by the mapping is set to equal the number
        of mesh cells.
    nP : int
        Set the number of parameters accepted by the mapping directly. Used if the
        number of parameters is known. Used generally when the number of parameters
        is not equal to the number of cells in a mesh.
    """

    def __init__(self, mesh=None, nP=None, **kwargs):
        super().__init__(mesh=mesh, nP=nP, **kwargs)

    def _transform(self, m):
        return np.log(mkvc(m))

    def deriv(self, m, v=None):
        r"""Derivative of mapping with respect to the input parameters.

        For a mapping :math:`\mathbf{u}(\mathbf{m})` that computes the
        natural logarithm for each parameter in the model :math:`\mathbf{m}`,
        i.e.:

        .. math::
            \mathbf{u}(\mathbf{m}) = log(\mathbf{m})

        the derivative of the mapping with respect to the model is a diagonal
        matrix of the form:

        .. math::
            \frac{\partial \mathbf{u}}{\partial \mathbf{m}}
            = \textrm{diag} \big ( \mathbf{m}^{-1} \big )

        Parameters
        ----------
        m : (nP) numpy.ndarray
            A vector representing a set of model parameters
        v : (nP) numpy.ndarray
            If not ``None``, the method returns the derivative times the vector *v*

        Returns
        -------
        scipy.sparse.csr_matrix
            Derivative of the mapping with respect to the model parameters. If the
            input argument *v* is not ``None``, the method returns the derivative times
            the vector *v*.
        """
        mod = mkvc(m)
        deriv = np.zeros(mod.shape)
        tol = 1e-16  # zero
        ind = np.greater_equal(np.abs(mod), tol)
        deriv[ind] = 1.0 / mod[ind]
        if v is not None:
            return sdiag(deriv) * v
        return sdiag(deriv)

    def inverse(self, m):
        r"""Apply the inverse of the natural log mapping to an array.

        For the natural log mapping :math:`\mathbf{u}(\mathbf{m})`,
        the inverse mapping on a variable :math:`\mathbf{x}` is performed by
        taking the natural exponent of the elements, i.e.:

        .. math::
            \mathbf{m} = \mathbf{u}^{-1}(\mathbf{x}) = exp(\mathbf{x})

        Parameters
        ----------
        D : numpy.ndarray
            A set of input values

        Returns
        -------
        numpy.ndarray
            A :class:`numpy.ndarray` containing result of applying the
            inverse mapping to the elements in *D*; which in this case
            is the natural exponent.
        """
        return np.exp(mkvc(m))

    @property
    def is_linear(self):
        return False


class LogisticSigmoidMap(IdentityMap):
    r"""Mapping that computes the logistic sigmoid of the model parameters.

    Where :math:`\mathbf{m}` is a set of model parameters, ``LogisticSigmoidMap`` creates
    a mapping :math:`\mathbf{u}(\mathbf{m})` that computes the logistic sigmoid
    of every element in :math:`\mathbf{m}`; i.e.:

    .. math::
        \mathbf{u}(\mathbf{m}) = sigmoid(\mathbf{m}) = \frac{1}{1+\exp{-\mathbf{m}}}

    ``LogisticSigmoidMap`` transforms values onto the interval (0,1), but can optionally
    be scaled and shifted to the interval (a,b). This can be useful for inversion
    of data that varies over a log scale and bounded on some interval:

    .. math::
        \mathbf{u}(\mathbf{m}) = a + (b - a) \cdot sigmoid(\mathbf{m})

    Parameters
    ----------
    mesh : discretize.BaseMesh
        The number of parameters accepted by the mapping is set to equal the number
        of mesh cells.
    nP : int
        Set the number of parameters accepted by the mapping directly. Used if the
        number of parameters is known. Used generally when the number of parameters
        is not equal to the number of cells in a mesh.
    lower_bound: float or (nP) numpy.ndarray
        lower bound (a) for the transform. Default 0. Defined \in \mathbf{u} space.
    upper_bound: float or (nP) numpy.ndarray
        upper bound (b) for the transform. Default 1. Defined \in \mathbf{u} space.

    """

    def __init__(self, mesh=None, nP=None, lower_bound=0, upper_bound=1, **kwargs):
        super().__init__(mesh=mesh, nP=nP, **kwargs)
        lower_bound = np.atleast_1d(lower_bound)
        upper_bound = np.atleast_1d(upper_bound)
        if self.nP != "*":
            # check if lower bound and upper bound broadcast to nP
            try:
                np.broadcast_shapes(lower_bound.shape, (self.nP,))
            except ValueError as err:
                raise ValueError(
                    f"Lower bound does not broadcast to the number of parameters. "
                    f"Lower bound shape is {lower_bound.shape} and tried against "
                    f"{self.nP} parameters."
                ) from err
            try:
                np.broadcast_shapes(upper_bound.shape, (self.nP,))
            except ValueError as err:
                raise ValueError(
                    f"Upper bound does not broadcast to the number of parameters. "
                    f"Upper bound shape is {upper_bound.shape} and tried against "
                    f"{self.nP} parameters."
                ) from err
        # make sure lower and upper bound broadcast to each other...
        try:
            np.broadcast_shapes(lower_bound.shape, upper_bound.shape)
        except ValueError as err:
            raise ValueError(
                f"Upper bound does not broadcast to the lower bound. "
                f"Shapes {upper_bound.shape} and {lower_bound.shape} "
                f"are incompatible with each other."
            ) from err

        if np.any(lower_bound >= upper_bound):
            raise ValueError(
                "A lower bound is greater than or equal to the upper bound."
            )

        self._lower_bound = lower_bound
        self._upper_bound = upper_bound

    @property
    def lower_bound(self):
        """The lower bound

        Returns
        -------
        numpy.ndarray
        """
        return self._lower_bound

    @property
    def upper_bound(self):
        """The upper bound

        Returns
        -------
        numpy.ndarray
        """
        return self._upper_bound

    def _transform(self, m):
        return self.lower_bound + (self.upper_bound - self.lower_bound) * expit(mkvc(m))

    def inverse(self, m):
        r"""Apply the inverse of the mapping to an array.

        For the logistic sigmoid mapping :math:`\mathbf{u}(\mathbf{m})`, the
        inverse mapping on a variable :math:`\mathbf{x}` is performed by taking
        the log-odds of elements, i.e.:

        .. math::
            \mathbf{m} = \mathbf{u}^{-1}(\mathbf{x}) = logit(\mathbf{x}) = \log \frac{\mathbf{x}}{1 - \mathbf{x}}

        or scaled and translated to interval (a,b):
        .. math::
            \mathbf{m} = logit(\frac{(\mathbf{x} - a)}{b-a})

        Parameters
        ----------
        m : numpy.ndarray
            A set of input values

        Returns
        -------
        numpy.ndarray
            the inverse mapping to the elements in *m*; which in this case
            is the log-odds function with scaled and shifted input.
        """
        return logit(
            (mkvc(m) - self.lower_bound) / (self.upper_bound - self.lower_bound)
        )

    def deriv(self, m, v=None):
        r"""Derivative of mapping with respect to the input parameters.

        For a mapping :math:`\mathbf{u}(\mathbf{m})` the derivative of the mapping with
        respect to the model is a diagonal matrix of the form:

        .. math::
            \frac{\partial \mathbf{u}}{\partial \mathbf{m}}
            = \textrm{diag} \big ( (b-a)\cdot sigmoid(\mathbf{m})\cdot(1-sigmoid(\mathbf{m})) \big )

        Parameters
        ----------
        m : (nP) numpy.ndarray
            A vector representing a set of model parameters
        v : (nP) numpy.ndarray
            If not ``None``, the method returns the derivative times the vector *v*

        Returns
        -------
        numpy.ndarray or scipy.sparse.csr_matrix
            Derivative of the mapping with respect to the model parameters. If the
            input argument *v* is not ``None``, the method returns the derivative times
            the vector *v*.
        """
        sigmoid = expit(mkvc(m))
        deriv = (self.upper_bound - self.lower_bound) * sigmoid * (1.0 - sigmoid)
        if v is not None:
            return deriv * v
        return sdiag(deriv)

    @property
    def is_linear(self):
        return False


class ChiMap(IdentityMap):
    r"""Mapping that computes the magnetic permeability given a set of magnetic susceptibilities.

    Where :math:`\boldsymbol{\chi}` is the input model parameters defining a set of magnetic
    susceptibilities, ``ChiMap`` creates a mapping :math:`\boldsymbol{\mu}(\boldsymbol{\chi})`
    that computes the corresponding magnetic permeabilities of every
    element in :math:`\boldsymbol{\chi}`; i.e.:

    .. math::
        \boldsymbol{\mu}(\boldsymbol{\chi}) = \mu_0 \big (1 + \boldsymbol{\chi} \big )

    where :math:`\mu_0` is the permeability of free space.

    Parameters
    ----------
    mesh : discretize.BaseMesh
        The number of parameters accepted by the mapping is set to equal the number
        of mesh cells.
    nP : int
        Set the number of parameters accepted by the mapping directly. Used if the
        number of parameters is known. Used generally when the number of parameters
        is not equal to the number of cells in a mesh.
    """

    def __init__(self, mesh=None, nP=None, **kwargs):
        super().__init__(mesh=mesh, nP=nP, **kwargs)

    def _transform(self, m):
        return mu_0 * (1 + m)

    def deriv(self, m, v=None):
        r"""Derivative of mapping with respect to the input parameters.

        For a mapping :math:`\boldsymbol{\mu}(\boldsymbol{\chi})` that transforms a
        set of magnetic susceptibilities :math:`\boldsymbol{\chi}` to their corresponding
        magnetic permeabilities, i.e.:

        .. math::
            \boldsymbol{\mu}(\boldsymbol{\chi}) = \mu_0 \big (1 + \boldsymbol{\chi} \big ),

        the derivative of the mapping with respect to the model is the identity
        matrix scaled by the permeability of free-space. Thus:

        .. math::
            \frac{\partial \boldsymbol{\mu}}{\partial \boldsymbol{\chi}} = \mu_0 \mathbf{I}

        Parameters
        ----------
        m : (nP) numpy.ndarray
            A vector representing a set of model parameters
        v : (nP) numpy.ndarray
            If not ``None``, the method returns the derivative times the vector *v*

        Returns
        -------
        scipy.sparse.csr_matrix
            Derivative of the mapping with respect to the model parameters. If the
            input argument *v* is not ``None``, the method returns the derivative times
            the vector *v*.
        """
        if v is not None:
            return mu_0 * v
        return mu_0 * sp.eye(self.nP)

    def inverse(self, m):
        r"""Apply the inverse mapping to an array.

        For the ``ChiMap`` class, the inverse mapping recoveres the set of
        magnetic susceptibilities :math:`\boldsymbol{\chi}` from a set of
        magnetic permeabilities :math:`\boldsymbol{\mu}`. Thus the inverse
        mapping is defined as:

        .. math::
            \boldsymbol{\chi}(\boldsymbol{\mu}) = \frac{\boldsymbol{\mu}}{\mu_0} - 1

        where :math:`\mu_0` is the permeability of free space.

        Parameters
        ----------
        D : numpy.ndarray
            A set of input values

        Returns
        -------
        numpy.ndarray
            A :class:`numpy.ndarray` containing result of applying the
            inverse mapping to the elements in *D*; which in this case
            represents the conversion of magnetic permeabilities
            to their corresponding magnetic susceptibility values.
        """
        return m / mu_0 - 1


class MuRelative(IdentityMap):
    r"""Mapping that computes the magnetic permeability given a set of relative permeabilities.

    Where :math:`\boldsymbol{\mu_r}` defines a set of relative permeabilities, ``MuRelative``
    creates a mapping :math:`\boldsymbol{\mu}(\boldsymbol{\mu_r})` that computes the
    corresponding magnetic permeabilities of every element in :math:`\boldsymbol{\mu_r}`;
    i.e.:

    .. math::
        \boldsymbol{\mu}(\boldsymbol{\mu_r}) = \mu_0 \boldsymbol{\mu_r}

    where :math:`\mu_0` is the permeability of free space.

    Parameters
    ----------
    mesh : discretize.BaseMesh
        The number of parameters accepted by the mapping is set to equal the number
        of mesh cells.
    nP : int
        Set the number of parameters accepted by the mapping directly. Used if the
        number of parameters is known. Used generally when the number of parameters
        is not equal to the number of cells in a mesh.
    """

    def __init__(self, mesh=None, nP=None, **kwargs):
        super().__init__(mesh=mesh, nP=nP, **kwargs)

    def _transform(self, m):
        return mu_0 * m

    def deriv(self, m, v=None):
        r"""Derivative of mapping with respect to the input parameters.

        For a mapping that transforms a set of relative permeabilities
        :math:`\boldsymbol{\mu_r}` to their corresponding magnetic permeabilities, i.e.:

        .. math::
            \boldsymbol{\mu}(\boldsymbol{\mu_r}) = \mu_0 \boldsymbol{\mu_r},

        the derivative of the mapping with respect to the model is the identity
        matrix scaled by the permeability of free-space. Thus:

        .. math::
            \frac{\partial \boldsymbol{\mu}}{\partial \boldsymbol{\mu_r}} = \mu_0 \mathbf{I}

        Parameters
        ----------
        m : (nP) numpy.ndarray
            A vector representing a set of model parameters
        v : (nP) numpy.ndarray
            If not ``None``, the method returns the derivative times the vector *v*

        Returns
        -------
        scipy.sparse.csr_matrix
            Derivative of the mapping with respect to the model parameters. If the
            input argument *v* is not ``None``, the method returns the derivative times
            the vector *v*.
        """
        if v is not None:
            return mu_0 * v
        return mu_0 * sp.eye(self.nP)

    def inverse(self, m):
        r"""Apply the inverse mapping to an array.

        For the ``MuRelative`` class, the inverse mapping recoveres the set of
        relative permeabilities :math:`\boldsymbol{\mu_r}` from a set of
        magnetic permeabilities :math:`\boldsymbol{\mu}`. Thus the inverse
        mapping is defined as:

        .. math::
            \boldsymbol{\mu_r}(\boldsymbol{\mu}) = \frac{\boldsymbol{\mu}}{\mu_0}

        where :math:`\mu_0` is the permeability of free space.

        Parameters
        ----------
        D : numpy.ndarray
            A set of input values

        Returns
        -------
        numpy.ndarray
            A :class:`numpy.ndarray` containing result of applying the
            inverse mapping to the elements in *D*; which in this case
            represents the conversion of magnetic permeabilities
            to their corresponding relative permeability values.
        """
        return 1.0 / mu_0 * m


class Weighting(IdentityMap):
    r"""Mapping that scales the elements of the model by a corresponding set of weights.

    Where :math:`\mathbf{m}` defines the set of input model parameters and
    :math:`\mathbf{w}` represents a corresponding set of model weight,
    ``Weighting`` constructs a mapping :math:`\mathbf{u}(\mathbf{m})` of the form:

    .. math::
        \mathbf{u}(\mathbf{m}) = \mathbf{w} \odot \mathbf{m}

    where :math:`\odot` is the Hadamard product. The mapping may also be
    defined using a linear operator as follows:

    .. math::
        \mathbf{u}(\mathbf{m}) = \mathbf{Pm} \;\;\;\;\; \textrm{where} \;\;\;\;\; \mathbf{P} = diag(\mathbf{w})

    Parameters
    ----------
    mesh : discretize.BaseMesh
        The number of parameters accepted by the mapping is set to equal the number
        of mesh cells.
    nP : int
        Set the number of parameters accepted by the mapping directly. Used if the
        number of parameters is known. Used generally when the number of parameters
        is not equal to the number of cells in a mesh.
    weights : (nP) numpy.ndarray
        A set of independent model weights. If ``None``, all model weights are set
        to *1*.
    """

    def __init__(self, mesh=None, nP=None, weights=None, **kwargs):
        if "nC" in kwargs:
            raise TypeError(
                "`nC` has been removed. Use `nP` to set the number of model "
                "parameters."
            )

        super(Weighting, self).__init__(mesh=mesh, nP=nP, **kwargs)

        if weights is None:
            weights = np.ones(self.nP)

        self.weights = np.array(weights, dtype=float)

    @property
    def shape(self):
        """Dimensions of the mapping.

        Returns
        -------
        tuple
            Dimensions of the mapping. Where *nP* is the number of parameters
            the mapping acts on, this method returns a tuple of the form
            (*nP*, *nP*).
        """
        return (self.nP, self.nP)

    @property
    def P(self):
        r"""The linear mapping operator

        This property returns the sparse matrix :math:`\mathbf{P}` that carries
        out the weighting mapping via matrix-vector product, i.e.:

        .. math::
            \mathbf{u}(\mathbf{m}) = \mathbf{Pm} \;\;\;\;\; \textrm{where} \;\;\;\;\; \mathbf{P} = diag(\mathbf{w})

        Returns
        -------
        scipy.sparse.csr_matrix
            Sparse linear mapping operator
        """
        return sdiag(self.weights)

    def _transform(self, m):
        return self.weights * m

    def inverse(self, D):
        r"""Apply the inverse of the weighting mapping to an array.

        For the weighting mapping :math:`\mathbf{u}(\mathbf{m})`, the inverse
        mapping on a variable :math:`\mathbf{x}` is performed by multplying each element by
        the reciprocal of its corresponding weighting value, i.e.:

        .. math::
            \mathbf{m} = \mathbf{u}^{-1}(\mathbf{x}) = \mathbf{w}^{-1} \odot \mathbf{x}

        where :math:`\odot` is the Hadamard product. The inverse mapping may also be defined
        using a linear operator as follows:

        .. math::
             \mathbf{m} = \mathbf{u}^{-1}(\mathbf{x}) = \mathbf{P^{-1} m}
             \;\;\;\;\; \textrm{where} \;\;\;\;\; \mathbf{P} = diag(\mathbf{w})

        Parameters
        ----------
        D : numpy.ndarray
            A set of input values

        Returns
        -------
        numpy.ndarray
            A :class:`numpy.ndarray` containing result of applying the
            inverse mapping to the elements in *D*; which in this case
            is simply dividing each element by its corresponding
            weight.
        """
        return self.weights ** (-1.0) * D

    def deriv(self, m, v=None):
        r"""Derivative of mapping with respect to the input parameters.

        For a weighting mapping :math:`\mathbf{u}(\mathbf{m})` that scales the
        input parameters in the model :math:`\mathbf{m}` by their corresponding
        weights :math:`\mathbf{w}`; i.e.:

        .. math::
            \mathbf{u}(\mathbf{m}) = \mathbf{w} \dot \mathbf{m},

        the derivative of the mapping with respect to the model is a diagonal
        matrix of the form:

        .. math::
            \frac{\partial \mathbf{u}}{\partial \mathbf{m}}
            = diag (\mathbf{w})

        Parameters
        ----------
        m : (nP) numpy.ndarray
            A vector representing a set of model parameters
        v : (nP) numpy.ndarray
            If not ``None``, the method returns the derivative times the vector *v*

        Returns
        -------
        scipy.sparse.csr_matrix
            Derivative of the mapping with respect to the model parameters. If the
            input argument *v* is not ``None``, the method returns the derivative times
            the vector *v*.
        """
        if v is not None:
            return self.weights * v
        return self.P


class ComplexMap(IdentityMap):
    r"""Maps the real and imaginary component values stored in a model to complex values.

    Let :math:`\mathbf{m}` be a model which stores the real and imaginary components of
    a set of complex values :math:`\mathbf{z}`. Where the model parameters are organized
    into a vector of the form
    :math:`\mathbf{m} = [\mathbf{z}^\prime , \mathbf{z}^{\prime\prime}]`, ``ComplexMap``
    constructs the following mapping:

    .. math::
        \mathbf{z}(\mathbf{m}) = \mathbf{z}^\prime + j \mathbf{z}^{\prime\prime}

    Note that the mapping is :math:`\mathbb{R}^{2n} \rightarrow \mathbb{C}^n`.

    Parameters
    ----------
    mesh : discretize.BaseMesh
        If a mesh is used to construct the mapping, the number of input model
        parameters is *2\*mesh.nC* and the number of complex values output from
        the mapping is equal to *mesh.nC*. If *mesh* is ``None``, the dimensions
        of the mapping are set using the *nP* input argument.
    nP : int
        Defines the number of input model parameters directly. Must be an even number!!!
        In this case, the number of complex values output from the mapping is *nP/2*.
        If *nP* = ``None``, the dimensions of the mapping are set using the *mesh*
        input argument.

    Examples
    --------
    Here we construct a complex mapping on a 1D mesh comprised
    of 4 cells. The input model is real-valued array of length 8
    (4 real and 4 imaginary values). The output of the mapping
    is a complex array with 4 values.

    >>> from simpeg.maps import ComplexMap
    >>> from discretize import TensorMesh
    >>> import numpy as np

    >>> nC = 4
    >>> mesh = TensorMesh([np.ones(nC)])

    >>> z_real = np.ones(nC)
    >>> z_imag = 2*np.ones(nC)
    >>> m = np.r_[z_real, z_imag]
    >>> m
    array([1., 1., 1., 1., 2., 2., 2., 2.])

    >>> mapping = ComplexMap(mesh=mesh)
    >>> z = mapping * m
    >>> z
    array([1.+2.j, 1.+2.j, 1.+2.j, 1.+2.j])

    """

    def __init__(self, mesh=None, nP=None, **kwargs):
        super().__init__(mesh=mesh, nP=nP, **kwargs)
        if nP is not None and mesh is not None:
            assert (
                2 * mesh.nC == nP
            ), "Number parameters must be 2 X number of mesh cells."
        if nP is not None:
            assert nP % 2 == 0, "nP must be even."
        self._nP = nP or int(self.mesh.nC * 2)

    @property
    def nP(self):
        r"""Number of parameters the mapping acts on.

        Returns
        -------
        int or '*'
            Number of parameters that the mapping acts on.
        """
        return self._nP

    @property
    def shape(self):
        """Dimensions of the mapping

        Returns
        -------
        tuple
            The dimensions of the mapping. Where *nP* is the number
            of input parameters, this property returns a tuple
            (*nP/2*, *nP*).
        """
        return (int(self.nP / 2), self.nP)

    def _transform(self, m):
        nC = int(self.nP / 2)
        return m[:nC] + m[nC:] * 1j

    def deriv(self, m, v=None):
        r"""Derivative of the complex mapping with respect to the input parameters.

        The complex mapping maps the real and imaginary components stored in a model
        of the form :math:`\mathbf{m} = [\mathbf{z}^\prime , \mathbf{z}^{\prime\prime}]`
        to their corresponding complex values :math:`\mathbf{z}`, i.e.

        .. math::
            \mathbf{z}(\mathbf{m}) = \mathbf{z}^\prime + j \mathbf{z}^{\prime\prime}

        The derivative of the mapping with respect to the model is block
        matrix of the form:

        .. math::
            \frac{\partial \mathbf{z}}{\partial \mathbf{m}} = \big ( \mathbf{I} \;\;\; j\mathbf{I} \big )

        where :math:`\mathbf{I}` is the identity matrix of shape (*nP/2*, *nP/2*) and
        :math:`j = \sqrt{-1}`.

        Parameters
        ----------
        m : (nP) numpy.ndarray
            A vector representing a set of model parameters
        v : (nP) numpy.ndarray
            If not ``None``, the method returns the derivative times the vector *v*

        Returns
        -------
        scipy.sparse.csr_matrix
            Derivative of the mapping with respect to the model parameters. If the
            input argument *v* is not ``None``, the method returns the derivative times
            the vector *v*.

        Examples
        --------
        Here we construct the derivative operator for the complex mapping on a 1D
        mesh comprised of 4 cells. We then demonstrate how the derivative of the
        mapping and its adjoint can be applied to a vector.

        >>> from simpeg.maps import ComplexMap
        >>> from discretize import TensorMesh
        >>> import numpy as np

        >>> nC = 4
        >>> mesh = TensorMesh([np.ones(nC)])

        >>> m = np.random.rand(2*nC)
        >>> mapping = ComplexMap(mesh=mesh)
        >>> M = mapping.deriv(m)

        When applying the derivative operator to a vector, it will convert
        the real and imaginary values stored in the vector to
        complex values; essentially applying the mapping.

        >>> v1 = np.arange(0, 2*nC, 1)
        >>> u1 = M * v1
        >>> u1
        array([0.+4.j, 1.+5.j, 2.+6.j, 3.+7.j])

        When applying the adjoint of the derivative operator to a set of
        complex values, the operator will decompose these values into
        their real and imaginary components.

        >>> v2 = np.arange(0, nC, 1) + 1j*np.arange(nC, 2*nC, 1)
        >>> u2 = M.adjoint() * v2
        >>> u2
        array([0., 1., 2., 3., 4., 5., 6., 7.])

        """
        nC = self.shape[0]
        shp = (nC, nC * 2)

        def fwd(v):
            return v[:nC] + v[nC:] * 1j

        def adj(v):
            return np.r_[v.real, v.imag]

        if v is not None:
            return LinearOperator(shp, matvec=fwd, rmatvec=adj) * v
        return LinearOperator(shp, matvec=fwd, rmatvec=adj)


class SelfConsistentEffectiveMedium(IdentityMap):
    r"""
        Two phase self-consistent effective medium theory mapping for
        ellipsoidal inclusions. The inversion model is the concentration
        (volume fraction) of the phase 2 material.

        The inversion model is :math:`\varphi`. We solve for :math:`\sigma`
        given :math:`\sigma_0`, :math:`\sigma_1` and :math:`\varphi` . Each of
        the following are implicit expressions of the effective conductivity.
        They are solved using a fixed point iteration.

        **Spherical Inclusions**

        If the shape of the inclusions are spheres, we use

        .. math::

            \sum_{j=1}^N (\sigma^* - \sigma_j)R^{j} = 0

        where :math:`j=[1,N]` is the each material phase, and N is the number
        of phases. Currently, the implementation is only set up for 2 phase
        materials, so we solve

        .. math::

            (1-\\varphi)(\sigma - \sigma_0)R^{(0)} + \varphi(\sigma - \sigma_1)R^{(1)} = 0.

        Where :math:`R^{(j)}` is given by

        .. math::

            R^{(j)} = \left[1 + \frac{1}{3}\frac{\sigma_j - \sigma}{\sigma} \right]^{-1}.

        **Ellipsoids**

        .. todo::

            Aligned Ellipsoids have not yet been implemented, only randomly
            oriented ellipsoids

        If the inclusions are aligned ellipsoids, we solve

        .. math::

            \sum_{j=1}^N \varphi_j (\Sigma^* - \sigma_j\mathbf{I}) \mathbf{R}^{j, *} = 0

        where

        .. math::

            \mathbf{R}^{(j, *)} = \left[ \mathbf{I} + \mathbf{A}_j {\Sigma^{*}}^{-1}(\sigma_j \mathbf{I} - \Sigma^*) \\right]^{-1}

        and the depolarization tensor :math:`\mathbf{A}_j` is given by

        .. math::

            \mathbf{A}^* = \left[\begin{array}{ccc}
                Q & 0 & 0 \\
                0 & Q & 0 \\
                0 & 0 & 1-2Q
            \end{array}\right]

        for a spheroid aligned along the z-axis. For an oblate spheroid
        (:math:`\alpha < 1`, pancake-like)

        .. math::

            Q = \frac{1}{2}\left(
                1 + \frac{1}{\alpha^2 - 1} \left[
                    1 - \frac{1}{\chi}\tan^{-1}(\chi)
                \right]
            \right)

        where

        .. math::

            \chi = \sqrt{\frac{1}{\alpha^2} - 1}


        For reference, see
        `Torquato (2002), Random Heterogeneous Materials <https://link.springer.com/book/10.1007/978-1-4757-6355-3>`_


    """

    def __init__(
        self,
        mesh=None,
        nP=None,
        sigma0=None,
        sigma1=None,
        alpha0=1.0,
        alpha1=1.0,
        orientation0="z",
        orientation1="z",
        random=True,
        rel_tol=1e-3,
        maxIter=50,
        **kwargs,
    ):
        self._sigstart = None
        self.sigma0 = sigma0
        self.sigma1 = sigma1
        self.alpha0 = alpha0
        self.alpha1 = alpha1
        self.orientation0 = orientation0
        self.orientation1 = orientation1
        self.random = random
        self.rel_tol = rel_tol
        self.maxIter = maxIter
        super(SelfConsistentEffectiveMedium, self).__init__(mesh, nP, **kwargs)

    @property
    def sigma0(self):
        """Physical property value for phase-0 material.

        Returns
        -------
        float
        """
        return self._sigma0

    @sigma0.setter
    def sigma0(self, value):
        self._sigma0 = validate_float("sigma0", value, min_val=0.0)

    @property
    def sigma1(self):
        """Physical property value for phase-1 material.

        Returns
        -------
        float
        """
        return self._sigma1

    @sigma1.setter
    def sigma1(self, value):
        self._sigma1 = validate_float("sigma1", value, min_val=0.0)

    @property
    def alpha0(self):
        """Aspect ratio of the phase-0 ellipsoids.

        Returns
        -------
        float
        """
        return self._alpha0

    @alpha0.setter
    def alpha0(self, value):
        self._alpha0 = validate_float("alpha0", value, min_val=0.0)

    @property
    def alpha1(self):
        """Aspect ratio of the phase-1 ellipsoids.

        Returns
        -------
        float
        """
        return self._alpha1

    @alpha1.setter
    def alpha1(self, value):
        self._alpha1 = validate_float("alpha1", value, min_val=0.0)

    @property
    def orientation0(self):
        """Orientation of the phase-0 inclusions.

        Returns
        -------
        numpy.ndarray
        """
        return self._orientation0

    @orientation0.setter
    def orientation0(self, value):
        self._orientation0 = validate_direction("orientation0", value, dim=3)

    @property
    def orientation1(self):
        """Orientation of the phase-0 inclusions.

        Returns
        -------
        numpy.ndarray
        """
        return self._orientation1

    @orientation1.setter
    def orientation1(self, value):
        self._orientation1 = validate_direction("orientation1", value, dim=3)

    @property
    def random(self):
        """Are the inclusions randomly oriented (True) or preferentially aligned (False)?

        Returns
        -------
        bool
        """
        return self._random

    @random.setter
    def random(self, value):
        self._random = validate_type("random", value, bool)

    @property
    def rel_tol(self):
        """relative tolerance for convergence for the fixed-point iteration.

        Returns
        -------
        float
        """
        return self._rel_tol

    @rel_tol.setter
    def rel_tol(self, value):
        self._rel_tol = validate_float(
            "rel_tol", value, min_val=0.0, inclusive_min=False
        )

    @property
    def maxIter(self):
        """Maximum number of iterations for the fixed point iteration calculation.

        Returns
        -------
        int
        """
        return self._maxIter

    @maxIter.setter
    def maxIter(self, value):
        self._maxIter = validate_integer("maxIter", value, min_val=0)

    @property
    def tol(self):
        """
        absolute tolerance for the convergence of the fixed point iteration
        calc
        """
        if getattr(self, "_tol", None) is None:
            self._tol = self.rel_tol * min(self.sigma0, self.sigma1)
        return self._tol

    @property
    def sigstart(self):
        """
        first guess for sigma
        """
        return self._sigstart

    @sigstart.setter
    def sigstart(self, value):
        if value is not None:
            value = validate_float("sigstart", value)
        self._sigstart = value

    def wiener_bounds(self, phi1):
        """Define Wenner Conductivity Bounds

        See Torquato, 2002
        """
        phi0 = 1.0 - phi1
        sigWup = phi0 * self.sigma0 + phi1 * self.sigma1
        sigWlo = 1.0 / (phi0 / self.sigma0 + phi1 / self.sigma1)
        W = np.array([sigWlo, sigWup])

        return W

    def hashin_shtrikman_bounds(self, phi1):
        """Hashin Shtrikman bounds

        See Torquato, 2002
        """
        # TODO: this should probably exsist on its own as a util

        phi0 = 1.0 - phi1
        sigWu = self.wiener_bounds(phi1)[1]
        sig_tilde = phi0 * self.sigma1 + phi1 * self.sigma0

        sigma_min = np.min([self.sigma0, self.sigma1])
        sigma_max = np.max([self.sigma0, self.sigma1])

        sigHSlo = sigWu - (
            (phi0 * phi1 * (self.sigma0 - self.sigma1) ** 2)
            / (sig_tilde + 2 * sigma_max)
        )
        sigHSup = sigWu - (
            (phi0 * phi1 * (self.sigma0 - self.sigma1) ** 2)
            / (sig_tilde + 2 * sigma_min)
        )

        return np.array([sigHSlo, sigHSup])

    def hashin_shtrikman_bounds_anisotropic(self, phi1):
        """Hashin Shtrikman bounds for anisotropic media

        See Torquato, 2002
        """
        phi0 = 1.0 - phi1
        sigWu = self.wiener_bounds(phi1)[1]

        sigma_min = np.min([self.sigma0, self.sigma1])
        sigma_max = np.max([self.sigma0, self.sigma1])

        phi_min = phi0 if self.sigma1 > self.sigma0 else phi1
        phi_max = phi1 if self.sigma1 > self.sigma0 else phi0

        amax = (
            -phi0
            * phi1
            * self.getA(
                self.alpha1 if self.sigma1 > self.sigma0 else self.alpha0,
                self.orientation1 if self.sigma1 > self.sigma0 else self.orientation0,
            )
        )
        I = np.eye(3)

        sigHSlo = sigWu * I + (
            (sigma_min - sigma_max) ** 2
            * amax
            * np.linalg.inv(sigma_min * I + (sigma_min - sigma_max) / phi_max * amax)
        )
        sigHSup = sigWu * I + (
            (sigma_max - sigma_min) ** 2
            * amax
            * np.linalg.inv(sigma_max * I + (sigma_max - sigma_min) / phi_min * amax)
        )

        return [sigHSlo, sigHSup]

    def getQ(self, alpha):
        """Geometric factor in the depolarization tensor"""
        if alpha < 1.0:  # oblate spheroid
            chi = np.sqrt((1.0 / alpha**2.0) - 1)
            return (
                1.0 / 2.0 * (1 + 1.0 / (alpha**2.0 - 1) * (1.0 - np.arctan(chi) / chi))
            )
        elif alpha > 1.0:  # prolate spheroid
            chi = np.sqrt(1 - (1.0 / alpha**2.0))
            return (
                1.0
                / 2.0
                * (
                    1
                    + 1.0
                    / (alpha**2.0 - 1)
                    * (1.0 - 1.0 / (2.0 * chi) * np.log((1 + chi) / (1 - chi)))
                )
            )
        elif alpha == 1:  # sphere
            return 1.0 / 3.0

    def getA(self, alpha, orientation):
        """Depolarization tensor"""
        Q = self.getQ(alpha)
        A = np.diag([Q, Q, 1 - 2 * Q])
        R = rotation_matrix_from_normals(np.r_[0.0, 0.0, 1.0], orientation)
        return (R.T).dot(A).dot(R)

    def getR(self, sj, se, alpha, orientation=None):
        """Electric field concentration tensor"""
        if self.random is True:  # isotropic
            if alpha == 1.0:
                return 3.0 * se / (2.0 * se + sj)
            Q = self.getQ(alpha)
            return (
                se
                / 3.0
                * (2.0 / (se + Q * (sj - se)) + 1.0 / (sj - 2.0 * Q * (sj - se)))
            )
        else:  # anisotropic
            if orientation is None:
                raise Exception("orientation must be provided if random=False")
            I = np.eye(3)
            seinv = np.linalg.inv(se)
            Rinv = I + self.getA(alpha, orientation) * seinv * (sj * I - se)
            return np.linalg.inv(Rinv)

    def getdR(self, sj, se, alpha, orientation=None):
        """
        Derivative of the electric field concentration tensor with respect
        to the concentration of the second phase material.
        """
        if self.random is True:
            if alpha == 1.0:
                return 3.0 / (2.0 * se + sj) - 6.0 * se / (2.0 * se + sj) ** 2
            Q = self.getQ(alpha)
            return (
                1
                / 3
                * (
                    2.0 / (se + Q * (sj - se))
                    + 1.0 / (sj - 2.0 * Q * (sj - se))
                    + se
                    * (
                        -2 * (1 - Q) / (se + Q * (sj - se)) ** 2
                        - 2 * Q / (sj - 2.0 * Q * (sj - se)) ** 2
                    )
                )
            )
        else:
            if orientation is None:
                raise Exception("orientation must be provided if random=False")
            raise NotImplementedError

    def _sc2phaseEMTSpheroidstransform(self, phi1):
        """
        Self Consistent Effective Medium Theory Model Transform,
        alpha = aspect ratio (c/a <= 1)
        """

        if not (np.all(0 <= phi1) and np.all(phi1 <= 1)):
            warnings.warn("there are phis outside bounds of 0 and 1", stacklevel=2)
            phi1 = np.median(np.c_[phi1 * 0, phi1, phi1 * 0 + 1.0])

        phi0 = 1.0 - phi1

        # starting guess
        if self.sigstart is None:
            sige1 = np.mean(self.wiener_bounds(phi1))
        else:
            sige1 = self.sigstart

        if self.random is False:
            sige1 = sige1 * np.eye(3)

        for _ in range(self.maxIter):
            R0 = self.getR(self.sigma0, sige1, self.alpha0, self.orientation0)
            R1 = self.getR(self.sigma1, sige1, self.alpha1, self.orientation1)

            den = phi0 * R0 + phi1 * R1
            num = phi0 * self.sigma0 * R0 + phi1 * self.sigma1 * R1

            if self.random is True:
                sige2 = num / den
                relerr = np.abs(sige2 - sige1)
            else:
                sige2 = num * np.linalg.inv(den)
                relerr = np.linalg.norm(np.abs(sige2 - sige1).flatten(), np.inf)

            if np.all(relerr <= self.tol):
                if self.sigstart is None:
                    self._sigstart = (
                        sige2  # store as a starting point for the next time around
                    )
                return sige2

            sige1 = sige2
        # TODO: make this a proper warning, and output relevant info (sigma0, sigma1, phi, sigstart, and relerr)
        warnings.warn("Maximum number of iterations reached", stacklevel=2)

        return sige2

    def _sc2phaseEMTSpheroidsinversetransform(self, sige):
        R0 = self.getR(self.sigma0, sige, self.alpha0, self.orientation0)
        R1 = self.getR(self.sigma1, sige, self.alpha1, self.orientation1)

        num = -(self.sigma0 - sige) * R0
        den = (self.sigma1 - sige) * R1 - (self.sigma0 - sige) * R0

        return num / den

    def _sc2phaseEMTSpheroidstransformDeriv(self, sige, phi1):
        phi0 = 1.0 - phi1

        R0 = self.getR(self.sigma0, sige, self.alpha0, self.orientation0)
        R1 = self.getR(self.sigma1, sige, self.alpha1, self.orientation1)

        dR0 = self.getdR(self.sigma0, sige, self.alpha0, self.orientation0)
        dR1 = self.getdR(self.sigma1, sige, self.alpha1, self.orientation1)

        num = (sige - self.sigma0) * R0 - (sige - self.sigma1) * R1
        den = phi0 * (R0 + (sige - self.sigma0) * dR0) + phi1 * (
            R1 + (sige - self.sigma1) * dR1
        )

        return sdiag(num / den)

    def _transform(self, m):
        return self._sc2phaseEMTSpheroidstransform(m)

    def deriv(self, m):
        """
        Derivative of the effective conductivity with respect to the
        volume fraction of phase 2 material
        """
        sige = self._transform(m)
        return self._sc2phaseEMTSpheroidstransformDeriv(sige, m)

    def inverse(self, sige):
        """
        Compute the concentration given the effective conductivity
        """
        return self._sc2phaseEMTSpheroidsinversetransform(sige)

    @property
    def is_linear(self):
        return False
