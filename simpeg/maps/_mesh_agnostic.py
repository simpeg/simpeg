"""
Mesh independent map classes.
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator
from scipy.constants import mu_0
from scipy.special import expit, logit
from discretize.utils import mkvc, sdiag

from ._base import IdentityMap


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
