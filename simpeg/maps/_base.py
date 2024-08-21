"""
Base and general map classes.
"""

from __future__ import annotations  # needed to use type operands in Python 3.8

from collections import namedtuple
import discretize
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix as csr
from discretize.tests import check_derivative
from discretize.utils import Zero, Identity, mkvc, speye, sdiag

from ..utils import (
    mat_utils,
    validate_type,
    validate_ndarray_with_shape,
    validate_list_of_types,
    validate_active_indices,
    validate_integer,
    validate_float,
)
from ..typing import RandomSeed


class IdentityMap:
    r"""Identity mapping and the base mapping class for all other SimPEG mappings.

    The ``IdentityMap`` class is used to define the mapping when
    the model parameters are the same as the parameters used in the forward
    simulation. For a discrete set of model parameters :math:`\mathbf{m}`,
    the mapping :math:`\mathbf{u}(\mathbf{m})` is equivalent to applying
    the identity matrix; i.e.:

    .. math::

        \mathbf{u}(\mathbf{m}) = \mathbf{Im}

    The ``IdentityMap`` also acts as the base class for all other SimPEG mapping classes.

    Using the *mesh* or *nP* input arguments, the dimensions of the corresponding
    mapping operator can be permanently set; i.e. (*mesh.nC*, *mesh.nC*) or (*nP*, *nP*).
    However if both input arguments *mesh* and *nP* are ``None``, the shape of
    mapping operator is arbitrary and can act on any vector; i.e. has shape (``*``, ``*``).

    Parameters
    ----------
    mesh : discretize.BaseMesh
        The number of parameters accepted by the mapping is set to equal the number
        of mesh cells.
    nP : int, or '*'
        Set the number of parameters accepted by the mapping directly. Used if the
        number of parameters is known. Used generally when the number of parameters
        is not equal to the number of cells in a mesh.
    """

    def __init__(self, mesh=None, nP=None, **kwargs):
        if (isinstance(nP, str) and nP == "*") or nP is None:
            if mesh is not None:
                nP = mesh.n_cells
            else:
                nP = "*"
        else:
            try:
                nP = int(nP)
            except (TypeError, ValueError) as err:
                raise TypeError(
                    f"Unrecognized input of {repr(nP)} for number of parameters, must be an integer or '*'."
                ) from err
        self.mesh = mesh
        self._nP = nP

        super().__init__(**kwargs)

    @property
    def nP(self):
        r"""Number of parameters the mapping acts on.

        Returns
        -------
        int or ``*``
            Number of parameters that the mapping acts on. Returns an
            ``int`` if the dimensions of the mapping are set. If the
            mapping can act on a vector of any length, ``*`` is returned.
        """
        if self._nP != "*":
            return int(self._nP)
        if self.mesh is None:
            return "*"
        return int(self.mesh.nC)

    @property
    def shape(self):
        r"""Dimensions of the mapping operator

        The dimensions of the mesh depend on the input arguments used
        during instantiation. If *mesh* is used to define the
        identity map, the shape of mapping operator is (*mesh.nC*, *mesh.nC*).
        If *nP* is used to define the identity map, the mapping operator
        has dimensions (*nP*, *nP*). However if both *mesh* and *nP* are
        used to define the identity map, the mapping will have shape
        (*mesh.nC*, *nP*)! And if *mesh* and *nP* were ``None`` when
        instantiating, the mapping has dimensions (``*``, ``*``) and may
        act on a vector of any length.

        Returns
        -------
        tuple
            Dimensions of the mapping operator. If the dimensions of
            the mapping are set, the return is a tuple (``int``,``int``).
            If the mapping can act on a vector of arbitrary length, the
            return is a tuple (``*``, ``*``).
        """
        if self.mesh is None:
            return (self.nP, self.nP)
        return (self.mesh.nC, self.nP)

    def _transform(self, m):
        """
        Changes the model into the physical property.

        .. note::

            This can be called by the __mul__ property against a
            :meth:numpy.ndarray.

        :param numpy.ndarray m: model
        :rtype: numpy.ndarray
        :return: transformed model

        """
        return m

    def inverse(self, D):
        """
        The transform inverse is not implemented.
        """
        raise NotImplementedError("The transform inverse is not implemented.")

    def deriv(self, m, v=None):
        r"""Derivative of the mapping with respect to the input parameters.

        Parameters
        ----------
        m : (nP) numpy.ndarray
            A vector representing a set of model parameters
        v : (nP) numpy.ndarray
            If not ``None``, the method returns the derivative times the vector *v*

        Returns
        -------
        scipy.sparse.csr_matrix or numpy.ndarray
            Derivative of the mapping with respect to the model parameters. For an
            identity mapping, this is just a sparse identity matrix. If the input
            argument *v* is not ``None``, the method returns the derivative times
            the vector *v*; which in this case is just *v*.

        Notes
        -----
        Let :math:`\mathbf{m}` be a set of model parameters and let :math:`\mathbf{I}`
        denote the identity map. Where the identity mapping acting on the model parameters
        can be expressed as:

        .. math::
            \mathbf{u} = \mathbf{I m},

        the **deriv** method returns the derivative of :math:`\mathbf{u}` with respect
        to the model parameters; i.e.:

        .. math::
            \frac{\partial \mathbf{u}}{\partial \mathbf{m}} = \mathbf{I}

        For the Identity map **deriv** simply returns a sparse identity matrix.
        """
        if v is not None:
            return v
        if isinstance(self.nP, (int, np.integer)):
            return sp.identity(self.nP)
        return Identity()

    def test(self, m=None, num=4, random_seed: RandomSeed | None = None, **kwargs):
        """Derivative test for the mapping.

        This test validates the mapping by performing a convergence test.

        Parameters
        ----------
        m : (nP) numpy.ndarray
            Starting vector of model parameters for the derivative test
        num : int
            Number of iterations for the derivative test
        random_seed : None or :class:`~simpeg.typing.RandomSeed`, optional
            Random seed used for generating a random array for ``m`` if it's
            None. It can either be an int, a predefined Numpy random number
            generator, or any valid input to ``numpy.random.default_rng``.
        kwargs: dict
            Keyword arguments and associated values in the dictionary must
            match those used in :meth:`discretize.tests.check_derivative`

        Returns
        -------
        bool
            Returns ``True`` if the test passes
        """
        print("Testing {0!s}".format(str(self)))
        if m is None:
            rng = np.random.default_rng(seed=random_seed)
            m = rng.uniform(size=self.nP)
        if "plotIt" not in kwargs:
            kwargs["plotIt"] = False

        assert isinstance(
            self.nP, (int, np.integer)
        ), "nP must be an integer for {}".format(self.__class__.__name__)
        return check_derivative(
            lambda m: [self * m, self.deriv(m)], m, num=num, **kwargs
        )

    def _assertMatchesPair(self, pair):
        assert (
            isinstance(self, pair)
            or isinstance(self, ComboMap)
            and isinstance(self.maps[0], pair)
        ), "Mapping object must be an instance of a {0!s} class.".format(pair.__name__)

    def __mul__(self, val):
        if isinstance(val, IdentityMap):
            if (
                not (self.shape[1] == "*" or val.shape[0] == "*")
                and not self.shape[1] == val.shape[0]
            ):
                raise ValueError(
                    "Dimension mismatch in {0!s} and {1!s}.".format(str(self), str(val))
                )
            return ComboMap([self, val])

        elif isinstance(val, np.ndarray):
            if not self.shape[1] == "*" and not self.shape[1] == val.shape[0]:
                raise ValueError(
                    "Dimension mismatch in {0!s} and np.ndarray{1!s}.".format(
                        str(self), str(val.shape)
                    )
                )
            return self._transform(val)

        elif isinstance(val, Zero):
            return Zero()

        raise Exception(
            "Unrecognized data type to multiply. Try a map or a numpy.ndarray!"
            "You used a {} of type {}".format(val, type(val))
        )

    def dot(self, map1):
        r"""Multiply two mappings to create a :class:`simpeg.maps.ComboMap`.

        Let :math:`\mathbf{f}_1` and :math:`\mathbf{f}_2` represent two mapping functions.
        Where :math:`\mathbf{m}` represents a set of input model parameters,
        the ``dot`` method is used to create a combination mapping:

        .. math::
            u(\mathbf{m}) = f_2(f_1(\mathbf{m}))

        Where :math:`\mathbf{f_1} : M \rightarrow K_1` and acts on the
        model first, and :math:`\mathbf{f_2} : K_1 \rightarrow K_2`, the combination
        mapping :math:`\mathbf{u} : M \rightarrow K_2`.

        When using the **dot** method, the input argument *map1* represents the first
        mapping that is be applied and *self* represents the second mapping
        that is be applied. Therefore, the correct syntax for using this method is::

            self.dot(map1)


        Parameters
        ----------
        map1 :
            A SimPEG mapping object.

        Examples
        --------
        Here we create a combination mapping that 1) projects a single scalar to
        a vector space of length 5, then takes the natural exponent.

        >>> import numpy as np
        >>> from simpeg.maps import ExpMap, Projection

        >>> nP1 = 1
        >>> nP2 = 5
        >>> ind = np.zeros(nP1, dtype=int)

        >>> projection_map = Projection(nP1, ind)
        >>> projection_map.shape
        (5, 1)

        >>> exp_map = ExpMap(nP=5)
        >>> exp_map.shape
        (5, 5)

        >>> combo_map = exp_map.dot(projection_map)
        >>> combo_map.shape
        (5, 1)

        >>> m = np.array([2])
        >>> combo_map * m
        array([7.3890561, 7.3890561, 7.3890561, 7.3890561, 7.3890561])

        """
        return self.__mul__(map1)

    def __matmul__(self, map1):
        return self.__mul__(map1)

    __numpy_ufunc__ = True

    def __add__(self, map1):
        return SumMap([self, map1])  # error-checking done inside of the SumMap

    def __str__(self):
        return "{0!s}({1!s},{2!s})".format(
            self.__class__.__name__, self.shape[0], self.shape[1]
        )

    def __len__(self):
        return 1

    @property
    def mesh(self):
        """
        The mesh used for the mapping

        Returns
        -------
        discretize.base.BaseMesh or None
        """
        return self._mesh

    @mesh.setter
    def mesh(self, value):
        if value is not None:
            value = validate_type("mesh", value, discretize.base.BaseMesh, cast=False)
        self._mesh = value

    @property
    def is_linear(self):
        """Determine whether or not this mapping is a linear operation.

        Returns
        -------
        bool
        """
        return True


class ComboMap(IdentityMap):
    r"""Combination mapping constructed by joining a set of other mappings.

    A ``ComboMap`` is a single mapping object made by joining a set
    of basic mapping operations by chaining them together, in order.
    When creating a ``ComboMap``, the user provides a list of SimPEG mapping objects they wish to join.
    The order of the mappings in this list is from last to first; i.e.
    :math:`[\mathbf{f}_n , ... , \mathbf{f}_2 , \mathbf{f}_1]`.

    The combination mapping :math:`\mathbf{u}(\mathbf{m})` that acts on a
    set of input model parameters :math:`\mathbf{m}` is defined as:

    .. math::
        \mathbf{u}(\mathbf{m}) = f_n(f_{n-1}(\cdots f_1(f_0(\mathbf{m}))))

    Note that any time that you create your own combination mapping,
    be sure to test that the derivative is correct.

    Parameters
    ----------
    maps : list of simpeg.maps.IdentityMap
        A ``list`` of SimPEG mapping objects. The ordering of the mapping
        objects in the ``list`` is from last applied to first applied!

    Examples
    --------
    Here we create a combination mapping that 1) projects a single scalar to
    a vector space of length 5, then takes the natural exponent.

    >>> import numpy as np
    >>> from simpeg.maps import ExpMap, Projection, ComboMap

    >>> nP1 = 1
    >>> nP2 = 5
    >>> ind = np.zeros(nP1, dtype=int)

    >>> projection_map = Projection(nP1, ind)
    >>> projection_map.shape
    (5, 1)

    >>> exp_map = ExpMap(nP=5)
    >>> exp_map.shape
    (5, 5)

    Recall that the order of the mapping objects is from last applied
    to first applied.

    >>> map_list = [exp_map, projection_map]
    >>> combo_map = ComboMap(map_list)
    >>> combo_map.shape
    (5, 1)

    >>> m = np.array([2.])
    >>> combo_map * m
    array([7.3890561, 7.3890561, 7.3890561, 7.3890561, 7.3890561])

    """

    def __init__(self, maps, **kwargs):
        super().__init__(mesh=None, **kwargs)

        self.maps = []
        for ii, m in enumerate(maps):
            assert isinstance(m, IdentityMap), "Unrecognized data type, "
            "inherit from an IdentityMap or ComboMap!"

            if (
                ii > 0
                and not (self.shape[1] == "*" or m.shape[0] == "*")
                and not self.shape[1] == m.shape[0]
            ):
                prev = self.maps[-1]

                raise ValueError(
                    "Dimension mismatch in map[{0!s}] ({1!s}, {2!s}) "
                    "and map[{3!s}] ({4!s}, {5!s}).".format(
                        prev.__class__.__name__,
                        prev.shape[0],
                        prev.shape[1],
                        m.__class__.__name__,
                        m.shape[0],
                        m.shape[1],
                    )
                )

            if np.any([isinstance(m, SumMap), isinstance(m, IdentityMap)]):
                self.maps += [m]
            elif isinstance(m, ComboMap):
                self.maps += m.maps
            else:
                raise ValueError("Map[{0!s}] not supported", m.__class__.__name__)

    @property
    def shape(self):
        r"""Dimensions of the mapping.

        For a list of SimPEG mappings [:math:`\mathbf{f}_n,...,\mathbf{f}_1`]
        that have been joined to create a ``ComboMap``, this method returns
        the dimensions of the combination mapping. Recall that the ordering
        of the list of mappings is from last to first.

        Returns
        -------
        (2) tuple of int
            Dimensions of the mapping operator.
        """
        return (self.maps[0].shape[0], self.maps[-1].shape[1])

    @property
    def nP(self):
        r"""Number of parameters the mapping acts on.

        Returns
        -------
        int
            Number of parameters that the mapping acts on.
        """
        return self.maps[-1].nP

    def _transform(self, m):
        for map_i in reversed(self.maps):
            m = map_i * m
        return m

    def deriv(self, m, v=None):
        r"""Derivative of the mapping with respect to the input parameters.

        Any time that you create your own combination mapping,
        be sure to test that the derivative is correct.

        Parameters
        ----------
        m : (nP) numpy.ndarray
            A vector representing a set of model parameters
        v : (nP) numpy.ndarray
            If not ``None``, the method returns the derivative times the vector *v*

        Returns
        -------
        scipy.sparse.csr_matrix
            Derivative of the mapping with respect to the model parameters.
            If the input argument *v* is not ``None``, the method returns
            the derivative times the vector *v*.

        Notes
        -----
        Let :math:`\mathbf{m}` be a set of model parameters and let
        [:math:`\mathbf{f}_n,...,\mathbf{f}_1`] be the list of SimPEG mappings joined
        to create a combination mapping. Recall that the list of mappings is ordered
        from last applied to first applied.

        Where the combination mapping acting on the model parameters
        can be expressed as:

        .. math::
            \mathbf{u}(\mathbf{m}) = f_n(f_{n-1}(\cdots f_1(f_0(\mathbf{m}))))

        The **deriv** method returns the derivative of :math:`\mathbf{u}` with respect
        to the model parameters. To do this, we use the chain rule, i.e.:

        .. math::
            \frac{\partial \mathbf{u}}{\partial \mathbf{m}} =
            \frac{\partial \mathbf{f_n}}{\partial \mathbf{f_{n-1}}}
            \cdots
            \frac{\partial \mathbf{f_2}}{\partial \mathbf{f_{1}}}
            \frac{\partial \mathbf{f_1}}{\partial \mathbf{m}}
        """

        if v is not None:
            deriv = v
        else:
            deriv = 1

        mi = m
        for map_i in reversed(self.maps):
            deriv = map_i.deriv(mi) * deriv
            mi = map_i * mi
        return deriv

    def __str__(self):
        return "ComboMap[{0!s}]({1!s},{2!s})".format(
            " * ".join([m.__str__() for m in self.maps]), self.shape[0], self.shape[1]
        )

    def __len__(self):
        return len(self.maps)

    @property
    def is_linear(self):
        return all(m.is_linear for m in self.maps)


class LinearMap(IdentityMap):
    """A generalized linear mapping.

    A simple map that implements the linear mapping,

    >>> y = A @ x + b

    Parameters
    ----------
    A : (M, N) array_like, optional
        The matrix operator, can be any object that implements `__matmul__`
        and has a `shape` attribute.
    b : (M) array_like, optional
        Additive part of the linear operation.
    """

    def __init__(self, A, b=None, **kwargs):
        kwargs.pop("mesh", None)
        kwargs.pop("nP", None)
        super().__init__(**kwargs)
        self.A = A
        self.b = b

    @property
    def A(self):
        """The linear operator matrix.

        Returns
        -------
        LinearOperator
            Must support matrix multiplication and have a shape attribute.
        """
        return self._A

    @A.setter
    def A(self, value):
        if not hasattr(value, "__matmul__"):
            raise TypeError(
                f"{repr(value)} does not implement the matrix multiplication operator."
            )
        if not hasattr(value, "shape"):
            raise TypeError(f"{repr(value)} does not have a shape attribute.")
        self._A = value
        self._nP = value.shape[1]
        self._shape = value.shape

    @property
    def shape(self):
        return self._shape

    @property
    def b(self):
        """Added part of the linear operation.

        Returns
        -------
        numpy.ndarray
        """
        return self._b

    @b.setter
    def b(self, value):
        if value is not None:
            value = validate_ndarray_with_shape("b", value, shape=(self.shape[0],))
        self._b = value

    def _transform(self, m):
        if self.b is None:
            return self.A @ m
        return self.A @ m + self.b

    def deriv(self, m, v=None):
        if v is None:
            return self.A
        return self.A @ v


class Projection(IdentityMap):
    r"""Projection mapping.

    ``Projection`` mapping can be used to project and/or rearange model
    parameters. For a set of model parameter :math:`\mathbf{m}`,
    the mapping :math:`\mathbf{u}(\mathbf{m})` can be defined by a linear
    projection matrix :math:`\mathbf{P}` acting on the model, i.e.:

    .. math::
        \mathbf{u}(\mathbf{m}) = \mathbf{Pm}

    The number of model parameters the mapping acts on is
    defined by *nP*. Projection and/or rearrangement of the parameters
    is defined by *index*. Thus the dimensions of the mapping is
    (*nInd*, *nP*).

    Parameters
    ----------
    nP : int
        Number of model parameters the mapping acts on
    index : numpy.ndarray of int
        Indexes defining the projection from the model space

    Examples
    --------
    Here we define a mapping that rearranges and projects 2 model
    parameters to a vector space spanning 4 parameters.

    >>> from simpeg.maps import Projection
    >>> import numpy as np

    >>> nP = 2
    >>> index = np.array([1, 0, 1, 0], dtype=int)
    >>> mapping = Projection(nP, index)

    >>> m = np.array([6, 8])
    >>> mapping * m
    array([8, 6, 8, 6])

    """

    def __init__(self, nP, index, **kwargs):
        assert isinstance(
            index, (np.ndarray, slice, list)
        ), "index must be a np.ndarray or slice, not {}".format(type(index))
        super(Projection, self).__init__(nP=nP, **kwargs)

        if isinstance(index, slice):
            index = list(range(*index.indices(self.nP)))

        if isinstance(index, np.ndarray):
            if index.dtype is np.dtype("bool"):
                index = np.where(index)[0]

        self.index = index
        self._shape = nI, nP = len(self.index), self.nP

        assert max(index) < nP, "maximum index must be less than {}".format(nP)

        # sparse projection matrix
        self.P = sp.csr_matrix((np.ones(nI), (range(nI), self.index)), shape=(nI, nP))

    def _transform(self, m):
        return m[self.index]

    @property
    def shape(self):
        r"""Dimensions of the mapping.

        Returns
        -------
        tuple
            Where *nP* is the number of parameters the mapping acts on and
            *nInd* is the length of the vector defining the mapping, the
            dimensions of the mapping operator is a tuple of the
            form (*nInd*, *nP*).
        """
        return self._shape

    def deriv(self, m, v=None):
        r"""Derivative of the mapping with respect to the input parameters.

        Let :math:`\mathbf{m}` be a set of model parameters and let :math:`\mathbf{P}`
        be a matrix denoting the projection mapping. Where the projection mapping acting
        on the model parameters can be expressed as:

        .. math::
            \mathbf{u} = \mathbf{P m},

        the **deriv** method returns the derivative of :math:`\mathbf{u}` with respect
        to the model parameters; i.e.:

        .. math::
            \frac{\partial \mathbf{u}}{\partial \mathbf{m}} = \mathbf{P}

        Note that in this case, **deriv** simply returns a sparse projection matrix.

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
            return self.P * v
        return self.P


class SumMap(ComboMap):
    """Combination map constructed by summing multiple mappings
    to the same vector space.

    A map to add model parameters contributing to the
    forward operation e.g. F(m) = F(g(x) + h(y))

    Assumes that the model vectors defined by g(x) and h(y)
    are equal in length.
    Allows to assume different things about the model m:
    i.e. parametric + voxel models

    Parameters
    ----------
    maps : list
        A list of SimPEG mapping objects that are being summed.
        Each mapping object in the list must act on the same number
        of model parameters and must map to the same vector space!
    """

    def __init__(self, maps, **kwargs):
        maps = validate_list_of_types("maps", maps, IdentityMap)

        # skip ComboMap's init
        super(ComboMap, self).__init__(mesh=None, **kwargs)

        self.maps = []
        for ii, m in enumerate(maps):
            if not isinstance(m, IdentityMap):
                raise TypeError(
                    "Unrecognized data type {}, inherit from an "
                    "IdentityMap!".format(type(m))
                )

            if (
                ii > 0
                and not (self.shape == "*" or m.shape == "*")
                and not self.shape == m.shape
            ):
                raise ValueError(
                    "Dimension mismatch in map[{0!s}] ({1!s}, {2!s}) "
                    "and map[{3!s}] ({4!s}, {5!s}).".format(
                        self.maps[0].__class__.__name__,
                        self.maps[0].shape[0],
                        self.maps[0].shape[1],
                        m.__class__.__name__,
                        m.shape[0],
                        m.shape[1],
                    )
                )

            self.maps += [m]

    @property
    def shape(self):
        """Dimensions of the mapping.

        Returns
        -------
        tuple
            The dimensions of the mapping. A tuple of the form (``int``,``int``)
        """
        return (self.maps[0].shape[0], self.maps[0].shape[1])

    @property
    def nP(self):
        r"""Number of parameters the combined mapping acts on.

        Returns
        -------
        int
            Number of parameters that the mapping acts on.
        """
        return self.maps[-1].shape[1]

    def _transform(self, m):
        for ii, map_i in enumerate(self.maps):
            m0 = m.copy()
            m0 = map_i * m0

            if ii == 0:
                mout = m0
            else:
                mout += m0
        return mout

    def deriv(self, m, v=None):
        """Derivative of mapping with respect to the input parameters

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

        for ii, map_i in enumerate(self.maps):
            m0 = m.copy()

            if v is not None:
                deriv = v
            else:
                deriv = sp.eye(self.nP)

            deriv = map_i.deriv(m0, v=deriv)
            if ii == 0:
                sumDeriv = deriv
            else:
                sumDeriv += deriv

        return sumDeriv


class SphericalSystem(IdentityMap):
    r"""Mapping vectors from spherical to Cartesian coordinates.

    Let :math:`\mathbf{m}` be a model containing the amplitudes
    (:math:`\mathbf{a}`), azimuthal angles (:math:`\mathbf{t}`)
    and radial angles (:math:`\mathbf{p}`) for a set of vectors
    in spherical space such that:

    .. math::
        \mathbf{m} = \begin{bmatrix} \mathbf{a} \\ \mathbf{t} \\ \mathbf{p} \end{bmatrix}

    ``SphericalSystem`` constructs a mapping :math:`\mathbf{u}(\mathbf{m})
    that converts the set of vectors in spherical coordinates to
    their representation in Cartesian coordinates, i.e.:

    .. math::
        \mathbf{u}(\mathbf{m}) = \begin{bmatrix} \mathbf{v_x} \\ \mathbf{v_y} \\ \mathbf{v_z} \end{bmatrix}

    where :math:`\mathbf{v_x}`, :math:`\mathbf{v_y}` and :math:`\mathbf{v_z}`
    store the x, y and z components of the vectors, respectively.

    Using the *mesh* or *nP* input arguments, the dimensions of the corresponding
    mapping operator can be permanently set; i.e. (*3\*mesh.nC*, *3\*mesh.nC*) or (*nP*, *nP*).
    However if both input arguments *mesh* and *nP* are ``None``, the shape of
    mapping operator is arbitrary and can act on any vector whose length
    is a multiple of 3; i.e. has shape (``*``, ``*``).

    Notes
    -----

    In Cartesian space, the components of each vector are defined as

    .. math::
        \mathbf{v} = (v_x, v_y, v_z)

    In spherical coordinates, vectors are is defined as:

    .. math::
        \mathbf{v^\prime} = (a, t, p)

    where

        - :math:`a` is the amplitude of the vector
        - :math:`t` is the azimuthal angle defined positive from vertical
        - :math:`p` is the radial angle defined positive CCW from Easting

    Parameters
    ----------
    mesh : discretize.BaseMesh
        The number of parameters accepted by the mapping is set to equal
        *3\*mesh.nC* .
    nP : int
        Set the number of parameters accepted by the mapping directly. Used if the
        number of parameters is known. Used generally when the number of parameters
        is not equal to the number of cells in a mesh.
    """

    def __init__(self, mesh=None, nP=None, **kwargs):
        if nP is not None:
            assert nP % 3 == 0, "Number of parameters must be a multiple of 3"
        super().__init__(mesh, nP, **kwargs)
        self.model = None

    def sphericalDeriv(self, model):
        if getattr(self, "model", None) is None:
            self.model = model

        if getattr(self, "_sphericalDeriv", None) is None or not all(
            self.model == model
        ):
            self.model = model

            # Do a double projection to make sure the parameters are bounded
            m_xyz = mat_utils.spherical2cartesian(model.reshape((-1, 3), order="F"))
            m_atp = mat_utils.cartesian2spherical(
                m_xyz.reshape((-1, 3), order="F")
            ).reshape((-1, 3), order="F")

            nC = m_atp[:, 0].shape[0]

            dm_dx = sp.hstack(
                [
                    sp.diags(np.cos(m_atp[:, 1]) * np.cos(m_atp[:, 2]), 0),
                    sp.diags(
                        -m_atp[:, 0] * np.sin(m_atp[:, 1]) * np.cos(m_atp[:, 2]), 0
                    ),
                    sp.diags(
                        -m_atp[:, 0] * np.cos(m_atp[:, 1]) * np.sin(m_atp[:, 2]), 0
                    ),
                ]
            )

            dm_dy = sp.hstack(
                [
                    sp.diags(np.cos(m_atp[:, 1]) * np.sin(m_atp[:, 2]), 0),
                    sp.diags(
                        -m_atp[:, 0] * np.sin(m_atp[:, 1]) * np.sin(m_atp[:, 2]), 0
                    ),
                    sp.diags(
                        m_atp[:, 0] * np.cos(m_atp[:, 1]) * np.cos(m_atp[:, 2]), 0
                    ),
                ]
            )

            dm_dz = sp.hstack(
                [
                    sp.diags(np.sin(m_atp[:, 1]), 0),
                    sp.diags(m_atp[:, 0] * np.cos(m_atp[:, 1]), 0),
                    csr((nC, nC)),
                ]
            )

            self._sphericalDeriv = sp.vstack([dm_dx, dm_dy, dm_dz])

        return self._sphericalDeriv

    def _transform(self, model):
        return mat_utils.spherical2cartesian(model.reshape((-1, 3), order="F"))

    def inverse(self, u):
        r"""Maps vectors in Cartesian coordinates to spherical coordinates.

        Let :math:`\mathbf{v_x}`, :math:`\mathbf{v_y}` and :math:`\mathbf{v_z}`
        store the x, y and z components of a set of vectors in Cartesian
        coordinates such that:

        .. math::
            \mathbf{u} = \begin{bmatrix} \mathbf{x} \\ \mathbf{y} \\ \mathbf{z} \end{bmatrix}

        The inverse mapping recovers the vectors in spherical coordinates, i.e.:

        .. math::
            \mathbf{m}(\mathbf{u}) = \begin{bmatrix} \mathbf{a} \\ \mathbf{t} \\ \mathbf{p} \end{bmatrix}

        where :math:`\mathbf{a}` are the amplitudes, :math:`\mathbf{t}` are the
        azimuthal angles and :math:`\mathbf{p}` are the radial angles.

        Parameters
        ----------
        u : numpy.ndarray
            The x, y and z components of a set of vectors in Cartesian coordinates.
            If the mapping is defined for a mesh, the numpy.ndarray has length
            *3\*mesh.nC* .

        Returns
        -------
        numpy.ndarray
            The amplitudes (:math:`\mathbf{a}`), azimuthal angles (:math:`\mathbf{t}`)
            and radial angles (:math:`\mathbf{p}`) for the set of vectors in spherical
            coordinates. If the mapping is defined for a mesh, the numpy.ndarray has length
            *3\*mesh.nC* .
        """
        return mat_utils.cartesian2spherical(u.reshape((-1, 3), order="F"))

    @property
    def shape(self):
        r"""Dimensions of the mapping

        The dimensions of the mesh depend on the input arguments used
        during instantiation. If *mesh* is used to define the
        mapping, the shape of mapping operator is (*3\*mesh.nC*, *3\*mesh.nC*).
        If *nP* is used to define the identity map, the mapping operator
        has dimensions (*nP*, *nP*). If *mesh* and *nP* were ``None`` when
        instantiating, the mapping has dimensions (``*``, ``*``) and may
        act on a vector whose length is a multiple of 3.

        Returns
        -------
        tuple
            Dimensions of the mapping operator. If the dimensions of
            the mapping are set, the return is a tuple (``int``,``int``).
            If the mapping can act on a vector of arbitrary length, the
            return is a tuple (``*``, ``*``).
        """
        # return self.n_block*len(self.indices[0]), self.n_block*len(self.indices)
        return (self.nP, self.nP)

    def deriv(self, m, v=None):
        """Derivative of mapping with respect to the input parameters

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
            return self.sphericalDeriv(m) * v
        return self.sphericalDeriv(m)

    @property
    def is_linear(self):
        return False


class Wires(object):
    r"""Mapping class for organizing multiple parameter types into a single model.

    Let :math:`\mathbf{p_1}` and :math:`\mathbf{p_2}` be vectors that
    contain the parameter values for two different parameter types; for example,
    electrical conductivity and magnetic permeability. Here, all parameters
    are organized into a single model :math:`\mathbf{m}` of the form:

    .. math::
        \mathbf{m} = \begin{bmatrix} \mathbf{p_1} \\ \mathbf{p_2} \end{bmatrix}

    The ``Wires`` class constructs and applies the basic projection mappings
    for extracting the values of a particular parameter type from the model.
    For example:

    .. math::
        \mathbf{p_1} = \mathbf{P_{\! 1} m}

    where :math:`\mathbf{P_1}` is the projection matrix that extracts parameters
    :math:`\mathbf{p_1}` from the complete set of model parameters :math:`\mathbf{m}`.
    Likewise, there is a projection matrix for extracting :math:`\mathbf{p_2}`.
    This can be extended to a model that containing more than 2 parameter types.

    Parameters
    ----------
    args : tuple
        Each input argument is a tuple (``str``, ``int``) that provides the name
        and number of parameters for a given parameters type.

    Examples
    --------
    Here we construct a wire mapping for a model where there
    are two parameters types. Note that the number of parameters
    of each type does not need to be the same.

    >>> from simpeg.maps import Wires, ReciprocalMap
    >>> import numpy as np

    >>> p1 = np.r_[4.5, 2.7, 6.9, 7.1, 1.2]
    >>> p2 = np.r_[10., 2., 5.]**-1
    >>> nP1 = len(p1)
    >>> nP2 = len(p2)
    >>> m = np.r_[p1, p2]
    >>> m
    array([4.5, 2.7, 6.9, 7.1, 1.2, 0.1, 0.5, 0.2])

    Here we construct the wire map. The user provides a name
    and the number of parameters for each type. The name
    provided becomes the name of the method for constructing
    the projection mapping.

    >>> wire_map = Wires(('name_1', nP1), ('name_2', nP2))

    Here, we extract the values for the first parameter type.

    >>> wire_map.name_1 * m
    array([4.5, 2.7, 6.9, 7.1, 1.2])

    And here, we extract the values for the second parameter
    type then apply a reciprocal mapping.

    >>> reciprocal_map = ReciprocalMap()
    >>> reciprocal_map * wire_map.name_2 * m
    array([10.,  2.,  5.])

    """

    def __init__(self, *args):
        for arg in args:
            assert (
                isinstance(arg, tuple)
                and len(arg) == 2
                and isinstance(arg[0], str)
                and
                # TODO: this should be extended to a slice.
                isinstance(arg[1], (int, np.integer))
            ), (
                "Each wire needs to be a tuple: (name, length). "
                "You provided: {}".format(arg)
            )

        self._nP = int(np.sum([w[1] for w in args]))
        start = 0
        maps = []
        for arg in args:
            wire = Projection(self.nP, slice(start, start + arg[1]))
            setattr(self, arg[0], wire)
            maps += [(arg[0], wire)]
            start += arg[1]
        self.maps = maps

        self._tuple = namedtuple("Model", [w[0] for w in args])

    def __mul__(self, val):
        assert isinstance(val, np.ndarray)
        split = []
        for _, w in self.maps:
            split += [w * val]
        return self._tuple(*split)

    @property
    def nP(self):
        r"""Number of parameters the mapping acts on.

        Returns
        -------
        int
            Number of parameters that the mapping acts on.
        """
        return self._nP


class TileMap(IdentityMap):
    """
    Mapping for tiled inversion.

    Uses volume averaging to map a model defined on a global mesh to the
    local mesh. Everycell in the local mesh must also be in the global mesh.
    """

    def __init__(
        self,
        global_mesh,
        global_active,
        local_mesh,
        tol=1e-8,
        components=1,
        **kwargs,
    ):
        """
        Parameters
        ----------
        global_mesh : discretize.TreeMesh
            Global TreeMesh defining the entire domain.
        global_active : numpy.ndarray of bool or int
            Defines the active cells in the global mesh.
        local_mesh : discretize.TreeMesh
            Local TreeMesh for the simulation.
        tol : float, optional
            Tolerance to avoid zero division
        components : int, optional
            Number of components in the model. E.g. a vector model in 3D would have 3
            components.
        """
        super().__init__(mesh=None, **kwargs)
        self._global_mesh = validate_type(
            "global_mesh", global_mesh, discretize.TreeMesh, cast=False
        )
        self._local_mesh = validate_type(
            "local_mesh", local_mesh, discretize.TreeMesh, cast=False
        )

        self._global_active = validate_active_indices(
            "global_active", global_active, self.global_mesh.n_cells
        )

        self._tol = validate_float("tol", tol, min_val=0.0, inclusive_min=False)
        self._components = validate_integer("components", components, min_val=1)

        # trigger creation of P
        self.P

    @property
    def global_mesh(self):
        """Global TreeMesh defining the entire domain.

        Returns
        -------
        discretize.TreeMesh
        """
        return self._global_mesh

    @property
    def local_mesh(self):
        """Local TreeMesh defining the local domain.

        Returns
        -------
        discretize.TreeMesh
        """
        return self._local_mesh

    @property
    def global_active(self):
        """Defines the active cells in the global mesh.

        Returns
        -------
        (global_mesh.n_cells) numpy.ndarray of bool
        """
        return self._global_active

    @property
    def local_active(self):
        """
        This is the local_active of the global_active used in the global problem.

        Returns
        -------
        (local_mesh.n_cells) numpy.ndarray of bool
        """
        return self._local_active

    @property
    def tol(self):
        """Tolerance to avoid zero division.

        Returns
        -------
        float
        """
        return self._tol

    @property
    def components(self):
        """Number of components in the model.

        Returns
        -------
        int
        """
        return self._components

    @property
    def P(self):
        """
        Set the projection matrix with partial volumes
        """
        if getattr(self, "_P", None) is None:
            in_local = self.local_mesh._get_containing_cell_indexes(
                self.global_mesh.cell_centers
            )

            P = (
                sp.csr_matrix(
                    (
                        self.global_mesh.cell_volumes,
                        (in_local, np.arange(self.global_mesh.nC)),
                    ),
                    shape=(self.local_mesh.nC, self.global_mesh.nC),
                )
                * speye(self.global_mesh.nC)[:, self.global_active]
            )

            self._local_active = mkvc(np.sum(P, axis=1) > 0)

            P = P[self.local_active, :]

            self._P = sp.block_diag(
                [
                    sdiag(1.0 / self.local_mesh.cell_volumes[self.local_active]) * P
                    for ii in range(self.components)
                ]
            )

        return self._P

    def _transform(self, m):
        return self.P * m

    @property
    def shape(self):
        """
        Shape of the matrix operation (number of indices x nP)
        """
        return self.P.shape

    def deriv(self, m, v=None):
        """
        :param numpy.ndarray m: model
        :rtype: scipy.sparse.csr_matrix
        :return: derivative of transformed model
        """
        if v is not None:
            return self.P * v
        return self.P
