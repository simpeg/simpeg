from collections import namedtuple
import warnings
import discretize
import numpy as np
from numpy.polynomial import polynomial
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator
from scipy.interpolate import UnivariateSpline
from scipy.constants import mu_0
from scipy.sparse import csr_matrix as csr

from discretize.tests import check_derivative
from discretize import TensorMesh, CylindricalMesh
from discretize.utils import (
    mkvc,
    rotation_matrix_from_normals,
    Zero,
    Identity,
    sdiag,
    speye,
)

from .utils import (
    mat_utils,
    validate_type,
    validate_ndarray_with_shape,
    validate_float,
    validate_direction,
    validate_integer,
    validate_string,
    validate_active_indices,
    validate_list_of_types,
)


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

    def test(self, m=None, num=4, **kwargs):
        """Derivative test for the mapping.

        This test validates the mapping by performing a convergence test.

        Parameters
        ----------
        m : (nP) numpy.ndarray
            Starting vector of model parameters for the derivative test
        num : int
            Number of iterations for the derivative test
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
            m = abs(np.random.rand(self.nP))
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
        r"""Multiply two mappings to create a :class:`SimPEG.maps.ComboMap`.

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
        >>> from SimPEG.maps import ExpMap, Projection

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
    maps : list of SimPEG.maps.IdentityMap
        A ``list`` of SimPEG mapping objects. The ordering of the mapping
        objects in the ``list`` is from last applied to first applied!

    Examples
    --------
    Here we create a combination mapping that 1) projects a single scalar to
    a vector space of length 5, then takes the natural exponent.

    >>> import numpy as np
    >>> from SimPEG.maps import ExpMap, Projection, ComboMap

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

    >>> from SimPEG.maps import Projection
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


class SurjectUnits(IdentityMap):
    r"""Surjective mapping to all mesh cells.

    Let :math:`\mathbf{m}` be a model that contains a physical property value
    for *nP* geological units. ``SurjectUnits`` is used to construct a surjective
    mapping that projects :math:`\mathbf{m}` to the set of voxel cells defining a mesh.
    As a result, the mapping :math:`\mathbf{u(\mathbf{m})}` is defined as
    a projection matrix :math:`\mathbf{P}` acting on the model. Thus:

    .. math::
        \mathbf{u}(\mathbf{m}) = \mathbf{Pm}


    The mapping therefore has dimensions (*mesh.nC*, *nP*).

    Parameters
    ----------
    indices : (nP) list of (mesh.nC) numpy.ndarray
        Each entry in the :class:`list` is a boolean :class:`numpy.ndarray` of length
        *mesh.nC* that assigns the corresponding physical property value to the
        appropriate mesh cells.

    Examples
    --------
    For this example, we have a model that defines the property values
    for two units. Using ``SurjectUnit``, we construct the mapping from
    the model to a 1D mesh where the 1st unit's value is assigned to
    all cells whose centers are located at *x < 0* and the 2nd unit's value
    is assigned to all cells whose centers are located at *x > 0*.

    >>> from SimPEG.maps import SurjectUnits
    >>> from discretize import TensorMesh
    >>> import numpy as np

    >>> nP = 8
    >>> mesh = TensorMesh([np.ones(nP)], 'C')
    >>> unit_1_ind = mesh.cell_centers < 0

    >>> indices_list = [unit_1_ind, ~unit_1_ind]
    >>> mapping = SurjectUnits(indices_list, nP=nP)

    >>> m = np.r_[0.01, 0.05]
    >>> mapping * m
    array([0.01, 0.01, 0.01, 0.01, 0.05, 0.05, 0.05, 0.05])

    """

    def __init__(self, indices, **kwargs):
        super().__init__(**kwargs)
        self.indices = indices

    @property
    def indices(self):
        """List assigning a given physical property to specific model cells.

        Each entry in the :class:`list` is a boolean :class:`numpy.ndarray` of length
        *mesh.nC* that assigns the corresponding physical property value to the
        appropriate mesh cells.

        Returns
        -------
        (nP) list of (mesh.n_cells) numpy.ndarray
        """
        return self._indices

    @indices.setter
    def indices(self, values):
        values = validate_type("indices", values, list)
        mesh = self.mesh
        last_shape = None
        for i in range(len(values)):
            if mesh is not None:
                values[i] = validate_active_indices(
                    "indices", values[i], self.mesh.n_cells
                )
            else:
                values[i] = validate_ndarray_with_shape(
                    "indices", values[i], shape=("*",), dtype=int
                )
                if last_shape is not None and last_shape != values[i].shape:
                    raise ValueError("all indicies must have the same shape.")
                last_shape = values[i].shape
        self._indices = values

    @property
    def P(self):
        """
        Projection matrix from model parameters to mesh cells.
        """
        if getattr(self, "_P", None) is None:
            # sparse projection matrix
            row = []
            col = []
            val = []
            for ii, ind in enumerate(self.indices):
                col += [ii] * ind.sum()
                row += np.where(ind)[0].tolist()
                val += [1] * ind.sum()

            self._P = sp.csr_matrix(
                (val, (row, col)), shape=(len(self.indices[0]), self.nP)
            )

            # self._P = sp.block_diag([P for ii in range(self.nBlock)])

        return self._P

    def _transform(self, m):
        return self.P * m

    @property
    def nP(self):
        r"""Number of parameters the mapping acts on.

        Returns
        -------
        int
            Number of parameters that the mapping acts on.
        """
        return len(self.indices)

    @property
    def shape(self):
        """Dimensions of the mapping

        Returns
        -------
        tuple
            Dimensions of the mapping. Where *nP* is the number of parameters the
            mapping acts on and *mesh.nC* is the number of cells the corresponding
            mesh, the return is a tuple of the form (*mesh.nC*, *nP*).
        """
        # return self.n_block*len(self.indices[0]), self.n_block*len(self.indices)
        return (len(self.indices[0]), self.nP)

    def deriv(self, m, v=None):
        r"""Derivative of the mapping with respect to the input parameters.

        Let :math:`\mathbf{m}` be a set of model parameters. The surjective mapping
        can be defined as a sparse projection matrix :math:`\mathbf{P}`. Therefore
        we can define the surjective mapping acting on the model parameters as:

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
            Derivative of the mapping with respect to the model parameters.
            If the input argument *v* is not ``None``, the method returns
            the derivative times the vector *v*.
        """

        if v is not None:
            return self.P * v
        return self.P


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

    >>> from SimPEG.maps import Wires, ReciprocalMap
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
                1.0
                / 2.0
                * (1 + 1.0 / (alpha**2.0 - 1) * (1.0 - np.arctan(chi) / chi))
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
            warnings.warn("there are phis outside bounds of 0 and 1")
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
        warnings.warn("Maximum number of iterations reached")

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


###############################################################################
#                                                                             #
#                          Mesh Independent Maps                              #
#                                                                             #
###############################################################################


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

    >>> from SimPEG.maps import ComplexMap
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

        >>> from SimPEG.maps import ComplexMap
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

    # inverse = deriv


###############################################################################
#                                                                             #
#                 Surjection, Injection and Interpolation Maps                #
#                                                                             #
###############################################################################


class SurjectFull(IdentityMap):
    r"""Mapping a single property value to all mesh cells.

    Let :math:`m` be a model defined by a single physical property value
    ``SurjectFull`` construct a surjective mapping that projects :math:`m`
    to the set of voxel cells defining a mesh. The mapping
    :math:`\mathbf{u(m)}` is a matrix of 1s of shape (*mesh.nC* , 1) that
    projects the model to all mesh cells, i.e.:

    .. math::
        \mathbf{u}(\mathbf{m}) = \mathbf{Pm}

    Parameters
    ----------
    mesh : discretize.BaseMesh
        A discretize mesh

    """

    def __init__(self, mesh, **kwargs):
        super().__init__(mesh=mesh, **kwargs)

    @property
    def nP(self):
        r"""Number of parameters the mapping acts on; i.e. 1.

        Returns
        -------
        int
            Returns an integer value of 1
        """
        return 1

    def _transform(self, m):
        """
        :param m: model (scalar)
        :rtype: numpy.ndarray
        :return: transformed model
        """
        return np.ones(self.mesh.nC) * m

    def deriv(self, m, v=None):
        r"""Derivative of the mapping with respect to the input parameters.

        Let :math:`m` be the single parameter that the mapping acts on. The
        ``SurjectFull`` class constructs a mapping that can be defined as
        a projection matrix :math:`\mathbf{P}`; i.e.:

        .. math::
            \mathbf{u} = \mathbf{P m},

        the **deriv** method returns the derivative of :math:`\mathbf{u}` with respect
        to the model parameters; i.e.:

        .. math::
            \frac{\partial \mathbf{u}}{\partial \mathbf{m}} = \mathbf{P}

        Note that in this case, **deriv** simply returns the original operator
        :math:`\mathbf{P}`; a (*mesh.nC* , 1) numpy.ndarray of 1s.

        Parameters
        ----------
        m : (nP) numpy.ndarray
            A vector representing a set of model parameters
        v : (nP) numpy.ndarray
            If not ``None``, the method returns the derivative times the vector *v*
        """
        deriv = sp.csr_matrix(np.ones([self.mesh.nC, 1]))
        if v is not None:
            return deriv * v
        return deriv


class SurjectVertical1D(IdentityMap):
    r"""Map 1D layered Earth model to 2D or 3D tensor mesh.

    Let :math:`m` be a 1D model that defines the property values along
    the last dimension of a tensor mesh; i.e. the y-direction for 2D
    meshes and the z-direction for 3D meshes. ``SurjectVertical1D``
    construct a surjective mapping from the 1D model to all voxel cells
    in the 2D or 3D tensor mesh provided.

    Mathematically, the mapping :math:`\mathbf{u}(\mathbf{m})` can be
    represented by a projection matrix:

    .. math::
        \mathbf{u}(\mathbf{m}) = \mathbf{Pm}

    Parameters
    ----------
    mesh : discretize.TensorMesh
        A 2D or 3D tensor mesh

    Examples
    --------
    Here we define a 1D layered Earth model comprised of 3 layers
    on a 1D tensor mesh. We then use ``SurjectVertical1D`` to
    construct a mapping which projects the 1D model onto a 2D
    tensor mesh.

    >>> from SimPEG.maps import SurjectVertical1D
    >>> from SimPEG.utils import plot_1d_layer_model
    >>> from discretize import TensorMesh
    >>> import numpy as np
    >>> import matplotlib as mpl
    >>> import matplotlib.pyplot as plt

    >>> dh = np.ones(20)
    >>> mesh1D = TensorMesh([dh], 'C')
    >>> mesh2D = TensorMesh([dh, dh], 'CC')

    >>> m = np.zeros(mesh1D.nC)
    >>> m[mesh1D.cell_centers < 0] = 10.
    >>> m[mesh1D.cell_centers < -5] = 5.

    >>> fig1 = plt.figure(figsize=(5,5))
    >>> ax1 = fig1.add_subplot(111)
    >>> plot_1d_layer_model(
    >>>     mesh1D.h[0], np.flip(m), ax=ax1, z0=0,
    >>>     scale='linear', show_layers=True, plot_elevation=True
    >>> )
    >>> ax1.set_xlim([-0.1, 11])
    >>> ax1.set_title('1D Model')

    >>> mapping = SurjectVertical1D(mesh2D)
    >>> u = mapping * m

    >>> fig2 = plt.figure(figsize=(6, 5))
    >>> ax2a = fig2.add_axes([0.1, 0.15, 0.7, 0.8])
    >>> mesh2D.plot_image(u, ax=ax2a, grid=True)
    >>> ax2a.set_title('Projected to 2D Mesh')
    >>> ax2b = fig2.add_axes([0.83, 0.15, 0.05, 0.8])
    >>> norm = mpl.colors.Normalize(vmin=np.min(m), vmax=np.max(m))
    >>> cbar = mpl.colorbar.ColorbarBase(ax2b, norm=norm, orientation="vertical")

    """

    def __init__(self, mesh, **kwargs):
        assert isinstance(
            mesh, (TensorMesh, CylindricalMesh)
        ), "Only implemented for tensor meshes"
        super().__init__(mesh=mesh, **kwargs)

    @property
    def nP(self):
        r"""Number of parameters the mapping acts on.

        Returns
        -------
        int
            Number of parameters the mapping acts on. Should equal the
            number of cells along the last dimension of the tensor mesh
            supplied when defining the mapping.
        """
        return int(self.mesh.vnC[self.mesh.dim - 1])

    def _transform(self, m):
        repNum = np.prod(self.mesh.vnC[: self.mesh.dim - 1])
        return mkvc(m).repeat(repNum)

    def deriv(self, m, v=None):
        r"""Derivative of the mapping with respect to the model paramters.

        Let :math:`\mathbf{m}` be a set of parameter values for the 1D model
        and let :math:`\mathbf{P}` be a projection matrix that maps the 1D
        model the 2D/3D tensor mesh. The forward mapping :math:`\mathbf{u}(\mathbf{m})`
        is given by:

        .. math::
            \mathbf{u} = \mathbf{P m},

        the **deriv** method returns the derivative of :math:`\mathbf{u}` with respect
        to the model parameters; i.e.:

        .. math::
            \frac{\partial \mathbf{u}}{\partial \mathbf{m}} = \mathbf{P}

        Note that in this case, **deriv** simply returns the projection matrix.

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
        repNum = np.prod(self.mesh.vnC[: self.mesh.dim - 1])
        repVec = sp.csr_matrix(
            (np.ones(repNum), (range(repNum), np.zeros(repNum))), shape=(repNum, 1)
        )
        deriv = sp.kron(sp.identity(self.nP), repVec)
        if v is not None:
            return deriv * v
        return deriv


class Surject2Dto3D(IdentityMap):
    r"""Map 2D tensor model to 3D tensor mesh.

    Let :math:`m` define the parameters for a 2D tensor model.
    ``Surject2Dto3D`` constructs a surjective mapping that projects
    the 2D tensor model to a 3D tensor mesh.

    Mathematically, the mapping :math:`\mathbf{u}(\mathbf{m})` can be
    represented by a projection matrix:

    .. math::
        \mathbf{u}(\mathbf{m}) = \mathbf{Pm}

    Parameters
    ----------
    mesh : discretize.TensorMesh
        A 3D tensor mesh
    normal : {'y', 'x', 'z'}
        Define the projection axis.

    Examples
    --------
    Here we project a 3 layered Earth model defined on a 2D tensor mesh
    to a 3D tensor mesh. We assume that at for some y-location, we
    have a 2D tensor model which defines the physical property distribution
    as a function of the *x* and *z* location. Using ``Surject2Dto3D``,
    we project the model along the y-axis to obtain a 3D distribution
    for the physical property (i.e. a 3D tensor model).

    >>> from SimPEG.maps import Surject2Dto3D
    >>> from discretize import TensorMesh
    >>> import numpy as np
    >>> import matplotlib as mpl
    >>> import matplotlib.pyplot as plt

    >>> dh = np.ones(20)
    >>> mesh2D = TensorMesh([dh, dh], 'CC')
    >>> mesh3D = TensorMesh([dh, dh, dh], 'CCC')

    Here, we define the 2D tensor model.

    >>> m = np.zeros(mesh2D.nC)
    >>> m[mesh2D.cell_centers[:, 1] < 0] = 10.
    >>> m[mesh2D.cell_centers[:, 1] < -5] = 5.

    We then plot the 2D tensor model; which is defined along the
    x and z axes.

    >>> fig1 = plt.figure(figsize=(6, 5))
    >>> ax11 = fig1.add_axes([0.1, 0.15, 0.7, 0.8])
    >>> mesh2D.plot_image(m, ax=ax11, grid=True)
    >>> ax11.set_ylabel('z')
    >>> ax11.set_title('2D Tensor Model')
    >>> ax12 = fig1.add_axes([0.83, 0.15, 0.05, 0.8])
    >>> norm1 = mpl.colors.Normalize(vmin=np.min(m), vmax=np.max(m))
    >>> cbar1 = mpl.colorbar.ColorbarBase(ax12, norm=norm1, orientation="vertical")

    By setting *normal = 'Y'* we are projecting along the y-axis.

    >>> mapping = Surject2Dto3D(mesh3D, normal='Y')
    >>> u = mapping * m

    Finally we plot a slice of the resulting 3D tensor model.

    >>> fig2 = plt.figure(figsize=(6, 5))
    >>> ax21 = fig2.add_axes([0.1, 0.15, 0.7, 0.8])
    >>> mesh3D.plot_slice(u, ax=ax21, ind=10, normal='Y', grid=True)
    >>> ax21.set_ylabel('z')
    >>> ax21.set_title('Projected to 3D Mesh (y=0)')
    >>> ax22 = fig2.add_axes([0.83, 0.15, 0.05, 0.8])
    >>> norm2 = mpl.colors.Normalize(vmin=np.min(m), vmax=np.max(m))
    >>> cbar2 = mpl.colorbar.ColorbarBase(ax22, norm=norm2, orientation="vertical")

    """

    def __init__(self, mesh, normal="y", **kwargs):
        self.normal = normal
        super().__init__(mesh=mesh, **kwargs)

    @IdentityMap.mesh.setter
    def mesh(self, value):
        value = validate_type("mesh", value, discretize.TensorMesh, cast=False)
        if value.dim != 3:
            raise ValueError("Surject2Dto3D Only works for a 3D Mesh")
        self._mesh = value

    @property
    def normal(self):
        """The projection axis.

        Returns
        -------
        str
        """
        return self._normal

    @normal.setter
    def normal(self, value):
        self._normal = validate_string("normal", value, ("x", "y", "z"))

    @property
    def nP(self):
        """Number of model properties.

        The number of cells in the
        last dimension of the mesh."""
        if self.normal == "z":
            return self.mesh.shape_cells[0] * self.mesh.shape_cells[1]
        elif self.normal == "y":
            return self.mesh.shape_cells[0] * self.mesh.shape_cells[2]
        elif self.normal == "x":
            return self.mesh.shape_cells[1] * self.mesh.shape_cells[2]

    def _transform(self, m):
        m = mkvc(m)
        if self.normal == "z":
            return mkvc(
                m.reshape(self.mesh.vnC[:2], order="F")[:, :, np.newaxis].repeat(
                    self.mesh.shape_cells[2], axis=2
                )
            )
        elif self.normal == "y":
            return mkvc(
                m.reshape(self.mesh.vnC[::2], order="F")[:, np.newaxis, :].repeat(
                    self.mesh.shape_cells[1], axis=1
                )
            )
        elif self.normal == "x":
            return mkvc(
                m.reshape(self.mesh.vnC[1:], order="F")[np.newaxis, :, :].repeat(
                    self.mesh.shape_cells[0], axis=0
                )
            )

    def deriv(self, m, v=None):
        r"""Derivative of the mapping with respect to the model paramters.

        Let :math:`\mathbf{m}` be a set of parameter values for the 2D tensor model
        and let :math:`\mathbf{P}` be a projection matrix that maps the 2D tensor model
        to the 3D tensor mesh. The forward mapping :math:`\mathbf{u}(\mathbf{m})`
        is given by:

        .. math::
            \mathbf{u} = \mathbf{P m},

        the **deriv** method returns the derivative of :math:`\mathbf{u}` with respect
        to the model parameters; i.e.:

        .. math::
            \frac{\partial \mathbf{u}}{\partial \mathbf{m}} = \mathbf{P}

        Note that in this case, **deriv** simply returns the projection matrix.

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
        inds = self * np.arange(self.nP)
        nC, nP = self.mesh.nC, self.nP
        P = sp.csr_matrix((np.ones(nC), (range(nC), inds)), shape=(nC, nP))
        if v is not None:
            return P * v
        return P


class Mesh2Mesh(IdentityMap):
    """
    Takes a model on one mesh are translates it to another mesh.
    """

    def __init__(self, meshes, indActive=None, **kwargs):
        # Sanity checks for the meshes parameter
        try:
            mesh, mesh2 = meshes
        except TypeError:
            raise TypeError("Couldn't unpack 'meshes' into two meshes.")

        super().__init__(mesh=mesh, **kwargs)

        self.mesh2 = mesh2
        # Check dimensions of both meshes
        if mesh.dim != mesh2.dim:
            raise ValueError(
                f"Found meshes with dimensions '{mesh.dim}' and '{mesh2.dim}'. "
                + "Both meshes must have the same dimension."
            )
        self.indActive = indActive

    # reset to not accepted None for mesh
    @IdentityMap.mesh.setter
    def mesh(self, value):
        self._mesh = validate_type("mesh", value, discretize.base.BaseMesh, cast=False)

    @property
    def mesh2(self):
        """The source mesh used for the mapping.

        Returns
        -------
        discretize.base.BaseMesh
        """
        return self._mesh2

    @mesh2.setter
    def mesh2(self, value):
        self._mesh2 = validate_type(
            "mesh2", value, discretize.base.BaseMesh, cast=False
        )

    @property
    def indActive(self):
        """Active indices on target mesh.

        Returns
        -------
        (mesh.n_cells) numpy.ndarray of bool or none
        """
        return self._indActive

    @indActive.setter
    def indActive(self, value):
        if value is not None:
            value = validate_active_indices("indActive", value, self.mesh.n_cells)
        self._indActive = value

    @property
    def P(self):
        if getattr(self, "_P", None) is None:
            self._P = self.mesh2.get_interpolation_matrix(
                self.mesh.cell_centers[self.indActive, :]
                if self.indActive is not None
                else self.mesh.cell_centers,
                "CC",
                zeros_outside=True,
            )
        return self._P

    @property
    def shape(self):
        """Number of parameters in the model."""
        if self.indActive is not None:
            return (self.indActive.sum(), self.mesh2.nC)
        return (self.mesh.nC, self.mesh2.nC)

    @property
    def nP(self):
        """Number of parameters in the model."""
        return self.mesh2.nC

    def _transform(self, m):
        return self.P * m

    def deriv(self, m, v=None):
        if v is not None:
            return self.P * v
        return self.P


class InjectActiveCells(IdentityMap):
    r"""Map active cells model to all cell of a mesh.

    The ``InjectActiveCells`` class is used to define the mapping when
    the model consists of physical property values for a set of active
    mesh cells; e.g. cells below topography. For a discrete set of
    model parameters :math:`\mathbf{m}` defined on a set of active
    cells, the mapping :math:`\mathbf{u}(\mathbf{m})` is defined as:

    .. math::
        \mathbf{u}(\mathbf{m}) = \mathbf{Pm} + \mathbf{d}\, m_\perp

    where :math:`\mathbf{P}` is a (*nC* , *nP*) projection matrix from
    active cells to all mesh cells, and :math:`\mathbf{d}` is a
    (*nC* , 1) matrix that projects the inactive cell value
    :math:`m_\perp` to all inactive mesh cells.

    Parameters
    ----------
    mesh : discretize.BaseMesh
        A discretize mesh
    indActive : numpy.ndarray
        Active cells array. Can be a boolean ``numpy.ndarray`` of length *mesh.nC*
        or a ``numpy.ndarray`` of ``int`` containing the indices of the active cells.
    valInactive : float or numpy.ndarray
        The physical property value assigned to all inactive cells in the mesh

    """

    def __init__(self, mesh, indActive=None, valInactive=0.0, nC=None):
        self.mesh = mesh
        self.nC = nC or mesh.nC

        self._indActive = validate_active_indices("indActive", indActive, self.nC)
        self._nP = np.sum(self.indActive)

        self.P = sp.eye(self.nC, format="csr")[:, self.indActive]

        self.valInactive = valInactive

    @property
    def valInactive(self):
        """The physical property value assigned to all inactive cells in the mesh.

        Returns
        -------
        numpy.ndarray
        """
        return self._valInactive

    @valInactive.setter
    def valInactive(self, value):
        n_inactive = self.nC - self.nP
        try:
            value = validate_float("valInactive", value)
            value = np.full(n_inactive, value)
        except Exception:
            pass
        value = validate_ndarray_with_shape("valInactive", value, shape=(n_inactive,))

        self._valInactive = np.zeros(self.nC, dtype=float)
        self._valInactive[~self.indActive] = value

    @property
    def indActive(self):
        """

        Returns
        -------
        numpy.ndarray of bool

        """
        return self._indActive

    @property
    def shape(self):
        """Dimensions of the mapping

        Returns
        -------
        tuple of int
            Where *nP* is the number of active cells and *nC* is
            number of cell in the mesh, **shape** returns a
            tuple (*nC* , *nP*).
        """
        return (self.nC, self.nP)

    @property
    def nP(self):
        """Number of parameters the model acts on.

        Returns
        -------
        int
            Number of parameters the model acts on; i.e. the number of active cells
        """
        return int(self.indActive.sum())

    def _transform(self, m):
        if m.ndim > 1:
            return self.P * m + self.valInactive[:, None]
        return self.P * m + self.valInactive

    def inverse(self, u):
        r"""Recover the model parameters (active cells) from a set of physical
        property values defined on the entire mesh.

        For a discrete set of model parameters :math:`\mathbf{m}` defined
        on a set of active cells, the mapping :math:`\mathbf{u}(\mathbf{m})`
        is defined as:

        .. math::
            \mathbf{u}(\mathbf{m}) = \mathbf{Pm} + \mathbf{d} \,m_\perp

        where :math:`\mathbf{P}` is a (*nC* , *nP*) projection matrix from
        active cells to all mesh cells, and :math:`\mathbf{d}` is a
        (*nC* , 1) matrix that projects the inactive cell value
        :math:`m_\perp` to all inactive mesh cells.

        The inverse mapping is given by:

        .. math::
            \mathbf{m}(\mathbf{u}) = \mathbf{P^T u}

        Parameters
        ----------
        u : (mesh.nC) numpy.ndarray
            A vector which contains physical property values for all
            mesh cells.
        """
        return self.P.T * u

    def deriv(self, m, v=None):
        r"""Derivative of the mapping with respect to the input parameters.

        For a discrete set of model parameters :math:`\mathbf{m}` defined
        on a set of active cells, the mapping :math:`\mathbf{u}(\mathbf{m})`
        is defined as:

        .. math::
            \mathbf{u}(\mathbf{m}) = \mathbf{Pm} + \mathbf{d} \, m_\perp

        where :math:`\mathbf{P}` is a (*nC* , *nP*) projection matrix from
        active cells to all mesh cells, and :math:`\mathbf{d}` is a
        (*nC* , 1) matrix that projects the inactive cell value
        :math:`m_\perp` to all inactive mesh cells.

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


###############################################################################
#                                                                             #
#                             Parametric Maps                                 #
#                                                                             #
###############################################################################


class ParametricCircleMap(IdentityMap):
    r"""Mapping for a parameterized circle.

    Define the mapping from a parameterized model for a circle in a wholespace
    to all cells within a 2D mesh. For a circle within a wholespace, the
    model is defined by 5 parameters: the background physical property value
    (:math:`\sigma_0`), the physical property value for the circle
    (:math:`\sigma_c`), the x location :math:`x_0` and y location :math:`y_0`
    for center of the circle, and the circle's radius (:math:`R`).

    Let :math:`\mathbf{m} = [\sigma_0, \sigma_1, x_0, y_0, R]` be the set of
    model parameters the defines a circle within a wholespace. The mapping
    :math:`\mathbf{u}(\mathbf{m})` from the parameterized model to all cells
    within a 2D mesh is given by:

    .. math::

        \mathbf{u}(\mathbf{m}) = \sigma_0 + (\sigma_1 - \sigma_0)
        \bigg [ \frac{1}{2} + \pi^{-1} \arctan \bigg ( a \big [ \sqrt{(\mathbf{x_c}-x_0)^2 +
        (\mathbf{y_c}-y_0)^2} - R \big ] \bigg ) \bigg ]

    where :math:`\mathbf{x_c}` and :math:`\mathbf{y_c}` are vectors storing
    the x and y positions of all cell centers for the 2D mesh and :math:`a`
    is a user-defined constant which defines the sharpness of boundary of the
    circular structure.

    Parameters
    ----------
    mesh : discretize.BaseMesh
        A 2D discretize mesh
    logSigma : bool
        If ``True``, parameters :math:`\sigma_0` and :math:`\sigma_1` represent the
        natural log of the physical property values for the background and circle,
        respectively.
    slope : float
        A constant for defining the sharpness of the boundary between the circle
        and the wholespace. The sharpness increases as *slope* is increased.

    Examples
    --------
    Here we define the parameterized model for a circle in a wholespace. We then
    create and use a ``ParametricCircleMap`` to map the model to a 2D mesh.

    >>> from SimPEG.maps import ParametricCircleMap
    >>> from discretize import TensorMesh
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt

    >>> h = 0.5*np.ones(20)
    >>> mesh = TensorMesh([h, h])

    >>> sigma0, sigma1, x0, y0, R = 0., 10., 4., 6., 2.
    >>> model = np.r_[sigma0, sigma1, x0, y0, R]
    >>> mapping = ParametricCircleMap(mesh, logSigma=False, slope=2)

    >>> fig = plt.figure(figsize=(5, 5))
    >>> ax = fig.add_subplot(111)
    >>> mesh.plot_image(mapping * model, ax=ax)

    """

    def __init__(self, mesh, logSigma=True, slope=0.1):
        super().__init__(mesh=mesh)
        if mesh.dim != 2:
            raise NotImplementedError(
                "Mesh must be 2D, not implemented yet for other dimensions."
            )
        # TODO: this should be done through a composition with and ExpMap
        self.logSigma = logSigma
        self.slope = slope

    @property
    def slope(self):
        """Sharpness of the boundary.

        Larger number are sharper.

        Returns
        -------
        float
        """
        return self._slope

    @slope.setter
    def slope(self, value):
        self._slope = validate_float("slope", value, min_val=0.0, inclusive_min=False)

    @property
    def logSigma(self):
        """Whether the input needs to be transformed by an exponential

        Returns
        -------
        float
        """
        return self._logSigma

    @logSigma.setter
    def logSigma(self, value):
        self._logSigma = validate_type("logSigma", value, bool)

    @property
    def nP(self):
        r"""Number of parameters the mapping acts on; i.e. 5.

        Returns
        -------
        int
            The ``ParametricCircleMap`` acts on 5 parameters.
        """
        return 5

    def _transform(self, m):
        a = self.slope
        sig1, sig2, x, y, r = m[0], m[1], m[2], m[3], m[4]
        if self.logSigma:
            sig1, sig2 = np.exp(sig1), np.exp(sig2)
        X = self.mesh.cell_centers[:, 0]
        Y = self.mesh.cell_centers[:, 1]
        return sig1 + (sig2 - sig1) * (
            np.arctan(a * (np.sqrt((X - x) ** 2 + (Y - y) ** 2) - r)) / np.pi + 0.5
        )

    def deriv(self, m, v=None):
        r"""Derivative of the mapping with respect to the input parameters.

        Let :math:`\mathbf{m} = [\sigma_0, \sigma_1, x_0, y_0, R]` be the set of
        model parameters the defines a circle within a wholespace. The mapping
        :math:`\mathbf{u}(\mathbf{m})`from the parameterized model to all cells
        within a 2D mesh is given by:

        .. math::
            \mathbf{u}(\mathbf{m}) = \sigma_0 + (\sigma_1 - \sigma_0)
            \bigg [ \frac{1}{2} + \pi^{-1} \arctan \bigg ( a \big [ \sqrt{(\mathbf{x_c}-x_0)^2 +
            (\mathbf{y_c}-y_0)^2} - R \big ] \bigg ) \bigg ]

        The derivative of the mapping with respect to the model parameters is a
        ``numpy.ndarray`` of shape (*mesh.nC*, 5) given by:

        .. math::
            \frac{\partial \mathbf{u}}{\partial \mathbf{m}} =
            \Bigg [ \frac{\partial \mathbf{u}}{\partial \sigma_0} \;\;
            \Bigg [ \frac{\partial \mathbf{u}}{\partial \sigma_1} \;\;
            \Bigg [ \frac{\partial \mathbf{u}}{\partial x_0} \;\;
            \Bigg [ \frac{\partial \mathbf{u}}{\partial y_0} \;\;
            \Bigg [ \frac{\partial \mathbf{u}}{\partial R}
            \Bigg ]

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
        a = self.slope
        sig1, sig2, x, y, r = m[0], m[1], m[2], m[3], m[4]
        if self.logSigma:
            sig1, sig2 = np.exp(sig1), np.exp(sig2)
        X = self.mesh.cell_centers[:, 0]
        Y = self.mesh.cell_centers[:, 1]
        if self.logSigma:
            g1 = (
                -(
                    np.arctan(a * (-r + np.sqrt((X - x) ** 2 + (Y - y) ** 2))) / np.pi
                    + 0.5
                )
                * sig1
                + sig1
            )
            g2 = (
                np.arctan(a * (-r + np.sqrt((X - x) ** 2 + (Y - y) ** 2))) / np.pi + 0.5
            ) * sig2
        else:
            g1 = (
                -(
                    np.arctan(a * (-r + np.sqrt((X - x) ** 2 + (Y - y) ** 2))) / np.pi
                    + 0.5
                )
                + 1.0
            )
            g2 = (
                np.arctan(a * (-r + np.sqrt((X - x) ** 2 + (Y - y) ** 2))) / np.pi + 0.5
            )

        g3 = (
            a
            * (-X + x)
            * (-sig1 + sig2)
            / (
                np.pi
                * (a**2 * (-r + np.sqrt((X - x) ** 2 + (Y - y) ** 2)) ** 2 + 1)
                * np.sqrt((X - x) ** 2 + (Y - y) ** 2)
            )
        )

        g4 = (
            a
            * (-Y + y)
            * (-sig1 + sig2)
            / (
                np.pi
                * (a**2 * (-r + np.sqrt((X - x) ** 2 + (Y - y) ** 2)) ** 2 + 1)
                * np.sqrt((X - x) ** 2 + (Y - y) ** 2)
            )
        )

        g5 = (
            -a
            * (-sig1 + sig2)
            / (np.pi * (a**2 * (-r + np.sqrt((X - x) ** 2 + (Y - y) ** 2)) ** 2 + 1))
        )

        if v is not None:
            return sp.csr_matrix(np.c_[g1, g2, g3, g4, g5]) * v
        return sp.csr_matrix(np.c_[g1, g2, g3, g4, g5])

    @property
    def is_linear(self):
        return False


class ParametricPolyMap(IdentityMap):
    r"""Mapping for 2 layer model whose interface is defined by a polynomial.

    This mapping is used when the cells lying below the Earth's surface can
    be parameterized by a 2 layer model whose interface is defined by a
    polynomial function. The model is defined by the physical property
    values for each unit (:math:`\sigma_1` and :math:`\sigma_2`) and the
    coefficients for the polynomial function (:math:`\mathbf{c}`).

    **For a 2D mesh** , the interface is defined by a polynomial function
    of the form:

    .. math::
        p(x) = \sum_{i=0}^N c_i x^i

    where :math:`c_i` are the polynomial coefficients and :math:`N` is
    the order of the polynomial. In this case, the model is defined as

    .. math::
        \mathbf{m} = [\sigma_1, \;\sigma_2,\; c_0 ,\;\ldots\; ,\; c_N]

    The mapping :math:`\mathbf{u}(\mathbf{m})` from the model to the mesh
    is given by:

    .. math::

        \mathbf{u}(\mathbf{m}) = \sigma_1 + (\sigma_2 - \sigma_1)
        \bigg [ \frac{1}{2} + \pi^{-1} \arctan \bigg (
        a \Big ( \mathbf{p}(\mathbf{x_c}) - \mathbf{y_c} \Big )
        \bigg ) \bigg ]

    where :math:`\mathbf{x_c}` and :math:`\mathbf{y_c}` are vectors containing the
    x and y cell center locations for all active cells in the mesh, and :math:`a` is a
    parameter which defines the sharpness of the boundary between the two layers.
    :math:`\mathbf{p}(\mathbf{x_c})` evaluates the polynomial function for
    every element in :math:`\mathbf{x_c}`.

    **For a 3D mesh** , the interface is defined by a 2D polynomial function
    of the form:

    .. math::
        p(x,y) =
        \sum_{j=0}^{N_y} \sum_{i=0}^{N_x} c_{ij} \, x^i y^j

    where :math:`c_{ij}` are the polynomial coefficients. :math:`N_x`
    and :math:`N_y` define the order of the polynomial in :math:`x` and
    :math:`y`, respectively. In this case, the model is defined as:

    .. math::
        \mathbf{m} = [\sigma_1, \; \sigma_2, \; c_{0,0} , \; c_{1,0} , \;\ldots , \; c_{N_x, N_y}]

    The mapping :math:`\mathbf{u}(\mathbf{m})` from the model to the mesh
    is given by:

    .. math::

        \mathbf{u}(\mathbf{m}) = \sigma_1 + (\sigma_2 - \sigma_1)
        \bigg [ \frac{1}{2} + \pi^{-1} \arctan \bigg (
        a \Big ( \mathbf{p}(\mathbf{x_c,y_c}) - \mathbf{z_c} \Big )
        \bigg ) \bigg ]

    where :math:`\mathbf{x_c}, \mathbf{y_c}` and :math:`\mathbf{y_z}` are vectors
    containing the x, y and z cell center locations for all active cells in the mesh.
    :math:`\mathbf{p}(\mathbf{x_c, y_c})` evaluates the polynomial function for
    every corresponding pair of :math:`\mathbf{x_c}` and :math:`\mathbf{y_c}`
    elements.

    Parameters
    ----------
    mesh : discretize.BaseMesh
        A discretize mesh
    order : int or list of int
        Order of the polynomial. For a 2D mesh, this is an ``int``. For a 3D
        mesh, the order for both variables is entered separately; i.e.
        [*order1* , *order2*].
    logSigma : bool
        If ``True``, parameters :math:`\sigma_1` and :math:`\sigma_2` represent
        the natural log of a physical property.
    normal : {'x', 'y', 'z'}
    actInd : numpy.ndarray
        Active cells array. Can be a boolean ``numpy.ndarray`` of length *mesh.nC*
        or a ``numpy.ndarray`` of ``int`` containing the indices of the active cells.

    Examples
    --------
    In this example, we define a 2 layer model whose interface is sharp and lies
    along a polynomial function :math:`y(x)=c_0 + c_1 x`. In this case, the model is
    defined as :math:`\mathbf{m} = [\sigma_1 , \sigma_2 , c_0 , c_1]`. We construct
    a polynomial mapping from the model to the set of active cells (i.e. below the surface),
    We then use an active cells mapping to map from the set of active cells to all
    cells in the 2D mesh.

    >>> from SimPEG.maps import ParametricPolyMap, InjectActiveCells
    >>> from discretize import TensorMesh
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt

    >>> h = 0.5*np.ones(20)
    >>> mesh = TensorMesh([h, h])
    >>> ind_active = mesh.cell_centers[:, 1] < 8
    >>>
    >>> sig1, sig2, c0, c1 = 10., 5., 2., 0.5
    >>> model = np.r_[sig1, sig2, c0, c1]

    >>> poly_map = ParametricPolyMap(
    >>>     mesh, order=1, logSigma=False, normal='Y', actInd=ind_active, slope=1e4
    >>> )
    >>> act_map = InjectActiveCells(mesh, ind_active, 0.)

    >>> fig = plt.figure(figsize=(5, 5))
    >>> ax = fig.add_subplot(111)
    >>> mesh.plot_image(act_map * poly_map * model, ax=ax)
    >>> ax.set_title('Mapping on a 2D mesh')

    Here, we recreate the previous example on a 3D mesh but with a smoother interface.
    For a 3D mesh, the 2D polynomial defining the sloping interface is given by
    :math:`z(x,y) = c_0 + c_x x + c_y y + c_{xy} xy`. In this case, the model is
    defined as :math:`\mathbf{m} = [\sigma_1 , \sigma_2 , c_0 , c_x, c_y, c_{xy}]`.

    >>> mesh = TensorMesh([h, h, h])
    >>> ind_active = mesh.cell_centers[:, 2] < 8
    >>>
    >>> sig1, sig2, c0, cx, cy, cxy = 10., 5., 2., 0.5, 0., 0.
    >>> model = np.r_[sig1, sig2, c0, cx, cy, cxy]
    >>>
    >>> poly_map = ParametricPolyMap(
    >>>     mesh, order=[1, 1], logSigma=False, normal='Z', actInd=ind_active, slope=2
    >>> )
    >>> act_map = InjectActiveCells(mesh, ind_active, 0.)
    >>>
    >>> fig = plt.figure(figsize=(5, 5))
    >>> ax = fig.add_subplot(111)
    >>> mesh.plot_slice(act_map * poly_map * model, ax=ax, normal='Y', ind=10)
    >>> ax.set_title('Mapping on a 3D mesh')

    """

    def __init__(self, mesh, order, logSigma=True, normal="X", actInd=None, slope=1e4):
        super().__init__(mesh=mesh)
        self.logSigma = logSigma
        self.order = order
        self.normal = normal
        self.slope = slope

        if actInd is None:
            actInd = np.ones(mesh.n_cells, dtype=bool)
        self.actInd = actInd

    @property
    def slope(self):
        """Sharpness of the boundary.

        Larger number are sharper.

        Returns
        -------
        float
        """
        return self._slope

    @slope.setter
    def slope(self, value):
        self._slope = validate_float("slope", value, min_val=0.0, inclusive_min=False)

    @property
    def logSigma(self):
        """Whether the input needs to be transformed by an exponential

        Returns
        -------
        float
        """
        return self._logSigma

    @logSigma.setter
    def logSigma(self, value):
        self._logSigma = validate_type("logSigma", value, bool)

    @property
    def normal(self):
        """The projection axis.

        Returns
        -------
        str
        """
        return self._normal

    @normal.setter
    def normal(self, value):
        self._normal = validate_string("normal", value, ("x", "y", "z"))

    @property
    def actInd(self):
        """Active indices of the mesh.

        Returns
        -------
        (mesh.n_cells) numpy.ndarray of bool
        """
        return self._actInd

    @actInd.setter
    def actInd(self, value):
        self._actInd = validate_active_indices("actInd", value, self.mesh.n_cells)
        self._nC = sum(self._actInd)

    @property
    def shape(self):
        """Dimensions of the mapping.

        Returns
        -------
        tuple of int
            The dimensions of the mapping as a tuple of the form
            (*nC* , *nP*), where *nP* is the number of model parameters
            the mapping acts on and *nC* is the number of active cells
            being mapping to. If *actInd* is ``None``, then
            *nC = mesh.nC*.
        """
        return (self.nC, self.nP)

    @property
    def nC(self):
        """Number of active cells being mapped too.

        Returns
        -------
        int
        """
        return self._nC

    @property
    def nP(self):
        """Number of parameters the mapping acts on.

        Returns
        -------
        int
            The number of parameters the mapping acts on.
        """
        if np.isscalar(self.order):
            nP = self.order + 3
        else:
            nP = (self.order[0] + 1) * (self.order[1] + 1) + 2
        return nP

    def _transform(self, m):
        # Set model parameters
        alpha = self.slope
        sig1, sig2 = m[0], m[1]
        c = m[2:]
        if self.logSigma:
            sig1, sig2 = np.exp(sig1), np.exp(sig2)

        # 2D
        if self.mesh.dim == 2:
            X = self.mesh.cell_centers[self.actInd, 0]
            Y = self.mesh.cell_centers[self.actInd, 1]
            if self.normal == "x":
                f = polynomial.polyval(Y, c) - X
            elif self.normal == "y":
                f = polynomial.polyval(X, c) - Y
            else:
                raise (Exception("Input for normal = X or Y or Z"))

        # 3D
        elif self.mesh.dim == 3:
            X = self.mesh.cell_centers[self.actInd, 0]
            Y = self.mesh.cell_centers[self.actInd, 1]
            Z = self.mesh.cell_centers[self.actInd, 2]

            if self.normal == "x":
                f = (
                    polynomial.polyval2d(
                        Y,
                        Z,
                        c.reshape((self.order[0] + 1, self.order[1] + 1), order="F"),
                    )
                    - X
                )
            elif self.normal == "y":
                f = (
                    polynomial.polyval2d(
                        X,
                        Z,
                        c.reshape((self.order[0] + 1, self.order[1] + 1), order="F"),
                    )
                    - Y
                )
            elif self.normal == "z":
                f = (
                    polynomial.polyval2d(
                        X,
                        Y,
                        c.reshape((self.order[0] + 1, self.order[1] + 1), order="F"),
                    )
                    - Z
                )
            else:
                raise (Exception("Input for normal = X or Y or Z"))

        else:
            raise (Exception("Only supports 2D or 3D"))

        return sig1 + (sig2 - sig1) * (np.arctan(alpha * f) / np.pi + 0.5)

    def deriv(self, m, v=None):
        r"""Derivative of the mapping with respect to the model.

        For a model :math:`\mathbf{m} = [\sigma_1, \sigma_2, \mathbf{c}]`,
        the derivative of the mapping with respect to the model parameters is a
        ``numpy.ndarray`` of shape (*mesh.nC*, *nP*) of the form:

        .. math::
            \frac{\partial \mathbf{u}}{\partial \mathbf{m}} =
            \Bigg [ \frac{\partial \mathbf{u}}{\partial \sigma_0} \;\;
            \Bigg [ \frac{\partial \mathbf{u}}{\partial \sigma_1} \;\;
            \Bigg [ \frac{\partial \mathbf{u}}{\partial c_0} \;\;
            \Bigg [ \frac{\partial \mathbf{u}}{\partial c_1} \;\;
            \cdots \;\;
            \Bigg [ \frac{\partial \mathbf{u}}{\partial c_N}
            \Bigg ]

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
        alpha = self.slope
        sig1, sig2, c = m[0], m[1], m[2:]
        if self.logSigma:
            sig1, sig2 = np.exp(sig1), np.exp(sig2)

        # 2D
        if self.mesh.dim == 2:
            X = self.mesh.cell_centers[self.actInd, 0]
            Y = self.mesh.cell_centers[self.actInd, 1]

            if self.normal == "x":
                f = polynomial.polyval(Y, c) - X
                V = polynomial.polyvander(Y, len(c) - 1)
            elif self.normal == "y":
                f = polynomial.polyval(X, c) - Y
                V = polynomial.polyvander(X, len(c) - 1)
            else:
                raise (Exception("Input for normal = X or Y"))

        # 3D
        elif self.mesh.dim == 3:
            X = self.mesh.cell_centers[self.actInd, 0]
            Y = self.mesh.cell_centers[self.actInd, 1]
            Z = self.mesh.cell_centers[self.actInd, 2]

            if self.normal == "x":
                f = (
                    polynomial.polyval2d(
                        Y, Z, c.reshape((self.order[0] + 1, self.order[1] + 1))
                    )
                    - X
                )
                V = polynomial.polyvander2d(Y, Z, self.order)
            elif self.normal == "y":
                f = (
                    polynomial.polyval2d(
                        X, Z, c.reshape((self.order[0] + 1, self.order[1] + 1))
                    )
                    - Y
                )
                V = polynomial.polyvander2d(X, Z, self.order)
            elif self.normal == "z":
                f = (
                    polynomial.polyval2d(
                        X, Y, c.reshape((self.order[0] + 1, self.order[1] + 1))
                    )
                    - Z
                )
                V = polynomial.polyvander2d(X, Y, self.order)
            else:
                raise (Exception("Input for normal = X or Y or Z"))

        if self.logSigma:
            g1 = -(np.arctan(alpha * f) / np.pi + 0.5) * sig1 + sig1
            g2 = (np.arctan(alpha * f) / np.pi + 0.5) * sig2
        else:
            g1 = -(np.arctan(alpha * f) / np.pi + 0.5) + 1.0
            g2 = np.arctan(alpha * f) / np.pi + 0.5

        g3 = sdiag(alpha * (sig2 - sig1) / (1.0 + (alpha * f) ** 2) / np.pi) * V

        if v is not None:
            return sp.csr_matrix(np.c_[g1, g2, g3]) * v
        return sp.csr_matrix(np.c_[g1, g2, g3])

    @property
    def is_linear(self):
        return False


class ParametricSplineMap(IdentityMap):
    r"""Mapping to parameterize the boundary between two geological units using
    spline interpolation.

    .. math::

        g = f(x)-y

    Define the model as:

    .. math::

        m = [\sigma_1, \sigma_2, y]

    Parameters
    ----------
    mesh : discretize.BaseMesh
        A discretize mesh
    pts : (n) numpy.ndarray
        Points for the 1D spline tie points.
    ptsv : (2) array_like
        Points for linear interpolation between two splines in 3D.
    order : int
        Order of the spline mapping; e.g. 3 is cubic spline
    logSigma : bool
        If ``True``, :math:`\sigma_1` and :math:`\sigma_2` represent the natural
        log of some physical property value for each unit.
    normal : {'x', 'y', 'z'}
        Defines the general direction of the normal vector for the interface.
    slope : float
        Parameter for defining the sharpness of the boundary. The sharpness is increased
        if *slope* is large.

    Examples
    --------
    In this example, we define a 2 layered model with a sloping
    interface on a 2D mesh. The model consists of the physical
    property values for the layers and the known elevations
    for the interface at the horizontal positions supplied when
    creating the mapping.

    >>> from SimPEG.maps import ParametricSplineMap
    >>> from discretize import TensorMesh
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt

    >>> h = 0.5*np.ones(20)
    >>> mesh = TensorMesh([h, h])

    >>> x = np.linspace(0, 10, 6)
    >>> y = 0.5*x + 2.5

    >>> model = np.r_[10., 0., y]
    >>> mapping = ParametricSplineMap(mesh, x, order=2, normal='Y', slope=2)

    >>> fig = plt.figure(figsize=(5, 5))
    >>> ax = fig.add_subplot(111)
    >>> mesh.plot_image(mapping * model, ax=ax)

    """

    def __init__(
        self, mesh, pts, ptsv=None, order=3, logSigma=True, normal="x", slope=1e4
    ):
        super().__init__(mesh=mesh)
        self.slope = slope
        self.logSigma = logSigma
        self.normal = normal
        self.order = order
        self.pts = pts
        self.ptsv = ptsv
        self.spl = None

    @IdentityMap.mesh.setter
    def mesh(self, value):
        self._mesh = validate_type(
            "mesh", value, discretize.base.BaseTensorMesh, cast=False
        )

    @property
    def slope(self):
        """Sharpness of the boundary.

        Larger number are sharper.

        Returns
        -------
        float
        """
        return self._slope

    @slope.setter
    def slope(self, value):
        self._slope = validate_float("slope", value, min_val=0.0, inclusive_min=False)

    @property
    def logSigma(self):
        """Whether the input needs to be transformed by an exponential

        Returns
        -------
        float
        """
        return self._logSigma

    @logSigma.setter
    def logSigma(self, value):
        self._logSigma = validate_type("logSigma", value, bool)

    @property
    def normal(self):
        """The projection axis.

        Returns
        -------
        str
        """
        return self._normal

    @normal.setter
    def normal(self, value):
        self._normal = validate_string("normal", value, ("x", "y", "z"))

    @property
    def order(self):
        """Order of the spline mapping.

        Returns
        -------
        int
        """
        return self._order

    @order.setter
    def order(self, value):
        self._order = validate_integer("order", value, min_val=1)

    @property
    def pts(self):
        """Points for the spline.

        Returns
        -------
        numpy.ndarray
        """
        return self._pts

    @pts.setter
    def pts(self, value):
        self._pts = validate_ndarray_with_shape("pts", value, shape=("*"), dtype=float)

    @property
    def npts(self):
        """The number of points.

        Returns
        -------
        int
        """
        return self._pts.shape[0]

    @property
    def ptsv(self):
        """Bottom and top values for the 3D spline surface.

        In 3D, two splines are created and linearly interpolated between these two
        points.

        Returns
        -------
        (2) numpy.ndarray
        """
        return self._ptsv

    @ptsv.setter
    def ptsv(self, value):
        if value is not None:
            value = validate_ndarray_with_shape("ptsv", value, shape=(2,))
        self._ptsv = value

    @property
    def nP(self):
        r"""Number of parameters the mapping acts on

        Returns
        -------
        int
            Number of parameters the mapping acts on.
            - **2D mesh:** the mapping acts on *mesh.nC + 2* parameters
            - **3D mesh:** the mapping acts on *2\*mesh.nC + 2* parameters
        """
        if self.mesh.dim == 2:
            return np.size(self.pts) + 2
        elif self.mesh.dim == 3:
            return np.size(self.pts) * 2 + 2
        else:
            raise (Exception("Only supports 2D and 3D"))

    def _transform(self, m):
        # Set model parameters
        alpha = self.slope
        sig1, sig2 = m[0], m[1]
        c = m[2:]
        if self.logSigma:
            sig1, sig2 = np.exp(sig1), np.exp(sig2)
        # 2D
        if self.mesh.dim == 2:
            X = self.mesh.cell_centers[:, 0]
            Y = self.mesh.cell_centers[:, 1]
            self.spl = UnivariateSpline(self.pts, c, k=self.order, s=0)
            if self.normal == "x":
                f = self.spl(Y) - X
            elif self.normal == "y":
                f = self.spl(X) - Y
            else:
                raise (Exception("Input for normal = X or Y or Z"))

        # 3D:
        # Comments:
        # Make two spline functions and link them using linear interpolation.
        # This is not quite direct extension of 2D to 3D case
        # Using 2D interpolation  is possible

        elif self.mesh.dim == 3:
            X = self.mesh.cell_centers[:, 0]
            Y = self.mesh.cell_centers[:, 1]
            Z = self.mesh.cell_centers[:, 2]

            npts = np.size(self.pts)
            if np.mod(c.size, 2):
                raise (Exception("Put even points!"))

            self.spl = {
                "splb": UnivariateSpline(self.pts, c[:npts], k=self.order, s=0),
                "splt": UnivariateSpline(self.pts, c[npts:], k=self.order, s=0),
            }

            if self.normal == "x":
                zb = self.ptsv[0]
                zt = self.ptsv[1]
                flines = (self.spl["splt"](Y) - self.spl["splb"](Y)) * (Z - zb) / (
                    zt - zb
                ) + self.spl["splb"](Y)
                f = flines - X
            # elif self.normal =='Y':
            # elif self.normal =='Z':
            else:
                raise (Exception("Input for normal = X or Y or Z"))
        else:
            raise (Exception("Only supports 2D and 3D"))

        return sig1 + (sig2 - sig1) * (np.arctan(alpha * f) / np.pi + 0.5)

    def deriv(self, m, v=None):
        alpha = self.slope
        sig1, sig2, c = m[0], m[1], m[2:]
        if self.logSigma:
            sig1, sig2 = np.exp(sig1), np.exp(sig2)
        # 2D
        if self.mesh.dim == 2:
            X = self.mesh.cell_centers[:, 0]
            Y = self.mesh.cell_centers[:, 1]

            if self.normal == "x":
                f = self.spl(Y) - X
            elif self.normal == "y":
                f = self.spl(X) - Y
            else:
                raise (Exception("Input for normal = X or Y or Z"))
        # 3D
        elif self.mesh.dim == 3:
            X = self.mesh.cell_centers[:, 0]
            Y = self.mesh.cell_centers[:, 1]
            Z = self.mesh.cell_centers[:, 2]

            if self.normal == "x":
                zb = self.ptsv[0]
                zt = self.ptsv[1]
                flines = (self.spl["splt"](Y) - self.spl["splb"](Y)) * (Z - zb) / (
                    zt - zb
                ) + self.spl["splb"](Y)
                f = flines - X
            # elif self.normal =='Y':
            # elif self.normal =='Z':
            else:
                raise (Exception("Not Implemented for Y and Z, your turn :)"))

        if self.logSigma:
            g1 = -(np.arctan(alpha * f) / np.pi + 0.5) * sig1 + sig1
            g2 = (np.arctan(alpha * f) / np.pi + 0.5) * sig2
        else:
            g1 = -(np.arctan(alpha * f) / np.pi + 0.5) + 1.0
            g2 = np.arctan(alpha * f) / np.pi + 0.5

        if self.mesh.dim == 2:
            g3 = np.zeros((self.mesh.nC, self.npts))
            if self.normal == "y":
                # Here we use perturbation to compute sensitivity
                # TODO: bit more generalization of this ...
                # Modfications for X and Z directions ...
                for i in range(np.size(self.pts)):
                    ctemp = c[i]
                    ind = np.argmin(abs(self.mesh.cell_centers_y - ctemp))
                    ca = c.copy()
                    cb = c.copy()
                    dy = self.mesh.h[1][ind] * 1.5
                    ca[i] = ctemp + dy
                    cb[i] = ctemp - dy
                    spla = UnivariateSpline(self.pts, ca, k=self.order, s=0)
                    splb = UnivariateSpline(self.pts, cb, k=self.order, s=0)
                    fderiv = (spla(X) - splb(X)) / (2 * dy)
                    g3[:, i] = (
                        sdiag(alpha * (sig2 - sig1) / (1.0 + (alpha * f) ** 2) / np.pi)
                        * fderiv
                    )

        elif self.mesh.dim == 3:
            g3 = np.zeros((self.mesh.nC, self.npts * 2))
            if self.normal == "x":
                # Here we use perturbation to compute sensitivity
                for i in range(self.npts * 2):
                    ctemp = c[i]
                    ind = np.argmin(abs(self.mesh.cell_centers_y - ctemp))
                    ca = c.copy()
                    cb = c.copy()
                    dy = self.mesh.h[1][ind] * 1.5
                    ca[i] = ctemp + dy
                    cb[i] = ctemp - dy

                    # treat bottom boundary
                    if i < self.npts:
                        splba = UnivariateSpline(
                            self.pts, ca[: self.npts], k=self.order, s=0
                        )
                        splbb = UnivariateSpline(
                            self.pts, cb[: self.npts], k=self.order, s=0
                        )
                        flinesa = (
                            (self.spl["splt"](Y) - splba(Y)) * (Z - zb) / (zt - zb)
                            + splba(Y)
                            - X
                        )
                        flinesb = (
                            (self.spl["splt"](Y) - splbb(Y)) * (Z - zb) / (zt - zb)
                            + splbb(Y)
                            - X
                        )

                    # treat top boundary
                    else:
                        splta = UnivariateSpline(
                            self.pts, ca[self.npts :], k=self.order, s=0
                        )
                        spltb = UnivariateSpline(
                            self.pts, ca[self.npts :], k=self.order, s=0
                        )
                        flinesa = (
                            (self.spl["splt"](Y) - splta(Y)) * (Z - zb) / (zt - zb)
                            + splta(Y)
                            - X
                        )
                        flinesb = (
                            (self.spl["splt"](Y) - spltb(Y)) * (Z - zb) / (zt - zb)
                            + spltb(Y)
                            - X
                        )
                    fderiv = (flinesa - flinesb) / (2 * dy)
                    g3[:, i] = (
                        sdiag(alpha * (sig2 - sig1) / (1.0 + (alpha * f) ** 2) / np.pi)
                        * fderiv
                    )
        else:
            raise (Exception("Not Implemented for Y and Z, your turn :)"))

        if v is not None:
            return sp.csr_matrix(np.c_[g1, g2, g3]) * v
        return sp.csr_matrix(np.c_[g1, g2, g3])

    @property
    def is_linear(self):
        return False


class BaseParametric(IdentityMap):
    """Base class for parametric mappings from simple geological structures to meshes.

    Parameters
    ----------
    mesh : discretize.BaseMesh
        A discretize mesh
    indActive : numpy.ndarray, optional
        Active cells array. Can be a boolean ``numpy.ndarray`` of length *mesh.nC*
        or a ``numpy.ndarray`` of ``int`` containing the indices of the active cells.
    slope : float, optional
        Directly set the scaling parameter *slope* which sets the sharpness of boundaries
        between units.
    slopeFact : float, optional
        Set sharpness of boundaries between units based on minimum cell size. If set,
        the scalaing parameter *slope = slopeFact / dh*.

    """

    def __init__(self, mesh, slope=None, slopeFact=1.0, indActive=None, **kwargs):
        super(BaseParametric, self).__init__(mesh, **kwargs)
        self.indActive = indActive
        self.slopeFact = slopeFact
        if slope is not None:
            self.slope = slope

    @property
    def slope(self):
        """Defines the sharpness of the boundaries.

        Returns
        -------
        float
        """
        return self._slope

    @slope.setter
    def slope(self, value):
        self._slope = validate_float("slope", value, min_val=0.0)

    @property
    def slopeFact(self):
        """Defines the slope scaled by the mesh.

        Returns
        -------
        float
        """
        return self._slopeFact

    @slopeFact.setter
    def slopeFact(self, value):
        self._slopeFact = validate_float("slopeFact", value, min_val=0.0)
        self.slope = self._slopeFact / self.mesh.edge_lengths.min()

    @property
    def indActive(self):
        return self._indActive

    @indActive.setter
    def indActive(self, value):
        if value is not None:
            value = validate_active_indices("indActive", value, self.mesh.n_cells)
        self._indActive = value

    @property
    def x(self):
        """X cell center locations (active) for the output of the mapping.

        Returns
        -------
        (n_active) numpy.ndarray
            X cell center locations (active) for the output of the mapping.
        """
        if getattr(self, "_x", None) is None:
            if self.mesh.dim == 1:
                self._x = [
                    self.mesh.cell_centers
                    if self.indActive is None
                    else self.mesh.cell_centers[self.indActive]
                ][0]
            else:
                self._x = [
                    self.mesh.cell_centers[:, 0]
                    if self.indActive is None
                    else self.mesh.cell_centers[self.indActive, 0]
                ][0]
        return self._x

    @property
    def y(self):
        """Y cell center locations (active) for the output of the mapping.

        Returns
        -------
        (n_active) numpy.ndarray
            Y cell center locations (active) for the output of the mapping.
        """
        if getattr(self, "_y", None) is None:
            if self.mesh.dim > 1:
                self._y = [
                    self.mesh.cell_centers[:, 1]
                    if self.indActive is None
                    else self.mesh.cell_centers[self.indActive, 1]
                ][0]
            else:
                self._y = None
        return self._y

    @property
    def z(self):
        """Z cell center locations (active) for the output of the mapping.

        Returns
        -------
        (n_active) numpy.ndarray
            Z cell center locations (active) for the output of the mapping.
        """
        if getattr(self, "_z", None) is None:
            if self.mesh.dim > 2:
                self._z = [
                    self.mesh.cell_centers[:, 2]
                    if self.indActive is None
                    else self.mesh.cell_centers[self.indActive, 2]
                ][0]
            else:
                self._z = None
        return self._z

    def _atanfct(self, val, slope):
        return np.arctan(slope * val) / np.pi + 0.5

    def _atanfctDeriv(self, val, slope):
        # d/dx(atan(x)) = 1/(1+x**2)
        x = slope * val
        dx = -slope
        return (1.0 / (1 + x**2)) / np.pi * dx

    @property
    def is_linear(self):
        return False


class ParametricLayer(BaseParametric):
    r"""Mapping for a horizontal layer within a wholespace.

    This mapping is used when the cells lying below the Earth's surface can
    be parameterized by horizontal layer within a homogeneous medium.
    The model is defined by the physical property value for the background
    (:math:`\sigma_0`), the physical property value for the layer
    (:math:`\sigma_1`), the elevation for the middle of the layer (:math:`z_L`)
    and the thickness of the layer :math:`h`.

    For this mapping, the set of input model parameters are organized:

    .. math::
        \mathbf{m} = [\sigma_0, \;\sigma_1,\; z_L , \; h]

    The mapping :math:`\mathbf{u}(\mathbf{m})` from the model to the mesh
    is given by:

    .. math::

        \mathbf{u}(\mathbf{m}) = \sigma_0 + \frac{(\sigma_1 - \sigma_0)}{\pi} \Bigg [
        \arctan \Bigg ( a \bigg ( \mathbf{z_c} - z_L + \frac{h}{2} \bigg ) \Bigg )
        - \arctan \Bigg ( a \bigg ( \mathbf{z_c} - z_L - \frac{h}{2} \bigg ) \Bigg ) \Bigg ]

    where :math:`\mathbf{z_c}` is a vectors containing the vertical cell center
    locations for all active cells in the mesh, and :math:`a` is a
    parameter which defines the sharpness of the boundaries between the layer
    and the background.

    Parameters
    ----------
    mesh : discretize.BaseMesh
        A discretize mesh
    indActive : numpy.ndarray
        Active cells array. Can be a boolean ``numpy.ndarray`` of length *mesh.nC*
        or a ``numpy.ndarray`` of ``int`` containing the indices of the active cells.
    slope : float
        Directly define the constant *a* in the mapping function which defines the
        sharpness of the boundaries.
    slopeFact : float
        Scaling factor for the sharpness of the boundaries based on cell size.
        Using this option, we set *a = slopeFact / dh*.

    Examples
    --------
    In this example, we define a layer in a wholespace whose interface is sharp.
    We construct the mapping from the model to the set of active cells
    (i.e. below the surface), We then use an active cells mapping to map from
    the set of active cells to all cells in the mesh.

    >>> from SimPEG.maps import ParametricLayer, InjectActiveCells
    >>> from discretize import TensorMesh
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt

    >>> dh = 0.25*np.ones(40)
    >>> mesh = TensorMesh([dh, dh])
    >>> ind_active = mesh.cell_centers[:, 1] < 8

    >>> sig0, sig1, zL, h = 5., 10., 4., 2
    >>> model = np.r_[sig0, sig1, zL, h]

    >>> layer_map = ParametricLayer(
    >>>     mesh, indActive=ind_active, slope=4
    >>> )
    >>> act_map = InjectActiveCells(mesh, ind_active, 0.)

    >>> fig = plt.figure(figsize=(5, 5))
    >>> ax = fig.add_subplot(111)
    >>> mesh.plot_image(act_map * layer_map * model, ax=ax)

    """

    def __init__(self, mesh, **kwargs):
        super().__init__(mesh, **kwargs)

    @property
    def nP(self):
        """Number of model parameters the mapping acts on; i.e 4

        Returns
        -------
        int
            Returns an integer value of *4*.
        """
        return 4

    @property
    def shape(self):
        """Dimensions of the mapping

        Returns
        -------
        tuple of int
            Where *nP=4* is the number of parameters the mapping acts on
            and *nAct* is the number of active cells in the mesh, **shape**
            returns a tuple (*nAct* , *4*).
        """
        if self.indActive is not None:
            return (sum(self.indActive), self.nP)
        return (self.mesh.nC, self.nP)

    def mDict(self, m):
        r"""Return model parameters as a dictionary.

        For a model :math:`\mathbf{m} = [\sigma_0, \;\sigma_1,\; z_L , \; h]`,
        **mDict** returns a dictionary::

            {"val_background": m[0], "val_layer": m[1], "layer_center": m[2], "layer_thickness": m[3]}

        Returns
        -------
        dict
            The model as a dictionary
        """
        return {
            "val_background": m[0],
            "val_layer": m[1],
            "layer_center": m[2],
            "layer_thickness": m[3],
        }

    def _atanLayer(self, mDict):
        if self.mesh.dim == 2:
            z = self.y
        elif self.mesh.dim == 3:
            z = self.z

        layer_bottom = mDict["layer_center"] - mDict["layer_thickness"] / 2.0
        layer_top = mDict["layer_center"] + mDict["layer_thickness"] / 2.0

        return self._atanfct(z - layer_bottom, self.slope) * self._atanfct(
            z - layer_top, -self.slope
        )

    def _atanLayerDeriv_layer_center(self, mDict):
        if self.mesh.dim == 2:
            z = self.y
        elif self.mesh.dim == 3:
            z = self.z

        layer_bottom = mDict["layer_center"] - mDict["layer_thickness"] / 2.0
        layer_top = mDict["layer_center"] + mDict["layer_thickness"] / 2.0

        return self._atanfctDeriv(z - layer_bottom, self.slope) * self._atanfct(
            z - layer_top, -self.slope
        ) + self._atanfct(z - layer_bottom, self.slope) * self._atanfctDeriv(
            z - layer_top, -self.slope
        )

    def _atanLayerDeriv_layer_thickness(self, mDict):
        if self.mesh.dim == 2:
            z = self.y
        elif self.mesh.dim == 3:
            z = self.z

        layer_bottom = mDict["layer_center"] - mDict["layer_thickness"] / 2.0
        layer_top = mDict["layer_center"] + mDict["layer_thickness"] / 2.0

        return -0.5 * self._atanfctDeriv(z - layer_bottom, self.slope) * self._atanfct(
            z - layer_top, -self.slope
        ) + 0.5 * self._atanfct(z - layer_bottom, self.slope) * self._atanfctDeriv(
            z - layer_top, -self.slope
        )

    def layer_cont(self, mDict):
        return mDict["val_background"] + (
            mDict["val_layer"] - mDict["val_background"]
        ) * self._atanLayer(mDict)

    def _transform(self, m):
        mDict = self.mDict(m)
        return self.layer_cont(mDict)

    def _deriv_val_background(self, mDict):
        return np.ones_like(self.x) - self._atanLayer(mDict)

    def _deriv_val_layer(self, mDict):
        return self._atanLayer(mDict)

    def _deriv_layer_center(self, mDict):
        return (
            mDict["val_layer"] - mDict["val_background"]
        ) * self._atanLayerDeriv_layer_center(mDict)

    def _deriv_layer_thickness(self, mDict):
        return (
            mDict["val_layer"] - mDict["val_background"]
        ) * self._atanLayerDeriv_layer_thickness(mDict)

    def deriv(self, m):
        r"""Derivative of the mapping with respect to the input parameters.

        Let :math:`\mathbf{m} = [\sigma_0, \;\sigma_1,\; z_L , \; h]` be the set of
        model parameters the defines a layer within a wholespace. The mapping
        :math:`\mathbf{u}(\mathbf{m})`from the parameterized model to all
        active cells is given by:

        .. math::
            \mathbf{u}(\mathbf{m}) = \sigma_0 + \frac{(\sigma_1 - \sigma_0)}{\pi} \Bigg [
            \arctan \Bigg ( a \bigg ( \mathbf{z_c} - z_L + \frac{h}{2} \bigg ) \Bigg )
            - \arctan \Bigg ( a \bigg ( \mathbf{z_c} - z_L - \frac{h}{2} \bigg ) \Bigg ) \Bigg ]

        where :math:`\mathbf{z_c}` is a vectors containing the vertical cell center
        locations for all active cells in the mesh. The derivative of the mapping
        with respect to the model parameters is a ``numpy.ndarray`` of
        shape (*nAct*, *4*) given by:

        .. math::
            \frac{\partial \mathbf{u}}{\partial \mathbf{m}} =
            \Bigg [ \frac{\partial \mathbf{u}}{\partial \sigma_0} \;\;
            \frac{\partial \mathbf{u}}{\partial \sigma_1} \;\;
            \frac{\partial \mathbf{u}}{\partial z_L} \;\;
            \frac{\partial \mathbf{u}}{\partial h}
            \Bigg ]

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

        mDict = self.mDict(m)

        return sp.csr_matrix(
            np.vstack(
                [
                    self._deriv_val_background(mDict),
                    self._deriv_val_layer(mDict),
                    self._deriv_layer_center(mDict),
                    self._deriv_layer_thickness(mDict),
                ]
            ).T
        )


class ParametricBlock(BaseParametric):
    r"""Mapping for a rectangular block within a wholespace.

    This mapping is used when the cells lying below the Earth's surface can
    be parameterized by rectangular block within a homogeneous medium.
    The model is defined by the physical property value for the background
    (:math:`\sigma_0`), the physical property value for the block
    (:math:`\sigma_b`), parameters for the center of the block
    (:math:`x_b [,y_b, z_b]`) and parameters for the dimensions along
    each Cartesian direction (:math:`dx [,dy, dz]`)

    For this mapping, the set of input model parameters are organized:

    .. math::
        \mathbf{m} = \begin{cases}
        1D: \;\; [\sigma_0, \;\sigma_b,\; x_b , \; dx] \\
        2D: \;\; [\sigma_0, \;\sigma_b,\; x_b , \; dx,\; y_b , \; dy] \\
        3D: \;\; [\sigma_0, \;\sigma_b,\; x_b , \; dx,\; y_b , \; dy,\; z_b , \; dz]
        \end{cases}

    The mapping :math:`\mathbf{u}(\mathbf{m})` from the model to the mesh
    is given by:

    .. math::

        \mathbf{u}(\mathbf{m}) = \sigma_0 + (\sigma_b - \sigma_0) \bigg [ \frac{1}{2} +
        \pi^{-1} \arctan \bigg ( a \, \boldsymbol{\eta} \big (
        x_b, y_b, z_b, dx, dy, dz \big ) \bigg ) \bigg ]

    where *a* is a parameter that impacts the sharpness of the arctan function, and

    .. math::
        \boldsymbol{\eta} \big ( x_b, y_b, z_b, dx, dy, dz \big ) = 1 -
        \sum_{\xi \in (x,y,z)} \bigg [ \bigg ( \frac{2(\boldsymbol{\xi_c} - \xi_b)}{d\xi} \bigg )^2  + \varepsilon^2
        \bigg ]^{p/2}

    Parameters :math:`p` and :math:`\varepsilon` define the parameters of the Ekblom
    function. :math:`\boldsymbol{\xi_c}` is a place holder for vectors containing
    the x, [y and z] cell center locations of the mesh, :math:`\xi_b` is a placeholder
    for the x[, y and z] location for the center of the block, and :math:`d\xi` is a
    placeholder for the x[, y and z] dimensions of the block.

    Parameters
    ----------
    mesh : discretize.BaseMesh
        A discretize mesh
    indActive : numpy.ndarray
        Active cells array. Can be a boolean ``numpy.ndarray`` of length *mesh.nC*
        or a ``numpy.ndarray`` of ``int`` containing the indices of the active cells.
    slope : float
        Directly define the constant *a* in the mapping function which defines the
        sharpness of the boundaries.
    slopeFact : float
        Scaling factor for the sharpness of the boundaries based on cell size.
        Using this option, we set *a = slopeFact / dh*.
    epsilon : float
        Epsilon value used in the ekblom representation of the block
    p : float
        p-value used in the ekblom representation of the block.

    Examples
    --------
    In this example, we define a rectangular block in a wholespace whose
    interface is sharp. We construct the mapping from the model to the
    set of active cells (i.e. below the surface), We then use an active
    cells mapping to map from the set of active cells to all cells in the mesh.

    >>> from SimPEG.maps import ParametricBlock, InjectActiveCells
    >>> from discretize import TensorMesh
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt

    >>> dh = 0.5*np.ones(20)
    >>> mesh = TensorMesh([dh, dh])
    >>> ind_active = mesh.cell_centers[:, 1] < 8

    >>> sig0, sigb, xb, Lx, yb, Ly = 5., 10., 5., 4., 4., 2.
    >>> model = np.r_[sig0, sigb, xb, Lx, yb, Ly]

    >>> block_map = ParametricBlock(mesh, indActive=ind_active)
    >>> act_map = InjectActiveCells(mesh, ind_active, 0.)

    >>> fig = plt.figure(figsize=(5, 5))
    >>> ax = fig.add_subplot(111)
    >>> mesh.plot_image(act_map * block_map * model, ax=ax)

    """

    def __init__(self, mesh, epsilon=1e-6, p=10, **kwargs):
        self.epsilon = epsilon
        self.p = p
        super(ParametricBlock, self).__init__(mesh, **kwargs)

    @property
    def epsilon(self):
        """epsilon value used in the ekblom representation of the block.

        Returns
        -------
        float
        """
        return self._epsilon

    @epsilon.setter
    def epsilon(self, value):
        self._epsilon = validate_float("epsilon", value, min_val=0.0)

    @property
    def p(self):
        """p-value used in the ekblom representation of the block.

        Returns
        -------
        float
        """
        return self._p

    @p.setter
    def p(self, value):
        self._p = validate_float("p", value, min_val=0.0)

    @property
    def nP(self):
        """Number of parameters the mapping acts on.

        Returns
        -------
        int
            The number of the parameters defining the model depends on the dimension
            of the mesh. *nP*

            - =4 for a 1D mesh
            - =6 for a 2D mesh
            - =8 for a 3D mesh
        """
        if self.mesh.dim == 1:
            return 4
        if self.mesh.dim == 2:
            return 6
        elif self.mesh.dim == 3:
            return 8

    @property
    def shape(self):
        """Dimensions of the mapping

        Returns
        -------
        tuple of int
            Where *nP* is the number of parameters the mapping acts on
            and *nAct* is the number of active cells in the mesh, **shape**
            returns a tuple (*nAct* , *nP*).
        """
        if self.indActive is not None:
            return (sum(self.indActive), self.nP)
        return (self.mesh.nC, self.nP)

    def _mDict1d(self, m):
        return {
            "val_background": m[0],
            "val_block": m[1],
            "x0": m[2],
            "dx": m[3],
        }

    def _mDict2d(self, m):
        mDict = self._mDict1d(m)
        mDict.update(
            {
                # 'theta_x': m[4],
                "y0": m[4],
                "dy": m[5],
                # 'theta_y': m[7]
            }
        )
        return mDict

    def _mDict3d(self, m):
        mDict = self._mDict2d(m)
        mDict.update(
            {
                "z0": m[6],
                "dz": m[7],
                # 'theta_z': m[10]
            }
        )
        return mDict

    def mDict(self, m):
        r"""Return model parameters as a dictionary.

        Returns
        -------
        dict
            The model as a dictionary
        """
        return getattr(self, "_mDict{}d".format(self.mesh.dim))(m)

    def _ekblom(self, val):
        return (val**2 + self.epsilon**2) ** (self.p / 2.0)

    def _ekblomDeriv(self, val):
        return (
            (self.p / 2)
            * (val**2 + self.epsilon**2) ** ((self.p / 2) - 1)
            * 2
            * val
        )

    # def _rotation(self, mDict):
    #     if self.mesh.dim == 2:

    #     elif self.mesh.dim == 3:

    def _block1D(self, mDict):
        return 1 - (self._ekblom((self.x - mDict["x0"]) / (0.5 * mDict["dx"])))

    def _block2D(self, mDict):
        return 1 - (
            self._ekblom((self.x - mDict["x0"]) / (0.5 * mDict["dx"]))
            + self._ekblom((self.y - mDict["y0"]) / (0.5 * mDict["dy"]))
        )

    def _block3D(self, mDict):
        return 1 - (
            self._ekblom((self.x - mDict["x0"]) / (0.5 * mDict["dx"]))
            + self._ekblom((self.y - mDict["y0"]) / (0.5 * mDict["dy"]))
            + self._ekblom((self.z - mDict["z0"]) / (0.5 * mDict["dz"]))
        )

    def _transform(self, m):
        mDict = self.mDict(m)
        return mDict["val_background"] + (
            mDict["val_block"] - mDict["val_background"]
        ) * self._atanfct(
            getattr(self, "_block{}D".format(self.mesh.dim))(mDict), slope=self.slope
        )

    def _deriv_val_background(self, mDict):
        return 1 - self._atanfct(
            getattr(self, "_block{}D".format(self.mesh.dim))(mDict), slope=self.slope
        )

    def _deriv_val_block(self, mDict):
        return self._atanfct(
            getattr(self, "_block{}D".format(self.mesh.dim))(mDict), slope=self.slope
        )

    def _deriv_center_block(self, mDict, orientation):
        x = getattr(self, orientation)
        x0 = mDict["{}0".format(orientation)]
        dx = mDict["d{}".format(orientation)]
        return (mDict["val_block"] - mDict["val_background"]) * (
            self._atanfctDeriv(
                getattr(self, "_block{}D".format(self.mesh.dim))(mDict),
                slope=self.slope,
            )
            * (self._ekblomDeriv((x - x0) / (0.5 * dx)))
            / -(0.5 * dx)
        )

    def _deriv_width_block(self, mDict, orientation):
        x = getattr(self, orientation)
        x0 = mDict["{}0".format(orientation)]
        dx = mDict["d{}".format(orientation)]
        return (mDict["val_block"] - mDict["val_background"]) * (
            self._atanfctDeriv(
                getattr(self, "_block{}D".format(self.mesh.dim))(mDict),
                slope=self.slope,
            )
            * (self._ekblomDeriv((x - x0) / (0.5 * dx)) * (-(x - x0) / (0.5 * dx**2)))
        )

    def _deriv1D(self, mDict):
        return np.vstack(
            [
                self._deriv_val_background(mDict),
                self._deriv_val_block(mDict),
                self._deriv_center_block(mDict, "x"),
                self._deriv_width_block(mDict, "x"),
            ]
        ).T

    def _deriv2D(self, mDict):
        return np.vstack(
            [
                self._deriv_val_background(mDict),
                self._deriv_val_block(mDict),
                self._deriv_center_block(mDict, "x"),
                self._deriv_width_block(mDict, "x"),
                self._deriv_center_block(mDict, "y"),
                self._deriv_width_block(mDict, "y"),
            ]
        ).T

    def _deriv3D(self, mDict):
        return np.vstack(
            [
                self._deriv_val_background(mDict),
                self._deriv_val_block(mDict),
                self._deriv_center_block(mDict, "x"),
                self._deriv_width_block(mDict, "x"),
                self._deriv_center_block(mDict, "y"),
                self._deriv_width_block(mDict, "y"),
                self._deriv_center_block(mDict, "z"),
                self._deriv_width_block(mDict, "z"),
            ]
        ).T

    def deriv(self, m):
        r"""Derivative of the mapping with respect to the input parameters.

        Let :math:`\mathbf{m} = [\sigma_0, \;\sigma_1,\; x_b, \; dx, (\; y_b, \; dy, \; z_b , dz)]`
        be the set of model parameters the defines a block/ellipsoid within a wholespace.
        The mapping :math:`\mathbf{u}(\mathbf{m})` from the parameterized model to all
        active cells is given by:

        The derivative of the mapping :math:`\mathbf{u}(\mathbf{m})` with respect to
        the model parameters is a ``numpy.ndarray`` of shape (*nAct*, *nP*) given by:

        .. math::
            \frac{\partial \mathbf{u}}{\partial \mathbf{m}} = \Bigg [
            \frac{\partial \mathbf{u}}{\partial \sigma_0} \;\;
            \frac{\partial \mathbf{u}}{\partial \sigma_1} \;\;
            \frac{\partial \mathbf{u}}{\partial x_b} \;\;
            \frac{\partial \mathbf{u}}{\partial dx} \;\;
            \frac{\partial \mathbf{u}}{\partial y_b} \;\;
            \frac{\partial \mathbf{u}}{\partial dy} \;\;
            \frac{\partial \mathbf{u}}{\partial z_b} \;\;
            \frac{\partial \mathbf{u}}{\partial dz}
            \Bigg ) \Bigg ]

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
        return sp.csr_matrix(
            getattr(self, "_deriv{}D".format(self.mesh.dim))(self.mDict(m))
        )


class ParametricEllipsoid(ParametricBlock):
    r"""Mapping for a rectangular block within a wholespace.

    This mapping is used when the cells lying below the Earth's surface can
    be parameterized by an ellipsoid within a homogeneous medium.
    The model is defined by the physical property value for the background
    (:math:`\sigma_0`), the physical property value for the layer
    (:math:`\sigma_b`), parameters for the center of the ellipsoid
    (:math:`x_b [,y_b, z_b]`) and parameters for the dimensions along
    each Cartesian direction (:math:`dx [,dy, dz]`)

    For this mapping, the set of input model parameters are organized:

    .. math::
        \mathbf{m} = \begin{cases}
        1D: \;\; [\sigma_0, \;\sigma_b,\; x_b , \; dx] \\
        2D: \;\; [\sigma_0, \;\sigma_b,\; x_b , \; dx,\; y_b , \; dy] \\
        3D: \;\; [\sigma_0, \;\sigma_b,\; x_b , \; dx,\; y_b , \; dy,\; z_b , \; dz]
        \end{cases}

    The mapping :math:`\mathbf{u}(\mathbf{m})` from the model to the mesh
    is given by:

    .. math::

        \mathbf{u}(\mathbf{m}) = \sigma_0 + (\sigma_b - \sigma_0) \bigg [ \frac{1}{2} +
        \pi^{-1} \arctan \bigg ( a \, \boldsymbol{\eta} \big (
        x_b, y_b, z_b, dx, dy, dz \big ) \bigg ) \bigg ]

    where *a* is a parameter that impacts the sharpness of the arctan function, and

    .. math::
        \boldsymbol{\eta} \big ( x_b, y_b, z_b, dx, dy, dz \big ) = 1 -
        \sum_{\xi \in (x,y,z)} \bigg [ \bigg ( \frac{2(\boldsymbol{\xi_c} - \xi_b)}{d\xi} \bigg )^2  + \varepsilon^2
        \bigg ]

    :math:`\boldsymbol{\xi_c}` is a place holder for vectors containing
    the x, [y and z] cell center locations of the mesh, :math:`\xi_b` is a placeholder
    for the x[, y and z] location for the center of the block, and :math:`d\xi` is a
    placeholder for the x[, y and z] dimensions of the block.

    Parameters
    ----------
    mesh : discretize.BaseMesh
        A discretize mesh
    indActive : numpy.ndarray
        Active cells array. Can be a boolean ``numpy.ndarray`` of length *mesh.nC*
        or a ``numpy.ndarray`` of ``int`` containing the indices of the active cells.
    slope : float
        Directly define the constant *a* in the mapping function which defines the
        sharpness of the boundaries.
    slopeFact : float
        Scaling factor for the sharpness of the boundaries based on cell size.
        Using this option, we set *a = slopeFact / dh*.
    epsilon : float
        Epsilon value used in the ekblom representation of the block

    Examples
    --------
    In this example, we define an ellipse in a wholespace whose
    interface is sharp. We construct the mapping from the model to the
    set of active cells (i.e. below the surface), We then use an active
    cells mapping to map from the set of active cells to all cells in the mesh.

    >>> from SimPEG.maps import ParametricEllipsoid, InjectActiveCells
    >>> from discretize import TensorMesh
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt

    >>> dh = 0.5*np.ones(20)
    >>> mesh = TensorMesh([dh, dh])
    >>> ind_active = mesh.cell_centers[:, 1] < 8

    >>> sig0, sigb, xb, Lx, yb, Ly = 5., 10., 5., 4., 4., 3.
    >>> model = np.r_[sig0, sigb, xb, Lx, yb, Ly]

    >>> ellipsoid_map = ParametricEllipsoid(mesh, indActive=ind_active)
    >>> act_map = InjectActiveCells(mesh, ind_active, 0.)

    >>> fig = plt.figure(figsize=(5, 5))
    >>> ax = fig.add_subplot(111)
    >>> mesh.plot_image(act_map * ellipsoid_map * model, ax=ax)

    """

    def __init__(self, mesh, **kwargs):
        super(ParametricEllipsoid, self).__init__(mesh, p=2, **kwargs)


class ParametricCasingAndLayer(ParametricLayer):
    """
    Parametric layered space with casing.

    .. code:: python

        m = [val_background,
             val_layer,
             val_casing,
             val_insideCasing,
             layer_center,
             layer_thickness,
             casing_radius,
             casing_thickness,
             casing_bottom,
             casing_top
        ]

    """

    def __init__(self, mesh, **kwargs):
        assert (
            mesh._meshType == "CYL"
        ), "Parametric Casing in a layer map only works for a cyl mesh."

        super().__init__(mesh, **kwargs)

    @property
    def nP(self):
        return 10

    @property
    def shape(self):
        if self.indActive is not None:
            return (sum(self.indActive), self.nP)
        return (self.mesh.nC, self.nP)

    def mDict(self, m):
        # m = [val_background, val_layer, val_casing, val_insideCasing,
        #      layer_center, layer_thickness, casing_radius, casing_thickness,
        #      casing_bottom, casing_top]

        return {
            "val_background": m[0],
            "val_layer": m[1],
            "val_casing": m[2],
            "val_insideCasing": m[3],
            "layer_center": m[4],
            "layer_thickness": m[5],
            "casing_radius": m[6],
            "casing_thickness": m[7],
            "casing_bottom": m[8],
            "casing_top": m[9],
        }

    def casing_a(self, mDict):
        return mDict["casing_radius"] - 0.5 * mDict["casing_thickness"]

    def casing_b(self, mDict):
        return mDict["casing_radius"] + 0.5 * mDict["casing_thickness"]

    def _atanCasingLength(self, mDict):
        return self._atanfct(self.z - mDict["casing_top"], -self.slope) * self._atanfct(
            self.z - mDict["casing_bottom"], self.slope
        )

    def _atanCasingLengthDeriv_casing_top(self, mDict):
        return self._atanfctDeriv(
            self.z - mDict["casing_top"], -self.slope
        ) * self._atanfct(self.z - mDict["casing_bottom"], self.slope)

    def _atanCasingLengthDeriv_casing_bottom(self, mDict):
        return self._atanfct(
            self.z - mDict["casing_top"], -self.slope
        ) * self._atanfctDeriv(self.z - mDict["casing_bottom"], self.slope)

    def _atanInsideCasing(self, mDict):
        return self._atanCasingLength(mDict) * self._atanfct(
            self.x - self.casing_a(mDict), -self.slope
        )

    def _atanInsideCasingDeriv_casing_radius(self, mDict):
        return self._atanCasingLength(mDict) * self._atanfctDeriv(
            self.x - self.casing_a(mDict), -self.slope
        )

    def _atanInsideCasingDeriv_casing_thickness(self, mDict):
        return (
            self._atanCasingLength(mDict)
            * -0.5
            * self._atanfctDeriv(self.x - self.casing_a(mDict), -self.slope)
        )

    def _atanInsideCasingDeriv_casing_top(self, mDict):
        return self._atanCasingLengthDeriv_casing_top(mDict) * self._atanfct(
            self.x - self.casing_a(mDict), -self.slope
        )

    def _atanInsideCasingDeriv_casing_bottom(self, mDict):
        return self._atanCasingLengthDeriv_casing_bottom(mDict) * self._atanfct(
            self.x - self.casing_a(mDict), -self.slope
        )

    def _atanCasing(self, mDict):
        return (
            self._atanCasingLength(mDict)
            * self._atanfct(self.x - self.casing_a(mDict), self.slope)
            * self._atanfct(self.x - self.casing_b(mDict), -self.slope)
        )

    def _atanCasingDeriv_casing_radius(self, mDict):
        return self._atanCasingLength(mDict) * (
            self._atanfctDeriv(self.x - self.casing_a(mDict), self.slope)
            * self._atanfct(self.x - self.casing_b(mDict), -self.slope)
            + self._atanfct(self.x - self.casing_a(mDict), self.slope)
            * self._atanfctDeriv(self.x - self.casing_b(mDict), -self.slope)
        )

    def _atanCasingDeriv_casing_thickness(self, mDict):
        return self._atanCasingLength(mDict) * (
            -0.5
            * self._atanfctDeriv(self.x - self.casing_a(mDict), self.slope)
            * self._atanfct(self.x - self.casing_b(mDict), -self.slope)
            + self._atanfct(self.x - self.casing_a(mDict), self.slope)
            * 0.5
            * self._atanfctDeriv(self.x - self.casing_b(mDict), -self.slope)
        )

    def _atanCasingDeriv_casing_bottom(self, mDict):
        return (
            self._atanCasingLengthDeriv_casing_bottom(mDict)
            * self._atanfct(self.x - self.casing_a(mDict), self.slope)
            * self._atanfct(self.x - self.casing_b(mDict), -self.slope)
        )

    def _atanCasingDeriv_casing_top(self, mDict):
        return (
            self._atanCasingLengthDeriv_casing_top(mDict)
            * self._atanfct(self.x - self.casing_a(mDict), self.slope)
            * self._atanfct(self.x - self.casing_b(mDict), -self.slope)
        )

    def layer_cont(self, mDict):
        # contribution from the layered background
        return mDict["val_background"] + (
            mDict["val_layer"] - mDict["val_background"]
        ) * self._atanLayer(mDict)

    def _transform(self, m):
        mDict = self.mDict(m)

        # assemble the model
        layer = self.layer_cont(mDict)
        casing = (mDict["val_casing"] - layer) * self._atanCasing(mDict)
        insideCasing = (mDict["val_insideCasing"] - layer) * self._atanInsideCasing(
            mDict
        )

        return layer + casing + insideCasing

    def _deriv_val_background(self, mDict):
        # contribution from the layered background
        d_layer_cont_dval_background = 1.0 - self._atanLayer(mDict)
        d_casing_cont_dval_background = (
            -1.0 * d_layer_cont_dval_background * self._atanCasing(mDict)
        )
        d_insideCasing_cont_dval_background = (
            -1.0 * d_layer_cont_dval_background * self._atanInsideCasing(mDict)
        )
        return (
            d_layer_cont_dval_background
            + d_casing_cont_dval_background
            + d_insideCasing_cont_dval_background
        )

    def _deriv_val_layer(self, mDict):
        d_layer_cont_dval_layer = self._atanLayer(mDict)
        d_casing_cont_dval_layer = (
            -1.0 * d_layer_cont_dval_layer * self._atanCasing(mDict)
        )
        d_insideCasing_cont_dval_layer = (
            -1.0 * d_layer_cont_dval_layer * self._atanInsideCasing(mDict)
        )
        return (
            d_layer_cont_dval_layer
            + d_casing_cont_dval_layer
            + d_insideCasing_cont_dval_layer
        )

    def _deriv_val_casing(self, mDict):
        d_layer_cont_dval_casing = 0.0
        d_casing_cont_dval_casing = self._atanCasing(mDict)
        d_insideCasing_cont_dval_casing = 0.0
        return (
            d_layer_cont_dval_casing
            + d_casing_cont_dval_casing
            + d_insideCasing_cont_dval_casing
        )

    def _deriv_val_insideCasing(self, mDict):
        d_layer_cont_dval_insideCasing = 0.0
        d_casing_cont_dval_insideCasing = 0.0
        d_insideCasing_cont_dval_insideCasing = self._atanInsideCasing(mDict)
        return (
            d_layer_cont_dval_insideCasing
            + d_casing_cont_dval_insideCasing
            + d_insideCasing_cont_dval_insideCasing
        )

    def _deriv_layer_center(self, mDict):
        d_layer_cont_dlayer_center = (
            mDict["val_layer"] - mDict["val_background"]
        ) * self._atanLayerDeriv_layer_center(mDict)
        d_casing_cont_dlayer_center = -d_layer_cont_dlayer_center * self._atanCasing(
            mDict
        )
        d_insideCasing_cont_dlayer_center = (
            -d_layer_cont_dlayer_center * self._atanInsideCasing(mDict)
        )
        return (
            d_layer_cont_dlayer_center
            + d_casing_cont_dlayer_center
            + d_insideCasing_cont_dlayer_center
        )

    def _deriv_layer_thickness(self, mDict):
        d_layer_cont_dlayer_thickness = (
            mDict["val_layer"] - mDict["val_background"]
        ) * self._atanLayerDeriv_layer_thickness(mDict)
        d_casing_cont_dlayer_thickness = (
            -d_layer_cont_dlayer_thickness * self._atanCasing(mDict)
        )
        d_insideCasing_cont_dlayer_thickness = (
            -d_layer_cont_dlayer_thickness * self._atanInsideCasing(mDict)
        )
        return (
            d_layer_cont_dlayer_thickness
            + d_casing_cont_dlayer_thickness
            + d_insideCasing_cont_dlayer_thickness
        )

    def _deriv_casing_radius(self, mDict):
        layer = self.layer_cont(mDict)
        d_layer_cont_dcasing_radius = 0.0
        d_casing_cont_dcasing_radius = (
            mDict["val_casing"] - layer
        ) * self._atanCasingDeriv_casing_radius(mDict)
        d_insideCasing_cont_dcasing_radius = (
            mDict["val_insideCasing"] - layer
        ) * self._atanInsideCasingDeriv_casing_radius(mDict)
        return (
            d_layer_cont_dcasing_radius
            + d_casing_cont_dcasing_radius
            + d_insideCasing_cont_dcasing_radius
        )

    def _deriv_casing_thickness(self, mDict):
        d_layer_cont_dcasing_thickness = 0.0
        d_casing_cont_dcasing_thickness = (
            mDict["val_casing"] - self.layer_cont(mDict)
        ) * self._atanCasingDeriv_casing_thickness(mDict)
        d_insideCasing_cont_dcasing_thickness = (
            mDict["val_insideCasing"] - self.layer_cont(mDict)
        ) * self._atanInsideCasingDeriv_casing_thickness(mDict)
        return (
            d_layer_cont_dcasing_thickness
            + d_casing_cont_dcasing_thickness
            + d_insideCasing_cont_dcasing_thickness
        )

    def _deriv_casing_bottom(self, mDict):
        d_layer_cont_dcasing_bottom = 0.0
        d_casing_cont_dcasing_bottom = (
            mDict["val_casing"] - self.layer_cont(mDict)
        ) * self._atanCasingDeriv_casing_bottom(mDict)
        d_insideCasing_cont_dcasing_bottom = (
            mDict["val_insideCasing"] - self.layer_cont(mDict)
        ) * self._atanInsideCasingDeriv_casing_bottom(mDict)
        return (
            d_layer_cont_dcasing_bottom
            + d_casing_cont_dcasing_bottom
            + d_insideCasing_cont_dcasing_bottom
        )

    def _deriv_casing_top(self, mDict):
        d_layer_cont_dcasing_top = 0.0
        d_casing_cont_dcasing_top = (
            mDict["val_casing"] - self.layer_cont(mDict)
        ) * self._atanCasingDeriv_casing_top(mDict)
        d_insideCasing_cont_dcasing_top = (
            mDict["val_insideCasing"] - self.layer_cont(mDict)
        ) * self._atanInsideCasingDeriv_casing_top(mDict)
        return (
            d_layer_cont_dcasing_top
            + d_casing_cont_dcasing_top
            + d_insideCasing_cont_dcasing_top
        )

    def deriv(self, m):
        mDict = self.mDict(m)

        return sp.csr_matrix(
            np.vstack(
                [
                    self._deriv_val_background(mDict),
                    self._deriv_val_layer(mDict),
                    self._deriv_val_casing(mDict),
                    self._deriv_val_insideCasing(mDict),
                    self._deriv_layer_center(mDict),
                    self._deriv_layer_thickness(mDict),
                    self._deriv_casing_radius(mDict),
                    self._deriv_casing_thickness(mDict),
                    self._deriv_casing_bottom(mDict),
                    self._deriv_casing_top(mDict),
                ]
            ).T
        )


class ParametricBlockInLayer(ParametricLayer):
    """
    Parametric Block in a Layered Space

    For 2D:

    .. code:: python

        m = [val_background,
             val_layer,
             val_block,
             layer_center,
             layer_thickness,
             block_x0,
             block_dx
        ]

    For 3D:

    .. code:: python

        m = [val_background,
             val_layer,
             val_block,
             layer_center,
             layer_thickness,
             block_x0,
             block_y0,
             block_dx,
             block_dy
        ]

    **Required**

    :param discretize.base.BaseMesh mesh: SimPEG Mesh, 2D or 3D

    **Optional**

    :param float slopeFact: arctan slope factor - divided by the minimum h
                            spacing to give the slope of the arctan
                            functions
    :param float slope: slope of the arctan function
    :param numpy.ndarray indActive: bool vector with

    """

    def __init__(self, mesh, **kwargs):
        super().__init__(mesh, **kwargs)

    @property
    def nP(self):
        if self.mesh.dim == 2:
            return 7
        elif self.mesh.dim == 3:
            return 9

    @property
    def shape(self):
        if self.indActive is not None:
            return (sum(self.indActive), self.nP)
        return (self.mesh.nC, self.nP)

    def _mDict2d(self, m):
        return {
            "val_background": m[0],
            "val_layer": m[1],
            "val_block": m[2],
            "layer_center": m[3],
            "layer_thickness": m[4],
            "x0": m[5],
            "dx": m[6],
        }

    def _mDict3d(self, m):
        return {
            "val_background": m[0],
            "val_layer": m[1],
            "val_block": m[2],
            "layer_center": m[3],
            "layer_thickness": m[4],
            "x0": m[5],
            "y0": m[6],
            "dx": m[7],
            "dy": m[8],
        }

    def mDict(self, m):
        if self.mesh.dim == 2:
            return self._mDict2d(m)
        elif self.mesh.dim == 3:
            return self._mDict3d(m)

    def xleft(self, mDict):
        return mDict["x0"] - 0.5 * mDict["dx"]

    def xright(self, mDict):
        return mDict["x0"] + 0.5 * mDict["dx"]

    def yleft(self, mDict):
        return mDict["y0"] - 0.5 * mDict["dy"]

    def yright(self, mDict):
        return mDict["y0"] + 0.5 * mDict["dy"]

    def _atanBlock2d(self, mDict):
        return (
            self._atanLayer(mDict)
            * self._atanfct(self.x - self.xleft(mDict), self.slope)
            * self._atanfct(self.x - self.xright(mDict), -self.slope)
        )

    def _atanBlock2dDeriv_layer_center(self, mDict):
        return (
            self._atanLayerDeriv_layer_center(mDict)
            * self._atanfct(self.x - self.xleft(mDict), self.slope)
            * self._atanfct(self.x - self.xright(mDict), -self.slope)
        )

    def _atanBlock2dDeriv_layer_thickness(self, mDict):
        return (
            self._atanLayerDeriv_layer_thickness(mDict)
            * self._atanfct(self.x - self.xleft(mDict), self.slope)
            * self._atanfct(self.x - self.xright(mDict), -self.slope)
        )

    def _atanBlock2dDeriv_x0(self, mDict):
        return self._atanLayer(mDict) * (
            (
                self._atanfctDeriv(self.x - self.xleft(mDict), self.slope)
                * self._atanfct(self.x - self.xright(mDict), -self.slope)
            )
            + (
                self._atanfct(self.x - self.xleft(mDict), self.slope)
                * self._atanfctDeriv(self.x - self.xright(mDict), -self.slope)
            )
        )

    def _atanBlock2dDeriv_dx(self, mDict):
        return self._atanLayer(mDict) * (
            (
                self._atanfctDeriv(self.x - self.xleft(mDict), self.slope)
                * -0.5
                * self._atanfct(self.x - self.xright(mDict), -self.slope)
            )
            + (
                self._atanfct(self.x - self.xleft(mDict), self.slope)
                * 0.5
                * self._atanfctDeriv(self.x - self.xright(mDict), -self.slope)
            )
        )

    def _atanBlock3d(self, mDict):
        return (
            self._atanLayer(mDict)
            * self._atanfct(self.x - self.xleft(mDict), self.slope)
            * self._atanfct(self.x - self.xright(mDict), -self.slope)
            * self._atanfct(self.y - self.yleft(mDict), self.slope)
            * self._atanfct(self.y - self.yright(mDict), -self.slope)
        )

    def _atanBlock3dDeriv_layer_center(self, mDict):
        return (
            self._atanLayerDeriv_layer_center(mDict)
            * self._atanfct(self.x - self.xleft(mDict), self.slope)
            * self._atanfct(self.x - self.xright(mDict), -self.slope)
            * self._atanfct(self.y - self.yleft(mDict), self.slope)
            * self._atanfct(self.y - self.yright(mDict), -self.slope)
        )

    def _atanBlock3dDeriv_layer_thickness(self, mDict):
        return (
            self._atanLayerDeriv_layer_thickness(mDict)
            * self._atanfct(self.x - self.xleft(mDict), self.slope)
            * self._atanfct(self.x - self.xright(mDict), -self.slope)
            * self._atanfct(self.y - self.yleft(mDict), self.slope)
            * self._atanfct(self.y - self.yright(mDict), -self.slope)
        )

    def _atanBlock3dDeriv_x0(self, mDict):
        return self._atanLayer(mDict) * (
            (
                self._atanfctDeriv(self.x - self.xleft(mDict), self.slope)
                * self._atanfct(self.x - self.xright(mDict), -self.slope)
                * self._atanfct(self.y - self.yleft(mDict), self.slope)
                * self._atanfct(self.y - self.yright(mDict), -self.slope)
            )
            + (
                self._atanfct(self.x - self.xleft(mDict), self.slope)
                * self._atanfctDeriv(self.x - self.xright(mDict), -self.slope)
                * self._atanfct(self.y - self.yleft(mDict), self.slope)
                * self._atanfct(self.y - self.yright(mDict), -self.slope)
            )
        )

    def _atanBlock3dDeriv_y0(self, mDict):
        return self._atanLayer(mDict) * (
            (
                self._atanfct(self.x - self.xleft(mDict), self.slope)
                * self._atanfct(self.x - self.xright(mDict), -self.slope)
                * self._atanfctDeriv(self.y - self.yleft(mDict), self.slope)
                * self._atanfct(self.y - self.yright(mDict), -self.slope)
            )
            + (
                self._atanfct(self.x - self.xleft(mDict), self.slope)
                * self._atanfct(self.x - self.xright(mDict), -self.slope)
                * self._atanfct(self.y - self.yleft(mDict), self.slope)
                * self._atanfctDeriv(self.y - self.yright(mDict), -self.slope)
            )
        )

    def _atanBlock3dDeriv_dx(self, mDict):
        return self._atanLayer(mDict) * (
            (
                self._atanfctDeriv(self.x - self.xleft(mDict), self.slope)
                * -0.5
                * self._atanfct(self.x - self.xright(mDict), -self.slope)
                * self._atanfct(self.y - self.yleft(mDict), self.slope)
                * self._atanfct(self.y - self.yright(mDict), -self.slope)
            )
            + (
                self._atanfct(self.x - self.xleft(mDict), self.slope)
                * self._atanfctDeriv(self.x - self.xright(mDict), -self.slope)
                * 0.5
                * self._atanfct(self.y - self.yleft(mDict), self.slope)
                * self._atanfct(self.y - self.yright(mDict), -self.slope)
            )
        )

    def _atanBlock3dDeriv_dy(self, mDict):
        return self._atanLayer(mDict) * (
            (
                self._atanfct(self.x - self.xleft(mDict), self.slope)
                * self._atanfct(self.x - self.xright(mDict), -self.slope)
                * self._atanfctDeriv(self.y - self.yleft(mDict), self.slope)
                * -0.5
                * self._atanfct(self.y - self.yright(mDict), -self.slope)
            )
            + (
                self._atanfct(self.x - self.xleft(mDict), self.slope)
                * self._atanfct(self.x - self.xright(mDict), -self.slope)
                * self._atanfct(self.y - self.yleft(mDict), self.slope)
                * self._atanfctDeriv(self.y - self.yright(mDict), -self.slope)
                * 0.5
            )
        )

    def _transform2d(self, m):
        mDict = self.mDict(m)
        # assemble the model
        # contribution from the layered background
        layer_cont = mDict["val_background"] + (
            mDict["val_layer"] - mDict["val_background"]
        ) * self._atanLayer(mDict)

        # perturbation due to the blocks
        block_cont = (mDict["val_block"] - layer_cont) * self._atanBlock2d(mDict)

        return layer_cont + block_cont

    def _deriv2d_val_background(self, mDict):
        d_layer_dval_background = np.ones_like(self.x) - self._atanLayer(mDict)
        d_block_dval_background = (-d_layer_dval_background) * self._atanBlock2d(mDict)
        return d_layer_dval_background + d_block_dval_background

    def _deriv2d_val_layer(self, mDict):
        d_layer_dval_layer = self._atanLayer(mDict)
        d_block_dval_layer = (-d_layer_dval_layer) * self._atanBlock2d(mDict)
        return d_layer_dval_layer + d_block_dval_layer

    def _deriv2d_val_block(self, mDict):
        d_layer_dval_block = 0.0
        d_block_dval_block = (1.0 - d_layer_dval_block) * self._atanBlock2d(mDict)
        return d_layer_dval_block + d_block_dval_block

    def _deriv2d_layer_center(self, mDict):
        d_layer_dlayer_center = (
            mDict["val_layer"] - mDict["val_background"]
        ) * self._atanLayerDeriv_layer_center(mDict)
        d_block_dlayer_center = (
            mDict["val_block"] - self.layer_cont(mDict)
        ) * self._atanBlock2dDeriv_layer_center(
            mDict
        ) - d_layer_dlayer_center * self._atanBlock2d(
            mDict
        )
        return d_layer_dlayer_center + d_block_dlayer_center

    def _deriv2d_layer_thickness(self, mDict):
        d_layer_dlayer_thickness = (
            mDict["val_layer"] - mDict["val_background"]
        ) * self._atanLayerDeriv_layer_thickness(mDict)
        d_block_dlayer_thickness = (
            mDict["val_block"] - self.layer_cont(mDict)
        ) * self._atanBlock2dDeriv_layer_thickness(
            mDict
        ) - d_layer_dlayer_thickness * self._atanBlock2d(
            mDict
        )
        return d_layer_dlayer_thickness + d_block_dlayer_thickness

    def _deriv2d_x0(self, mDict):
        d_layer_dx0 = 0.0
        d_block_dx0 = (
            mDict["val_block"] - self.layer_cont(mDict)
        ) * self._atanBlock2dDeriv_x0(mDict)
        return d_layer_dx0 + d_block_dx0

    def _deriv2d_dx(self, mDict):
        d_layer_ddx = 0.0
        d_block_ddx = (
            mDict["val_block"] - self.layer_cont(mDict)
        ) * self._atanBlock2dDeriv_dx(mDict)
        return d_layer_ddx + d_block_ddx

    def _deriv2d(self, m):
        mDict = self.mDict(m)

        return np.vstack(
            [
                self._deriv2d_val_background(mDict),
                self._deriv2d_val_layer(mDict),
                self._deriv2d_val_block(mDict),
                self._deriv2d_layer_center(mDict),
                self._deriv2d_layer_thickness(mDict),
                self._deriv2d_x0(mDict),
                self._deriv2d_dx(mDict),
            ]
        ).T

    def _transform3d(self, m):
        # parse model
        mDict = self.mDict(m)

        # assemble the model
        # contribution from the layered background
        layer_cont = mDict["val_background"] + (
            mDict["val_layer"] - mDict["val_background"]
        ) * self._atanLayer(mDict)
        # perturbation due to the block
        block_cont = (mDict["val_block"] - layer_cont) * self._atanBlock3d(mDict)

        return layer_cont + block_cont

    def _deriv3d_val_background(self, mDict):
        d_layer_dval_background = np.ones_like(self.x) - self._atanLayer(mDict)
        d_block_dval_background = (-d_layer_dval_background) * self._atanBlock3d(mDict)
        return d_layer_dval_background + d_block_dval_background

    def _deriv3d_val_layer(self, mDict):
        d_layer_dval_layer = self._atanLayer(mDict)
        d_block_dval_layer = (-d_layer_dval_layer) * self._atanBlock3d(mDict)
        return d_layer_dval_layer + d_block_dval_layer

    def _deriv3d_val_block(self, mDict):
        d_layer_dval_block = 0.0
        d_block_dval_block = (1.0 - d_layer_dval_block) * self._atanBlock3d(mDict)
        return d_layer_dval_block + d_block_dval_block

    def _deriv3d_layer_center(self, mDict):
        d_layer_dlayer_center = (
            mDict["val_layer"] - mDict["val_background"]
        ) * self._atanLayerDeriv_layer_center(mDict)
        d_block_dlayer_center = (
            mDict["val_block"] - self.layer_cont(mDict)
        ) * self._atanBlock3dDeriv_layer_center(
            mDict
        ) - d_layer_dlayer_center * self._atanBlock3d(
            mDict
        )
        return d_layer_dlayer_center + d_block_dlayer_center

    def _deriv3d_layer_thickness(self, mDict):
        d_layer_dlayer_thickness = (
            mDict["val_layer"] - mDict["val_background"]
        ) * self._atanLayerDeriv_layer_thickness(mDict)
        d_block_dlayer_thickness = (
            mDict["val_block"] - self.layer_cont(mDict)
        ) * self._atanBlock3dDeriv_layer_thickness(
            mDict
        ) - d_layer_dlayer_thickness * self._atanBlock3d(
            mDict
        )
        return d_layer_dlayer_thickness + d_block_dlayer_thickness

    def _deriv3d_x0(self, mDict):
        d_layer_dx0 = 0.0
        d_block_dx0 = (
            mDict["val_block"] - self.layer_cont(mDict)
        ) * self._atanBlock3dDeriv_x0(mDict)
        return d_layer_dx0 + d_block_dx0

    def _deriv3d_y0(self, mDict):
        d_layer_dy0 = 0.0
        d_block_dy0 = (
            mDict["val_block"] - self.layer_cont(mDict)
        ) * self._atanBlock3dDeriv_y0(mDict)
        return d_layer_dy0 + d_block_dy0

    def _deriv3d_dx(self, mDict):
        d_layer_ddx = 0.0
        d_block_ddx = (
            mDict["val_block"] - self.layer_cont(mDict)
        ) * self._atanBlock3dDeriv_dx(mDict)
        return d_layer_ddx + d_block_ddx

    def _deriv3d_dy(self, mDict):
        d_layer_ddy = 0.0
        d_block_ddy = (
            mDict["val_block"] - self.layer_cont(mDict)
        ) * self._atanBlock3dDeriv_dy(mDict)
        return d_layer_ddy + d_block_ddy

    def _deriv3d(self, m):
        mDict = self.mDict(m)

        return np.vstack(
            [
                self._deriv3d_val_background(mDict),
                self._deriv3d_val_layer(mDict),
                self._deriv3d_val_block(mDict),
                self._deriv3d_layer_center(mDict),
                self._deriv3d_layer_thickness(mDict),
                self._deriv3d_x0(mDict),
                self._deriv3d_y0(mDict),
                self._deriv3d_dx(mDict),
                self._deriv3d_dy(mDict),
            ]
        ).T

    def _transform(self, m):
        if self.mesh.dim == 2:
            return self._transform2d(m)
        elif self.mesh.dim == 3:
            return self._transform3d(m)

    def deriv(self, m):
        if self.mesh.dim == 2:
            return sp.csr_matrix(self._deriv2d(m))
        elif self.mesh.dim == 3:
            return sp.csr_matrix(self._deriv3d(m))


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


###############################################################################
#                                                                             #
#                       Maps for petrophsyics clusters                        #
#                                                                             #
###############################################################################


class PolynomialPetroClusterMap(IdentityMap):
    """
    Modeling polynomial relationships between physical properties

    Parameters
    ----------
    coeffxx : array_like, optional
        Coefficients for the xx component. Default is [0, 1]
    coeffxy : array_like, optional
        Coefficients for the xy component. Default is [0]
    coeffyx : array_like, optional
        Coefficients for the yx component. Default is [0]
    coeffyy : array_like, optional
        Coefficients for the yy component. Default is [0, 1]
    """

    def __init__(
        self,
        coeffxx=None,
        coeffxy=None,
        coeffyx=None,
        coeffyy=None,
        mesh=None,
        nP=None,
        **kwargs,
    ):
        if coeffxx is None:
            coeffxx = np.r_[0.0, 1.0]
        if coeffxy is None:
            coeffxy = np.r_[0.0]
        if coeffyx is None:
            coeffyx = np.r_[0.0]
        if coeffyy is None:
            coeffyy = np.r_[0.0, 1.0]

        self._coeffxx = validate_ndarray_with_shape("coeffxx", coeffxx, shape=("*",))
        self._coeffxy = validate_ndarray_with_shape("coeffxy", coeffxy, shape=("*",))
        self._coeffyx = validate_ndarray_with_shape("coeffyx", coeffyx, shape=("*",))
        self._coeffyy = validate_ndarray_with_shape("coeffyy", coeffyy, shape=("*",))

        self._polynomialxx = polynomial.Polynomial(self.coeffxx)
        self._polynomialxy = polynomial.Polynomial(self.coeffxy)
        self._polynomialyx = polynomial.Polynomial(self.coeffyx)
        self._polynomialyy = polynomial.Polynomial(self.coeffyy)
        self._polynomialxx_deriv = self._polynomialxx.deriv(m=1)
        self._polynomialxy_deriv = self._polynomialxy.deriv(m=1)
        self._polynomialyx_deriv = self._polynomialyx.deriv(m=1)
        self._polynomialyy_deriv = self._polynomialyy.deriv(m=1)

        super().__init__(mesh=mesh, nP=nP, **kwargs)

    @property
    def coeffxx(self):
        """Coefficients for the xx component.

        Returns
        -------
        numpy.ndarray
        """
        return self._coeffxx

    @property
    def coeffxy(self):
        """Coefficients for the xy component.

        Returns
        -------
        numpy.ndarray
        """
        return self._coeffxy

    @property
    def coeffyx(self):
        """Coefficients for the yx component.

        Returns
        -------
        numpy.ndarray
        """
        return self._coeffyx

    @property
    def coeffyy(self):
        """Coefficients for the yy component.

        Returns
        -------
        numpy.ndarray
        """
        return self._coeffyy

    def _transform(self, m):
        out = m.copy()
        out[:, 0] = self._polynomialxx(m[:, 0]) + self._polynomialxy(m[:, 1])
        out[:, 1] = self._polynomialyx(m[:, 0]) + self._polynomialyy(m[:, 1])
        return out

    def inverse(self, D):
        r"""
        :param numpy.array D: physical property
        :rtype: numpy.array
        :return: model

        The *transformInverse* changes the physical property into the
        model.

        .. math::

            m = \log{\sigma}

        """
        raise NotImplementedError("Inverse is not implemented.")

    def _derivmatrix(self, m):
        return np.r_[
            [
                [
                    self._polynomialxx_deriv(m[:, 0])[0],
                    self._polynomialyx_deriv(m[:, 0])[0],
                ],
                [
                    self._polynomialxy_deriv(m[:, 1])[0],
                    self._polynomialyy_deriv(m[:, 1])[0],
                ],
            ]
        ]

    def deriv(self, m, v=None):
        """"""
        if v is None:
            out = self._derivmatrix(m.reshape(-1, 2))
            return out
        else:
            out = np.dot(self._derivmatrix(m.reshape(-1, 2)), v.reshape(2, -1))
            return out

    @property
    def is_linear(self):
        return False
