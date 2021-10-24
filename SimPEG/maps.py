from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from .utils.code_utils import deprecate_class

from six import integer_types
from six import string_types
from collections import namedtuple
import warnings

import numpy as np
from numpy.polynomial import polynomial
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator
from scipy.interpolate import UnivariateSpline
from scipy.constants import mu_0
from scipy.sparse import csr_matrix as csr

import properties
from discretize.tests import checkDerivative

from .utils import (
    setKwargs,
    mkvc,
    rotationMatrixFromNormals,
    Zero,
    Identity,
    sdiag,
    mat_utils,
    speye,
)


class IdentityMap(properties.HasProperties):
    r"""Identity mapping and the base mapping class for all other SimPEG mappings.

    The ``IdentityMap`` class is used to define the mapping when
    the model parameters are the same as the parameters used in the forward
    simulation. For a discrete set of model parameters, the mapping is equivalent
    to the identity matrix. The ``IdentityMap`` also acts as the base class for
    all other SimPEG mapping classes.

    Using the *mesh* or *nP* input arguments, the dimensions of the corresponding
    mapping operator can be permanently set; i.e. (*mesh.nC*, *mesh.nC*) or (*nP*, *nP*).
    However if both input arguments *mesh* and *nP* are ``None``, the shape of
    mapping operator is arbitrary and can act on any vector; i.e. has shape (``*``, ``*``).
    
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
        setKwargs(self, **kwargs)

        if nP is not None:
            if isinstance(nP, string_types):
                assert nP == "*", "nP must be an integer or '*', not {}".format(nP)
            assert isinstance(
                nP, integer_types + (np.int64,)
            ), "Number of parameters must be an integer. Not `{}`.".format(type(nP))
            nP = int(nP)
        elif mesh is not None:
            nP = mesh.nC
        else:
            nP = "*"

        self.mesh = mesh
        self._nP = nP

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
        identity map, the shape of mapping operator is (mesh.nC, mesh.nC).
        If *nP* is used to define the identity map, the mapping operator
        has dimensions (*nP*, *nP*). However if both *mesh* and *nP* are
        used to define the identity map, the mapping will have shape
        (*mesh.nC*, *nP*)! And if *mesh* and *nP* were `None` when
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
        Perform the inverse mapping (not implemented)
        """
        raise NotImplementedError("The transformInverse is not implemented.")

    def deriv(self, m, v=None):
        r"""Derivative of the mapping with respect to the input parameters.

        Let :math:`\mathbf{m}` be a set of model parameters and let :math:`\mathbf{I}`
        denote the identity map. Where the identity mapping acting on the model parameters
        can be expressed as:

        .. math::
            \mathbf{u} = \mathbf{I m},

        the **deriv** method returns the derivative of :math:`\mathbf{u}` with respect
        to the model parameters; i.e.:

        .. math::
            \frac{\partial \mathbf{u}}{\partial \mathbf{m}} = \mathbf{I}

        Note that in this case, **deriv** simply returns a sparse identity matrix.

        Parameters
        ----------
        m : (nP) numpy.ndarray
            A vector representing a set of model parameters
        v : (nP) numpy.ndarray
            If not ``None``, the method returns the derivative times the vector *v*

        Returns
        -------
        scipy.sparse.csr_matrix
            Derivative of the mapping with respect to the model parameters. For an
            identity mapping, this is just a sparse identity matrix. If the input
            argument *v* is not ``None``, the method returns the derivative times
            the vector *v*; which in this case is just *v*.
        """
        if v is not None:
            return v
        if isinstance(self.nP, integer_types):
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
            match those used in :meth:`discretize.tests.checkDerivative`
        
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
            self.nP, integer_types
        ), "nP must be an integer for {}".format(self.__class__.__name__)
        return checkDerivative(
            lambda m: [self * m, self.deriv(m)], m, num=num, **kwargs
        )

    def testVec(self, m=None, **kwargs):
        """Derivative test for the mapping times the model.

        This test validates the mapping by performing a convergence test
        on the mapping time a model.
        
        Parameters
        ----------
        m : (nP) numpy.ndarray
            Starting vector of model parameters for the derivative test
        num : int
            Number of iterations for the derivative test
        kwargs: dict
            Keyword arguments and associated values in the dictionary must
            match those used in :meth:`discretize.tests.checkDerivative`
        
        Returns
        -------
        bool
            Returns ``True`` if the test passes
        """
        print("Testing {0!s}".format(self))
        if m is None:
            m = abs(np.random.rand(self.nP))
        if "plotIt" not in kwargs:
            kwargs["plotIt"] = False
        return checkDerivative(
            lambda m: [self * m, lambda x: self.deriv(m, x)], m, num=4, **kwargs
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
            \mathbf{u}(\mathbf{m}) = (\mathbf{f_2 \circ f_1})(\mathbf{m})
        
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


class ComboMap(IdentityMap):
    r"""Combination mapping constructed by joining a set of other mappings.

    A ``ComboMap`` is a single mapping object made by joining a set
    of basic mapping operations. When creating a ``ComboMap``, the
    user provides a list of SimPEG mapping objects they wish to join.
    The order of the mappings in this list is from last to first; i.e.
    :math:`[\mathbf{f}_n , ... , \mathbf{f}_2 , \mathbf{f}_1]`.

    The combination mapping :math:`\mathbf{u}(\mathbf{m})` that acts on a
    set of input model parameters :math:`\mathbf{m}` is defined as:

    .. math::
        \mathbf{u}(\mathbf{m}) = (\mathbf{f_n} \circ \cdots \circ \mathbf{f_2} \circ \mathbf{f_1})(\mathbf{m})


    Derivatives for the combination mapping are computed using the chain
    rule. Thus:

    .. math::
        \frac{\partial \mathbf{u}}{\partial \mathbf{m}} =
        \frac{\partial \mathbf{f_n}}{\partial \mathbf{f_{n-1}}}
        \cdots
        \frac{\partial \mathbf{f_2}}{\partial \mathbf{f_{1}}}
        \frac{\partial \mathbf{f_1}}{\partial \mathbf{m}}

    Note that any time that you create your own combination mapping,
    be sure to test that the derivative is correct.

    Parameters
    ----------
    maps : list
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
        IdentityMap.__init__(self, None, **kwargs)

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
        tuple
            Dimensions of the mapping operator as a tuple of the
            form (``int``,``int``).
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

        Let :math:`\mathbf{m}` be a set of model parameters and let
        [:math:`\mathbf{f}_n,...,\mathbf{f}_1`] be the list of SimPEG mappings joined
        to create a combination mapping. Recall that the list of mappings is ordered
        from last applied to first applied.

        Where the combination mapping acting on the model parameters
        can be expressed as:

        .. math::
            \mathbf{u}(\mathbf{m}) = (\mathbf{f_n} \circ \cdots \circ \mathbf{f_2} \circ \mathbf{f_1})(\mathbf{m}),

        the **deriv** method returns the derivative of :math:`\mathbf{u}` with respect
        to the model parameters. To do this, we use the chain rule, i.e.:

        .. math::
            \frac{\partial \mathbf{u}}{\partial \mathbf{m}} =
            \frac{\partial \mathbf{f_n}}{\partial \mathbf{f_{n-1}}}
            \cdots
            \frac{\partial \mathbf{f_2}}{\partial \mathbf{f_{1}}}
            \frac{\partial \mathbf{f_1}}{\partial \mathbf{m}}

        Note that any time that you create your own combination mapping,
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


class Projection(IdentityMap):
    """Projection mapping.

    ``Projection`` mapping can be used to project and/or rearange model
    parameters. The number of model parameters the mapping acts on is
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
        IdentityMap.__init__(self, None, **kwargs)

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
    mapping from :math:`\mathbf{m}` to the set of voxel cells defining a mesh.
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

    indices = properties.List(
        "list of indices for each unit to be surjected into",
        properties.Array(
            "indices for the unit to be mapped to", dtype=bool, shape=("*",)
        ),
        required=True,
    )

    # n_blocks = properties.Integer(
    #     "number of times to repeat the mapping", default=1, min=1
    # )

    def __init__(self, indices, **kwargs):
        super(SurjectUnits, self).__init__(**kwargs)
        self.indices = indices

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
    """Mapping vectors from spherical to Cartesian coordinates.


    A vector map to spherical parameters of amplitude, theta and phi
    """

    def __init__(self, mesh=None, nP=None, **kwargs):
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
        """

        :param model:
        :return:
        """
        return mat_utils.spherical2cartesian(model.reshape((-1, 3), order="F"))

    def inverse(self, model):
        """
        Cartesian to spherical.

        :param numpy.ndarray model: physical property in Cartesian
        :return: model

        """
        return mat_utils.cartesian2spherical(model.reshape((-1, 3), order="F"))

    @property
    def shape(self):
        """
        Shape of the matrix operation (number of indices x nP)
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


class Wires(object):
    """A mapping class for organizing multiple parameter types into a single model

    """
    def __init__(self, *args):
        for arg in args:
            assert (
                isinstance(arg, tuple)
                and len(arg) == 2
                and isinstance(arg[0], string_types)
                and
                # TODO: this should be extended to a slice.
                isinstance(arg[1], integer_types)
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
        for n, w in self.maps:
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


class SelfConsistentEffectiveMedium(IdentityMap, properties.HasProperties):
    """
        Two phase self-consistent effective medium theory mapping for
        ellipsoidal inclusions. The inversion model is the concentration
        (volume fraction) of the phase 2 material.

        The inversion model is :math:`\\varphi`. We solve for :math:`\sigma`
        given :math:`\sigma_0`, :math:`\sigma_1` and :math:`\\varphi` . Each of
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

            (1-\\varphi)(\sigma - \sigma_0)R^{(0)} + \\varphi(\sigma - \sigma_1)R^{(1)} = 0.

        Where :math:`R^{(j)}` is given by

        .. math::

            R^{(j)} = \\left[1 + \\frac{1}{3}\\frac{\sigma_j - \sigma}{\sigma} \\right]^{-1}.

        **Ellipsoids**

        .. todo::

            Aligned Ellipsoids have not yet been implemented, only randomly
            oriented ellipsoids

        If the inclusions are aligned ellipsoids, we solve

        .. math::

            \sum_{j=1}^N \\varphi_j (\Sigma^* - \sigma_j\mathbf{I}) \mathbf{R}^{j, *} = 0

        where

        .. math::

            \mathbf{R}^{(j, *)} = \left[ \mathbf{I} + \mathbf{A}_j {\Sigma^{*}}^{-1}(\sigma_j \mathbf{I} - \Sigma^*) \\right]^{-1}

        and the depolarization tensor :math:`\mathbf{A}_j` is given by

        .. math::

            \mathbf{A}^* = \\left[\\begin{array}{ccc}
                Q & 0 & 0 \\\\
                0 & Q & 0 \\\\
                0 & 0 & 1-2Q
            \end{array}\\right]

        for a spheroid aligned along the z-axis. For an oblate spheroid
        (:math:`\\alpha < 1`, pancake-like)

        .. math::

            Q = \\frac{1}{2}\\left(
                1 + \\frac{1}{\\alpha^2 - 1} \\left[
                    1 - \\frac{1}{\chi}\\tan^{-1}(\chi)
                \\right]
            \\right)

        where

        .. math::

            \chi = \sqrt{\\frac{1}{\\alpha^2} - 1}


        For reference, see
        `Torquato (2002), Random Heterogeneous Materials <https://link.springer.com/book/10.1007/978-1-4757-6355-3>`_


    """

    sigma0 = properties.Float(
        "physical property value for phase-0 material", min=0.0, required=True
    )  # this should also be allowed to be an array

    sigma1 = properties.Float(
        "physical property value for phase-1 material", min=0.0, required=True
    )

    alpha0 = properties.Float("aspect ratio of the phase-0 ellipsoids", default=1.0)

    alpha1 = properties.Float("aspect ratio of the phase-1 ellipsoids", default=1.0)

    orientation0 = properties.Vector3(
        "orientation of the phase-0 inclusions", default="Z"
    )

    orientation1 = properties.Vector3(
        "orientation of the phase-1 inclusions", default="Z"
    )

    random = properties.Bool(
        "are the inclusions randomly oriented (True) or preferentially "
        "aligned (False)?",
        default=True,
    )

    rel_tol = properties.Float(
        "relative tolerance for convergence for the fixed-point iteration", default=1e-3
    )

    maxIter = properties.Integer(
        "maximum number of iterations for the fixed point iteration " "calculation",
        default=50,
    )

    def __init__(self, mesh=None, nP=None, sigstart=None, **kwargs):
        self._sigstart = sigstart
        super(SelfConsistentEffectiveMedium, self).__init__(mesh, nP, **kwargs)

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
            chi = np.sqrt((1.0 / alpha ** 2.0) - 1)
            return (
                1.0
                / 2.0
                * (1 + 1.0 / (alpha ** 2.0 - 1) * (1.0 - np.arctan(chi) / chi))
            )
        elif alpha > 1.0:  # prolate spheroid
            chi = np.sqrt(1 - (1.0 / alpha ** 2.0))
            return (
                1.0
                / 2.0
                * (
                    1
                    + 1.0
                    / (alpha ** 2.0 - 1)
                    * (1.0 - 1.0 / (2.0 * chi) * np.log((1 + chi) / (1 - chi)))
                )
            )
        elif alpha == 1:  # sphere
            return 1.0 / 3.0

    def getA(self, alpha, orientation):
        """Depolarization tensor"""
        Q = self.getQ(alpha)
        A = np.diag([Q, Q, 1 - 2 * Q])
        R = rotationMatrixFromNormals(np.r_[0.0, 0.0, 1.0], orientation)
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

        for i in range(self.maxIter):
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
        super(ExpMap, self).__init__(mesh=mesh, nP=nP, **kwargs)

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
        super(ReciprocalMap, self).__init__(mesh=mesh, nP=nP, **kwargs)

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
        super(LogMap, self).__init__(mesh=mesh, nP=nP, **kwargs)

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
        super(ChiMap, self).__init__(mesh=mesh, nP=nP, **kwargs)

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
        super(MuRelative, self).__init__(mesh=mesh, nP=nP, **kwargs)

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
            warnings.warn(
                "`nC` is deprecated. Use `nP` to set the number of model "
                "parameters, This option will be removed in version 0.16.0 of SimPEG",
                FutureWarning,
            )
            nP = nC

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
        super(ComplexMap, self).__init__(mesh=mesh, nP=nP, **kwargs)
        if nP is not None and mesh is not None:
            assert 2*mesh.nC == nP, "Number parameters must be 2 X number of mesh cells."
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
    """
    SurjectFull

    Given a scalar, the SurjectFull maps the value to the
    full model space.
    """

    def __init__(self, mesh, **kwargs):
        IdentityMap.__init__(self, mesh, **kwargs)

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
        """
        :param numpy.ndarray m: model
        :rtype: numpy.ndarray
        :return: derivative of transformed model
        """
        deriv = sp.csr_matrix(np.ones([self.mesh.nC, 1]))
        if v is not None:
            return deriv * v
        return deriv


class SurjectVertical1D(IdentityMap):
    """SurjectVertical1DMap

    Given a 1D vector through the last dimension
    of the mesh, this will extend to the full
    model space.
    """

    def __init__(self, mesh, **kwargs):
        IdentityMap.__init__(self, mesh, **kwargs)

    @property
    def nP(self):
        r"""Number of parameters the mapping acts on.

        Returns
        -------
        int
            Number of parameters that the mapping acts on. Equal to the
            number of mesh cells; i.e. `mesh.nC`.
        """
        return int(self.mesh.vnC[self.mesh.dim - 1])

    def _transform(self, m):
        """
        :param numpy.ndarray m: model
        :rtype: numpy.ndarray
        :return: transformed model
        """
        repNum = np.prod(self.mesh.vnC[: self.mesh.dim - 1])
        return mkvc(m).repeat(repNum)

    def deriv(self, m, v=None):
        """
        :param numpy.ndarray m: model
        :rtype: scipy.sparse.csr_matrix
        :return: derivative of transformed model
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
    """Map2Dto3D

    Given a 2D vector, this will extend to the full
    3D model space.
    """

    normal = "Y"  #: The normal

    def __init__(self, mesh, **kwargs):
        assert mesh.dim == 3, "Surject2Dto3D Only works for a 3D Mesh"
        IdentityMap.__init__(self, mesh, **kwargs)
        assert self.normal in ["X", "Y", "Z"], 'For now, only "Y" normal is supported'

    @property
    def nP(self):
        """Number of model properties.

        The number of cells in the
        last dimension of the mesh."""
        if self.normal == "Z":
            return self.mesh.nCx * self.mesh.nCy
        elif self.normal == "Y":
            return self.mesh.nCx * self.mesh.nCz
        elif self.normal == "X":
            return self.mesh.nCy * self.mesh.nCz

    def _transform(self, m):
        """
        :param numpy.ndarray m: model
        :rtype: numpy.ndarray
        :return: transformed model
        """
        m = mkvc(m)
        if self.normal == "Z":
            return mkvc(
                m.reshape(self.mesh.vnC[:2], order="F")[:, :, np.newaxis].repeat(
                    self.mesh.nCz, axis=2
                )
            )
        elif self.normal == "Y":
            return mkvc(
                m.reshape(self.mesh.vnC[::2], order="F")[:, np.newaxis, :].repeat(
                    self.mesh.nCy, axis=1
                )
            )
        elif self.normal == "X":
            return mkvc(
                m.reshape(self.mesh.vnC[1:], order="F")[np.newaxis, :, :].repeat(
                    self.mesh.nCx, axis=0
                )
            )

    def deriv(self, m, v=None):
        """
        :param numpy.ndarray m: model
        :rtype: scipy.sparse.csr_matrix
        :return: derivative of transformed model
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

    indActive = properties.Array("active indices on target mesh", dtype=bool)

    def __init__(self, meshes, **kwargs):
        setKwargs(self, **kwargs)

        assert type(meshes) is list, "meshes must be a list of two meshes"
        assert len(meshes) == 2, "meshes must be a list of two meshes"
        assert (
            meshes[0].dim == meshes[1].dim
        ), "The two meshes must be the same dimension"

        self.mesh = meshes[0]
        self.mesh2 = meshes[1]

    @property
    def P(self):
        if getattr(self, "_P", None) is None:
            self._P = self.mesh2.getInterpolationMat(
                self.mesh.gridCC[self.indActive, :]
                if self.indActive is not None
                else self.mesh.gridCC,
                "CC",
                zerosOutside=True,
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
    """
    Active model parameters.

    """

    indActive = None  #: Active Cells
    valInactive = None  #: Values of inactive Cells

    def __init__(self, mesh, indActive, valInactive, nC=None):
        self.mesh = mesh

        self.nC = nC or mesh.nC

        if indActive.dtype is not bool:
            z = np.zeros(self.nC, dtype=bool)
            z[indActive] = True
            indActive = z
        self.indActive = indActive
        self.indInactive = np.logical_not(indActive)
        if np.isscalar(valInactive):
            self.valInactive = np.ones(self.nC) * float(valInactive)
        else:
            self.valInactive = np.ones(self.nC)
            self.valInactive[self.indInactive] = valInactive.copy()

        self.valInactive[self.indActive] = 0

        inds = np.nonzero(self.indActive)[0]
        self.P = sp.csr_matrix(
            (np.ones(inds.size), (inds, range(inds.size))), shape=(self.nC, self.nP)
        )

    @property
    def shape(self):
        return (self.nC, self.nP)

    @property
    def nP(self):
        """Number of parameters in the model."""
        return int(self.indActive.sum())

    def _transform(self, m):
        return self.P * m + self.valInactive

    def inverse(self, D):
        return self.P.T * D

    def deriv(self, m, v=None):
        if v is not None:
            return self.P * v
        return self.P


###############################################################################
#                                                                             #
#                             Parametric Maps                                 #
#                                                                             #
###############################################################################


class ParametricCircleMap(IdentityMap):
    """ParametricCircleMap

    Parameterize the model space using a circle in a wholespace.

    .. math::

        \sigma(m) = \sigma_1 + (\sigma_2 - \sigma_1)\left(
        \\arctan\left(100*\sqrt{(\\vec{x}-x_0)^2 + (\\vec{y}-y_0)}-r
        \\right) \pi^{-1} + 0.5\\right)

    Define the model as:

    .. math::

        m = [\sigma_1, \sigma_2, x_0, y_0, r]

    """

    slope = 1e-1

    def __init__(self, mesh, logSigma=True):
        assert mesh.dim == 2, (
            "Working for a 2D mesh only right now. "
            "But it isn't that hard to change.. :)"
        )
        IdentityMap.__init__(self, mesh)
        # TODO: this should be done through a composition with and ExpMap
        self.logSigma = logSigma

    @property
    def nP(self):
        return 5

    def _transform(self, m):
        a = self.slope
        sig1, sig2, x, y, r = m[0], m[1], m[2], m[3], m[4]
        if self.logSigma:
            sig1, sig2 = np.exp(sig1), np.exp(sig2)
        X = self.mesh.gridCC[:, 0]
        Y = self.mesh.gridCC[:, 1]
        return sig1 + (sig2 - sig1) * (
            np.arctan(a * (np.sqrt((X - x) ** 2 + (Y - y) ** 2) - r)) / np.pi + 0.5
        )

    def deriv(self, m, v=None):
        a = self.slope
        sig1, sig2, x, y, r = m[0], m[1], m[2], m[3], m[4]
        if self.logSigma:
            sig1, sig2 = np.exp(sig1), np.exp(sig2)
        X = self.mesh.gridCC[:, 0]
        Y = self.mesh.gridCC[:, 1]
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
                * (a ** 2 * (-r + np.sqrt((X - x) ** 2 + (Y - y) ** 2)) ** 2 + 1)
                * np.sqrt((X - x) ** 2 + (Y - y) ** 2)
            )
        )

        g4 = (
            a
            * (-Y + y)
            * (-sig1 + sig2)
            / (
                np.pi
                * (a ** 2 * (-r + np.sqrt((X - x) ** 2 + (Y - y) ** 2)) ** 2 + 1)
                * np.sqrt((X - x) ** 2 + (Y - y) ** 2)
            )
        )

        g5 = (
            -a
            * (-sig1 + sig2)
            / (np.pi * (a ** 2 * (-r + np.sqrt((X - x) ** 2 + (Y - y) ** 2)) ** 2 + 1))
        )

        if v is not None:
            return sp.csr_matrix(np.c_[g1, g2, g3, g4, g5]) * v
        return sp.csr_matrix(np.c_[g1, g2, g3, g4, g5])


class ParametricPolyMap(IdentityMap):

    """PolyMap

    Parameterize the model space using a polynomials in a wholespace.

    .. math::

        y = \mathbf{V} c

    Define the model as:

    .. math::

        m = [\sigma_1, \sigma_2, c]

    Can take in an actInd vector to account for topography.

    """

    def __init__(self, mesh, order, logSigma=True, normal="X", actInd=None):
        IdentityMap.__init__(self, mesh)
        self.logSigma = logSigma
        self.order = order
        self.normal = normal
        self.actInd = actInd

        if getattr(self, "actInd", None) is None:
            self.actInd = list(range(self.mesh.nC))
            self.nC = self.mesh.nC

        else:
            self.nC = len(self.actInd)

    slope = 1e4

    @property
    def shape(self):
        return (self.nC, self.nP)

    @property
    def nP(self):
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
            X = self.mesh.gridCC[self.actInd, 0]
            Y = self.mesh.gridCC[self.actInd, 1]
            if self.normal == "X":
                f = polynomial.polyval(Y, c) - X
            elif self.normal == "Y":
                f = polynomial.polyval(X, c) - Y
            else:
                raise (Exception("Input for normal = X or Y or Z"))

        # 3D
        elif self.mesh.dim == 3:
            X = self.mesh.gridCC[self.actInd, 0]
            Y = self.mesh.gridCC[self.actInd, 1]
            Z = self.mesh.gridCC[self.actInd, 2]

            if self.normal == "X":
                f = (
                    polynomial.polyval2d(
                        Y, Z, c.reshape((self.order[0] + 1, self.order[1] + 1))
                    )
                    - X
                )
            elif self.normal == "Y":
                f = (
                    polynomial.polyval2d(
                        X, Z, c.reshape((self.order[0] + 1, self.order[1] + 1))
                    )
                    - Y
                )
            elif self.normal == "Z":
                f = (
                    polynomial.polyval2d(
                        X, Y, c.reshape((self.order[0] + 1, self.order[1] + 1))
                    )
                    - Z
                )
            else:
                raise (Exception("Input for normal = X or Y or Z"))

        else:
            raise (Exception("Only supports 2D"))

        return sig1 + (sig2 - sig1) * (np.arctan(alpha * f) / np.pi + 0.5)

    def deriv(self, m, v=None):
        alpha = self.slope
        sig1, sig2, c = m[0], m[1], m[2:]
        if self.logSigma:
            sig1, sig2 = np.exp(sig1), np.exp(sig2)

        # 2D
        if self.mesh.dim == 2:
            X = self.mesh.gridCC[self.actInd, 0]
            Y = self.mesh.gridCC[self.actInd, 1]

            if self.normal == "X":
                f = polynomial.polyval(Y, c) - X
                V = polynomial.polyvander(Y, len(c) - 1)
            elif self.normal == "Y":
                f = polynomial.polyval(X, c) - Y
                V = polynomial.polyvander(X, len(c) - 1)
            else:
                raise (Exception("Input for normal = X or Y or Z"))

        # 3D
        elif self.mesh.dim == 3:
            X = self.mesh.gridCC[self.actInd, 0]
            Y = self.mesh.gridCC[self.actInd, 1]
            Z = self.mesh.gridCC[self.actInd, 2]

            if self.normal == "X":
                f = (
                    polynomial.polyval2d(
                        Y, Z, c.reshape((self.order[0] + 1, self.order[1] + 1))
                    )
                    - X
                )
                V = polynomial.polyvander2d(Y, Z, self.order)
            elif self.normal == "Y":
                f = (
                    polynomial.polyval2d(
                        X, Z, c.reshape((self.order[0] + 1, self.order[1] + 1))
                    )
                    - Y
                )
                V = polynomial.polyvander2d(X, Z, self.order)
            elif self.normal == "Z":
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


class ParametricSplineMap(IdentityMap):

    """SplineMap

    Parameterize the boundary of two geological units using
    a spline interpolation

    .. math::

        g = f(x)-y

    Define the model as:

    .. math::

        m = [\sigma_1, \sigma_2, y]

    """

    slope = 1e4

    def __init__(self, mesh, pts, ptsv=None, order=3, logSigma=True, normal="X"):
        IdentityMap.__init__(self, mesh)
        self.logSigma = logSigma
        self.order = order
        self.normal = normal
        self.pts = pts
        self.npts = np.size(pts)
        self.ptsv = ptsv
        self.spl = None

    @property
    def nP(self):
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
            X = self.mesh.gridCC[:, 0]
            Y = self.mesh.gridCC[:, 1]
            self.spl = UnivariateSpline(self.pts, c, k=self.order, s=0)
            if self.normal == "X":
                f = self.spl(Y) - X
            elif self.normal == "Y":
                f = self.spl(X) - Y
            else:
                raise (Exception("Input for normal = X or Y or Z"))

        # 3D:
        # Comments:
        # Make two spline functions and link them using linear interpolation.
        # This is not quite direct extension of 2D to 3D case
        # Using 2D interpolation  is possible

        elif self.mesh.dim == 3:
            X = self.mesh.gridCC[:, 0]
            Y = self.mesh.gridCC[:, 1]
            Z = self.mesh.gridCC[:, 2]

            npts = np.size(self.pts)
            if np.mod(c.size, 2):
                raise (Exception("Put even points!"))

            self.spl = {
                "splb": UnivariateSpline(self.pts, c[:npts], k=self.order, s=0),
                "splt": UnivariateSpline(self.pts, c[npts:], k=self.order, s=0),
            }

            if self.normal == "X":
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
            X = self.mesh.gridCC[:, 0]
            Y = self.mesh.gridCC[:, 1]

            if self.normal == "X":
                f = self.spl(Y) - X
            elif self.normal == "Y":
                f = self.spl(X) - Y
            else:
                raise (Exception("Input for normal = X or Y or Z"))
        # 3D
        elif self.mesh.dim == 3:
            X = self.mesh.gridCC[:, 0]
            Y = self.mesh.gridCC[:, 1]
            Z = self.mesh.gridCC[:, 2]

            if self.normal == "X":
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
            if self.normal == "Y":
                # Here we use perturbation to compute sensitivity
                # TODO: bit more generalization of this ...
                # Modfications for X and Z directions ...
                for i in range(np.size(self.pts)):
                    ctemp = c[i]
                    ind = np.argmin(abs(self.mesh.vectorCCy - ctemp))
                    ca = c.copy()
                    cb = c.copy()
                    dy = self.mesh.hy[ind] * 1.5
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
            if self.normal == "X":
                # Here we use perturbation to compute sensitivity
                for i in range(self.npts * 2):
                    ctemp = c[i]
                    ind = np.argmin(abs(self.mesh.vectorCCy - ctemp))
                    ca = c.copy()
                    cb = c.copy()
                    dy = self.mesh.hy[ind] * 1.5
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


class BaseParametric(IdentityMap):

    slopeFact = 1  # will be scaled by the mesh.
    slope = None
    indActive = None

    def __init__(self, mesh, **kwargs):
        super(BaseParametric, self).__init__(mesh, **kwargs)

        if self.slope is None:
            self.slope = self.slopeFact / np.hstack(self.mesh.h).min()

    @property
    def x(self):
        if getattr(self, "_x", None) is None:
            if self.mesh.dim == 1:
                self._x = [
                    self.mesh.gridCC
                    if self.indActive is None
                    else self.mesh.gridCC[self.indActive]
                ][0]
            else:
                self._x = [
                    self.mesh.gridCC[:, 0]
                    if self.indActive is None
                    else self.mesh.gridCC[self.indActive, 0]
                ][0]
        return self._x

    @property
    def y(self):
        if getattr(self, "_y", None) is None:
            if self.mesh.dim > 1:
                self._y = [
                    self.mesh.gridCC[:, 1]
                    if self.indActive is None
                    else self.mesh.gridCC[self.indActive, 1]
                ][0]
            else:
                self._y = None
        return self._y

    @property
    def z(self):
        if getattr(self, "_z", None) is None:
            if self.mesh.dim > 2:
                self._z = [
                    self.mesh.gridCC[:, 2]
                    if self.indActive is None
                    else self.mesh.gridCC[self.indActive, 2]
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
        return (1.0 / (1 + x ** 2)) / np.pi * dx


class ParametricLayer(BaseParametric):
    """
    Parametric Layer Space

    .. code:: python

        m = [
            val_background,
            val_layer,
            layer_center,
            layer_thickness
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
        super(ParametricLayer, self).__init__(mesh, **kwargs)

    @property
    def nP(self):
        return 4

    @property
    def shape(self):
        if self.indActive is not None:
            return (sum(self.indActive), self.nP)
        return (self.mesh.nC, self.nP)

    def mDict(self, m):
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
    """
    Parametric Block in a Homogeneous Space

    For 1D:

    .. code:: python

        m = [
            val_background,
            val_block,
            block_x0,
            block_dx,
        ]

    For 2D:

    .. code:: python

        m = [
            val_background,
            val_block,
            block_x0,
            block_dx,
            block_y0,
            block_dy
        ]

    For 3D:

    .. code:: python

        m = [
            val_background,
            val_block,
            block_x0,
            block_dx,
            block_y0,
            block_dy
            block_z0,
            block_dz
        ]

    **Required**

    :param discretize.base.BaseMesh mesh: SimPEG Mesh, 2D or 3D

    **Optional**

    :param float slopeFact: arctan slope factor - divided by the minimum h
                            spacing to give the slope of the arctan
                            functions
    :param float slope: slope of the arctan function
    :param numpy.ndarray indActive: bool vector with active indices

    """

    epsilon = properties.Float(
        "epsilon value used in the ekblom representation of the block", default=1e-6
    )

    p = properties.Float(
        "p-value used in the ekblom representation of the block", default=10
    )

    def __init__(self, mesh, **kwargs):
        super(ParametricBlock, self).__init__(mesh, **kwargs)

    @property
    def nP(self):
        if self.mesh.dim == 1:
            return 4
        if self.mesh.dim == 2:
            return 6
        elif self.mesh.dim == 3:
            return 8

    @property
    def shape(self):
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
        return getattr(self, "_mDict{}d".format(self.mesh.dim))(m)

    def _ekblom(self, val):
        return (val ** 2 + self.epsilon ** 2) ** (self.p / 2.0)

    def _ekblomDeriv(self, val):
        return (
            (self.p / 2)
            * (val ** 2 + self.epsilon ** 2) ** ((self.p / 2) - 1)
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
            * (self._ekblomDeriv((x - x0) / (0.5 * dx)) * (-(x - x0) / (0.5 * dx ** 2)))
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
        return sp.csr_matrix(
            getattr(self, "_deriv{}D".format(self.mesh.dim))(self.mDict(m))
        )


class ParametricEllipsoid(ParametricBlock):

    # """
    #     Parametric Ellipsoid in a Homogeneous Space

    #     **Required**

    #     :param discretize.base.BaseMesh mesh: SimPEG Mesh, 2D or 3D

    #     **Optional**

    #     :param float slopeFact: arctan slope factor - divided by the minimum h
    #                             spacing to give the slope of the arctan
    #                             functions
    #     :param float slope: slope of the arctan function
    #     :param numpy.ndarray indActive: bool vector with active indices

    # """

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

        super(ParametricCasingAndLayer, self).__init__(mesh, **kwargs)

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

        super(ParametricBlockInLayer, self).__init__(mesh, **kwargs)

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

    tol = 1e-8  # Tolerance to avoid zero division
    components = 1  # Number of components in the model. =3 for vector model

    def __init__(self, global_mesh, global_active, local_mesh, **kwargs):
        """
        Parameters
        ----------
        global_mesh : discretize.TreeMesh
            Global TreeMesh defining the entire domain.
        global_active : bool, array of bool, or array of indices
            Defines the active cells in the global_mesh.
        local_mesh : discretize.TreeMesh
            Local TreeMesh for the simulation.
        """
        kwargs.pop("mesh", None)
        if global_mesh._meshType != "TREE":
            raise ValueError("global_mesh must be a TreeMesh")
        if local_mesh._meshType != "TREE":
            raise ValueError("local_mesh must be a TreeMesh")

        super(TileMap, self).__init__(**kwargs)
        self.global_mesh = global_mesh
        self.global_active = global_active
        self.local_mesh = local_mesh

        if not isinstance(self.global_active, bool):
            temp = np.zeros(self.global_mesh.nC, dtype="bool")
            temp[self.global_active] = True
            self.global_active = temp

        self.P

    @property
    def local_active(self):
        """
        This is the local_active of the global_active used in the global problem.
        """
        return getattr(self, "_local_active", None)

    @local_active.setter
    def local_active(self, local_active):

        if not isinstance(local_active, bool):
            temp = np.zeros(self.local_mesh.nC, dtype="bool")
            temp[local_active] = True
            local_active = temp

        self._local_active = local_active

    @property
    def P(self):
        """
        Set the projection matrix with partial volumes
        """
        if getattr(self, "_P", None) is None:

            in_local = self.local_mesh._get_containing_cell_indexes(
                self.global_mesh.gridCC
            )

            P = (
                sp.csr_matrix(
                    (self.global_mesh.vol, (in_local, np.arange(self.global_mesh.nC))),
                    shape=(self.local_mesh.nC, self.global_mesh.nC),
                )
                * speye(self.global_mesh.nC)[:, self.global_active]
            )

            self.local_active = mkvc(np.sum(P, axis=1) > 0)

            P = P[self.local_active, :]

            self._P = sp.block_diag(
                [
                    sdiag(1.0 / self.local_mesh.vol[self.local_active]) * P
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

    """

    def __init__(
        self,
        coeffxx=np.r_[0.0, 1],
        coeffxy=np.zeros(1),
        coeffyx=np.zeros(1),
        coeffyy=np.r_[0.0, 1],
        mesh=None,
        nP=None,
        **kwargs
    ):

        self.coeffxx = coeffxx
        self.coeffxy = coeffxy
        self.coeffyx = coeffyx
        self.coeffyy = coeffyy
        self.polynomialxx = polynomial.Polynomial(self.coeffxx)
        self.polynomialxy = polynomial.Polynomial(self.coeffxy)
        self.polynomialyx = polynomial.Polynomial(self.coeffyx)
        self.polynomialyy = polynomial.Polynomial(self.coeffyy)
        self.polynomialxx_deriv = self.polynomialxx.deriv(m=1)
        self.polynomialxy_deriv = self.polynomialxy.deriv(m=1)
        self.polynomialyx_deriv = self.polynomialyx.deriv(m=1)
        self.polynomialyy_deriv = self.polynomialyy.deriv(m=1)

        super(PolynomialPetroClusterMap, self).__init__(mesh=mesh, nP=nP, **kwargs)

    def _transform(self, m):
        out = m.copy()
        out[:, 0] = self.polynomialxx(m[:, 0]) + self.polynomialxy(m[:, 1])
        out[:, 1] = self.polynomialyx(m[:, 0]) + self.polynomialyy(m[:, 1])
        return out

    def inverse(self, D):
        """
        :param numpy.array D: physical property
        :rtype: numpy.array
        :return: model

        The *transformInverse* changes the physical property into the
        model.

        .. math::

            m = \log{\sigma}

        """
        raise Exception("Not implemented")

    def _derivmatrix(self, m):
        return np.r_[
            [
                [
                    self.polynomialxx_deriv(m[:, 0])[0],
                    self.polynomialyx_deriv(m[:, 0])[0],
                ],
                [
                    self.polynomialxy_deriv(m[:, 1])[0],
                    self.polynomialyy_deriv(m[:, 1])[0],
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


###############################################################################
#                                                                             #
#                              Deprecated Maps                               #
#                                                                             #
###############################################################################


@deprecate_class(removal_version="0.16.0", future_warn=True)
class FullMap(SurjectFull):
    pass


@deprecate_class(removal_version="0.16.0", future_warn=True)
class Vertical1DMap(SurjectVertical1D):
    pass


@deprecate_class(removal_version="0.16.0", future_warn=True)
class Map2Dto3D(Surject2Dto3D):
    pass


@deprecate_class(removal_version="0.16.0", future_warn=True)
class ActiveCells(InjectActiveCells):
    pass


@deprecate_class(removal_version="0.16.0", future_warn=True)
class CircleMap(ParametricCircleMap):
    pass


@deprecate_class(removal_version="0.16.0", future_warn=True)
class PolyMap(ParametricPolyMap):
    pass


@deprecate_class(removal_version="0.16.0", future_warn=True)
class SplineMap(ParametricSplineMap):
    pass
