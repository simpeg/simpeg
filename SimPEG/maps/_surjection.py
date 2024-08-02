"""
Surjection map classes.
"""

import discretize
import numpy as np
import scipy.sparse as sp
from discretize import TensorMesh, CylindricalMesh
from discretize.utils import mkvc

from ..utils import (
    validate_type,
    validate_ndarray_with_shape,
    validate_string,
    validate_active_indices,
)
from ._base import IdentityMap


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

    >>> from simpeg.maps import SurjectVertical1D
    >>> from simpeg.utils import plot_1d_layer_model
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

    >>> from simpeg.maps import Surject2Dto3D
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

    >>> from simpeg.maps import SurjectUnits
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
