"""
Parametric map classes.
"""

import discretize
import numpy as np
from numpy.polynomial import polynomial
import scipy.sparse as sp
from scipy.interpolate import UnivariateSpline

from discretize.utils import sdiag

from ..utils import (
    validate_type,
    validate_ndarray_with_shape,
    validate_float,
    validate_integer,
    validate_string,
    validate_active_indices,
)
from ._base import IdentityMap


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

    >>> from simpeg.maps import ParametricCircleMap
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

    >>> from simpeg.maps import ParametricPolyMap, InjectActiveCells
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

    >>> from simpeg.maps import ParametricSplineMap
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
                    (
                        self.mesh.cell_centers
                        if self.indActive is None
                        else self.mesh.cell_centers[self.indActive]
                    )
                ][0]
            else:
                self._x = [
                    (
                        self.mesh.cell_centers[:, 0]
                        if self.indActive is None
                        else self.mesh.cell_centers[self.indActive, 0]
                    )
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
                    (
                        self.mesh.cell_centers[:, 1]
                        if self.indActive is None
                        else self.mesh.cell_centers[self.indActive, 1]
                    )
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
                    (
                        self.mesh.cell_centers[:, 2]
                        if self.indActive is None
                        else self.mesh.cell_centers[self.indActive, 2]
                    )
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

    >>> from simpeg.maps import ParametricLayer, InjectActiveCells
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

    >>> from simpeg.maps import ParametricBlock, InjectActiveCells
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
        return (self.p / 2) * (val**2 + self.epsilon**2) ** ((self.p / 2) - 1) * 2 * val

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

    >>> from simpeg.maps import ParametricEllipsoid, InjectActiveCells
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
