import numpy as np
from .sparse import SparseSmoothness, SparseSmallness, Sparse
from .regularization_mesh_lateral import LCRegularizationMesh


class LaterallyConstrainedSmallness(SparseSmallness):
    """
    Duplicate of SparseSmallness Class
    """


class LaterallyConstrainedSmoothness(SparseSmoothness):
    """
    Modification of SparseSmoothness Class
    for addressing radial and vertical gradients of model parameters,
    which is a 1D vertical resistivity profile at each of lateral locations.
    """

    def __init__(self, mesh, orientation="r", gradient_type="total", **kwargs):
        if "gradientType" in kwargs:
            self.gradientType = kwargs.pop("gradientType")
        else:
            self.gradient_type = gradient_type
        super().__init__(mesh=mesh, orientation=orientation, **kwargs)


class LaterallyConstrained(Sparse):
    """
    This regularization function is designed to regularize model parameters
    connected with a 2D simplex mesh and 1D vertical mesh.
    Motivating example is a stitched inversion of the electromagnetic data.
    In such a case, a model is a 1D vertical conductivity (or resistivity) profile
    at each sounding location. Each profile has the same number of layers.
    The 2D simplex mesh connects resistivity values of each layer in lateral dimensions
    while the 1D vertical mesh connects resistivity values along the vertical profile.
    This LaterallyConstrained class is designed in a way that can handle sparse norm inversion.
    And that is the reason why it inherits the Sparse Class.

    """

    def __init__(
        self,
        mesh,
        active_cells=None,
        active_edges=None,
        alpha_r=None,
        length_scale_r=None,
        norms=None,
        gradient_type="total",
        irls_scaled=True,
        irls_threshold=1e-8,
        objfcts=None,
        **kwargs,
    ):
        if not isinstance(mesh, LCRegularizationMesh):
            mesh = LCRegularizationMesh(mesh)

        if not isinstance(mesh, LCRegularizationMesh):
            TypeError(
                f"'regularization_mesh' must be of type {LCRegularizationMesh}. "
                f"Value of type {type(mesh)} provided."
            )
        self._regularization_mesh = mesh
        if active_cells is not None:
            self._regularization_mesh.active_cells = active_cells
        if active_edges is not None:
            self._regularization_mesh.active_edges = active_edges

        if alpha_r is not None:
            if length_scale_r is not None:
                raise ValueError(
                    "Attempted to set both alpha_r and length_scale_r at the same time. Please "
                    "use only one of them"
                )
            self.alpha_r = alpha_r
        else:
            self.length_scale_r = length_scale_r

        if objfcts is None:
            objfcts = [
                SparseSmallness(mesh=self.regularization_mesh),
                SparseSmoothness(mesh=self.regularization_mesh, orientation="r"),
                SparseSmoothness(mesh=self.regularization_mesh, orientation="z"),
            ]

        super().__init__(
            self.regularization_mesh,
            objfcts=objfcts,
            **kwargs,
        )

    @property
    def alpha_r(self):
        """Multiplier constant for first-order smoothness along x.

        Returns
        -------
        float
            Multiplier constant for first-order smoothness along x.
        """
        return self._alpha_r

    @alpha_r.setter
    def alpha_r(self, value):
        try:
            value = float(value)
        except (ValueError, TypeError):
            raise TypeError(f"alpha_r must be a real number, saw type{type(value)}")
        if value < 0:
            raise ValueError(f"alpha_r must be non-negative, not {value}")
        self._alpha_r = value

    @property
    def length_scale_r(self):
        r"""Multiplier constant for smoothness along x relative to base scale length.

        Where the :math:`\Delta h` defines the base length scale (i.e. minimum cell dimension),
        and  :math:`\alpha_r` defines the multiplier constant for first-order smoothness along x,
        the length-scale is given by:

        .. math::
            L_x = \bigg ( \frac{\alpha_r}{\Delta h} \bigg )^{1/2}

        Returns
        -------
        float
            Multiplier constant for smoothness along x relative to base scale length.
        """
        return np.sqrt(self.alpha_r) / self.regularization_mesh.base_length

    @length_scale_r.setter
    def length_scale_r(self, value: float):
        if value is None:
            value = 1.0
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise TypeError(
                f"length_scale_r must be a real number, saw type{type(value)}"
            )
        print("Set alpha_s")
        self.alpha_r = (value * self.regularization_mesh.base_length) ** 2
