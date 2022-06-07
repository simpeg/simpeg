import scipy as sp
import numpy as np
from .sparse import SparseDeriv, SparseSmall, Sparse
from .. import utils
import properties
from .. import props

class LaterallyConstrainedSmall(SparseSmall):
    """
    Duplicate of Sparse smallness regularization

    """

    _multiplier_pair = "alpha_s"

    def __init__(self, mesh, **kwargs):
       super(LaterallyConstrainedSmall, self).__init__(mesh=mesh, **kwargs)


class LaterallyConstrainedDeriv(SparseDeriv):
    """
    Modification of SparseDeriv Class
    for addressing radial and vertical gradients of model parameters,
    which is a 1D vertical resistivity profile at each of lateral locations.
    """

    def __init__(self, mesh, **kwargs):
        super(LaterallyConstrainedDeriv, self).__init__(mesh=mesh, **kwargs)

    @property
    def W(self):

        gradient = getattr(self.regmesh, "gradient_{}".format(self.orientation))

        if getattr(self, "model", None) is None:
            R = utils.speye(gradient.shape[0])

        else:
            r = self.R(self.f_m)
            R = utils.sdiag(r)

        if self.scale is None:
            self.scale = np.ones(self.mapping.shape[0])

        weights = self.scale * self.regmesh.vol

        if self.cell_weights is not None:
            weights *= self.cell_weights
        gradient = getattr(self.regmesh, "gradient_{}".format(self.orientation))
        average = getattr(self.regmesh, "average_{}".format(self.orientation))
        return utils.sdiag((average * weights ** 0.5)) * R * gradient

    @property
    def f_m(self):

        if self.mrefInSmooth:

            f_m = self._delta_m(self.model)

        else:
            f_m = self.model

        # Not sure how effective it is
        if self.gradientType == "total":

            average = getattr(self.regmesh, "average_{}".format(self.orientation))

            dm_dx = np.abs(
                self.regmesh.aveE2N
                * self.regmesh.gradient_r
                * (self.mapping * f_m)
            )

            dm_dx += np.abs(
                self.regmesh.aveFz2CC
                * self.regmesh.gradient_z
                * (self.mapping * f_m)
            )

            dm_dx = average * dm_dx

        else:
            gradient = getattr(self.regmesh, "gradient_{}".format(self.orientation))
            dm_dx = gradient * (self.mapping * f_m)

        return dm_dx

    @property
    def length_scales(self):
        """
        Normalized cell based weighting

        """
        average = getattr(self.regmesh, "average_{}".format(self.orientation))

        if getattr(self, "_length_scales", None) is None:
            if self.orientation == 'r':
                # removing the length scale for the radial component seems better
                # length_scales = average * self.regmesh.h_gridded_r
                length_scales = np.ones(average.shape[0], dtype=float)
            elif self.orientation == 'z':
                length_scales = average * self.regmesh.h_gridded_z
            self._length_scales = length_scales.min() / length_scales

        return self._length_scales


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
        self, mesh, alpha_s=1.0, alpha_r=1.0, alpha_z=1.0, **kwargs
    ):
        objfcts = [
            LaterallyConstrainedSmall(mesh=mesh, **kwargs),
            LaterallyConstrainedDeriv(mesh=mesh, orientation="r", **kwargs),
            LaterallyConstrainedDeriv(mesh=mesh, orientation="z", **kwargs),
        ]
        # Inherits the upper level class of Sparse
        self.alpha_r = alpha_r

        super(Sparse, self).__init__(
            mesh=mesh,
            objfcts=objfcts,
            alpha_s=alpha_s,
            alpha_z=alpha_z,
            **kwargs
        )

    alpha_r = props.Float("weight for the first radial-derivative")
    # Observers
    @properties.observer("norms")
    def _mirror_norms_to_objfcts(self, change):
        self.objfcts[0].norm = change["value"][:, 0]
        for i, objfct in enumerate(self.objfcts[1:]):
            ave_cc_f = getattr(objfct.regmesh, "average_{}".format(objfct.orientation))
            objfct.norm = ave_cc_f * change["value"][:, i + 1]