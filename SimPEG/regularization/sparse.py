import numpy as np
import scipy.sparse as sp
import warnings
import properties

from .base import BaseRegularization, BaseComboRegularization
from .. import utils


class BaseSparse(BaseRegularization):
    """
    Base class for building up the components of the Sparse Regularization
    """

    def __init__(self, mesh, **kwargs):
        self._stashedR = None
        super(BaseSparse, self).__init__(mesh=mesh, **kwargs)

    model = properties.Array("current model", dtype=float)

    epsilon = properties.Float(
        "Threshold value for the model norm", default=1e-3, required=True
    )

    norm = properties.Array("norm used", dtype=float)

    space = properties.String("By default inherit the objctive", default="linear")

    gradientType = properties.String("type of gradient", default="components")

    scale = properties.Array("General nob for scaling", dtype=float,)

    # Give the option to scale or not
    scaledIRLS = properties.Bool("Scale the gradients of the IRLS norms", default=True)

    @properties.validator("scale")
    def _validate_scale(self, change):
        if change["value"] is not None:
            # todo: residual size? we need to know the expected end shape
            if self._nC_residual != "*":
                assert (
                    len(change["value"]) == self._nC_residual
                ), "scale must be length {} not {}".format(
                    self._nC_residual, len(change["value"])
                )

    @property
    def stashedR(self):
        return self._stashedR

    @stashedR.setter
    def stashedR(self, value):
        self._stashedR = value


class SparseSmall(BaseSparse):
    """
    Sparse smallness regularization

    **Inputs**

    :param int norm: norm on the smallness
    """

    _multiplier_pair = "alpha_s"

    def __init__(self, mesh, **kwargs):
        super(SparseSmall, self).__init__(mesh=mesh, **kwargs)

    # Give the option to scale or not
    scaledIRLS = properties.Bool("Scale the gradients of the IRLS norms", default=True)

    @property
    def f_m(self):

        return self.mapping * self._delta_m(self.model)

    @property
    def W(self):
        if getattr(self, "model", None) is None:
            R = utils.speye(self.mapping.shape[0])
        else:
            r = self.R(self.f_m)
            R = utils.sdiag(r)

        if self.scale is None:
            self.scale = np.ones(self.mapping.shape[0])

        if self.cell_weights is not None:
            return utils.sdiag((self.scale * self.cell_weights) ** 0.5) * R

        else:
            return utils.sdiag((self.scale * self.regmesh.vol) ** 0.5) * R

    def R(self, f_m):
        # if R is stashed, return that instead
        if getattr(self, "stashedR") is not None:
            return self.stashedR

        # Default
        eta = np.ones_like(f_m)

        if self.scaledIRLS:
            # Eta scaling is important for mix-norms...do not mess with it
            # Scale on l2-norm gradient: f_m.max()
            maxVal = np.ones_like(f_m) * np.abs(f_m).max()

            # Compute theoritical maximum gradients for p < 1
            maxVal[self.norm < 1] = self.epsilon / np.sqrt(
                1.0 - self.norm[self.norm < 1]
            )
            maxGrad = maxVal / (maxVal ** 2.0 + self.epsilon ** 2.0) ** (
                1.0 - self.norm / 2.0
            )
            # Scaling factor
            eta[maxGrad != 0] = np.abs(f_m).max() / maxGrad[maxGrad != 0]

        # Scaled IRLS weights
        r = (eta / (f_m ** 2.0 + self.epsilon ** 2.0) ** (1.0 - self.norm / 2.0)) ** 0.5

        self.stashedR = r  # stash on the first calculation
        return r

    @utils.timeIt
    def deriv(self, m):
        """

        The regularization is:

        .. math::

            R(m) = \\frac{1}{2}\mathbf{(m-m_\\text{ref})^\\top W^\\top
                   W(m-m_\\text{ref})}

        So the derivative is straight forward:

        .. math::

            R(m) = \mathbf{W^\\top W (m-m_\\text{ref})}

        """

        mD = self.mapping.deriv(self._delta_m(m))
        r = self.W * (self.mapping * (self._delta_m(m)))
        return mD.T * (self.W.T * r)


class SparseDeriv(BaseSparse):
    """
    Base Class for sparse regularization on first spatial derivatives
    """

    def __init__(self, mesh, orientation="x", **kwargs):
        self.orientation = orientation
        super(SparseDeriv, self).__init__(mesh=mesh, **kwargs)

    mrefInSmooth = properties.Bool(
        "include mref in the smoothness calculation?", default=False
    )

    # Give the option to scale or not
    scaledIRLS = properties.Bool("Scale the gradients of the IRLS norms", default=True)

    @utils.timeIt
    def __call__(self, m):
        """
        We use a weighted 2-norm objective function

        .. math::

            r(m) = \\frac{1}{2}
        """
        if self.mrefInSmooth:

            f_m = self._delta_m(m)

        else:
            f_m = m
        if self.scale is None:
            self.scale = np.ones(self.mapping.shape[0])

        if self.space == "spherical":
            Ave = getattr(self.regmesh, "aveCC2F{}".format(self.orientation))

            if getattr(self, "model", None) is None:
                R = utils.speye(self.cellDiffStencil.shape[0])

            else:
                r = self.R(self.f_m)
                R = utils.sdiag(r)

            if self.cell_weights is not None:
                W = utils.sdiag((Ave * (self.scale * self.cell_weights)) ** 0.5) * R

            else:
                W = utils.sdiag((Ave * (self.scale * self.regmesh.vol)) ** 0.5) * R

            theta = self.cellDiffStencil * (self.mapping * f_m)
            dmdx = utils.mat_utils.coterminal(theta)
            r = W * dmdx

        else:
            r = self.W * (self.mapping * f_m)

        return 0.5 * r.dot(r)

    def R(self, f_m):
        # if R is stashed, return that instead
        if getattr(self, "stashedR") is not None:
            return self.stashedR

        # Default
        eta = np.ones_like(f_m)

        if self.scaledIRLS:
            # Eta scaling is important for mix-norms...do not mess with it
            # Scale on l2-norm gradient: f_m.max()
            maxVal = np.ones_like(f_m) * np.abs(f_m).max()

            # Compute theoritical maximum gradients for p < 1
            maxVal[self.norm < 1] = self.epsilon / np.sqrt(
                1.0 - self.norm[self.norm < 1]
            )
            maxGrad = maxVal / (
                maxVal ** 2.0 + (self.epsilon * self.length_scales) ** 2.0
            ) ** (1.0 - self.norm / 2.0)

            # Scaling Factor
            eta[maxGrad != 0] = np.abs(f_m).max() / maxGrad[maxGrad != 0]

        # Scaled-IRLS weights
        r = (
            eta
            / (f_m ** 2.0 + (self.epsilon * self.length_scales) ** 2.0)
            ** (1.0 - self.norm / 2.0)
        ) ** 0.5
        self.stashedR = r  # stash on the first calculation
        return r

    @utils.timeIt
    def deriv(self, m):
        """

        The regularization is:

        .. math::

            R(m) = \\frac{1}{2}\mathbf{(m-m_\\text{ref})^\\top W^\\top
                   W(m-m_\\text{ref})}

        So the derivative is straight forward:

        .. math::

            R(m) = \mathbf{W^\\top W (m-m_\\text{ref})}

        """

        if self.mrefInSmooth:

            model = self._delta_m(m)

        else:
            model = m
        if self.scale is None:
            self.scale = np.ones(self.mapping.shape[0])

        if self.space == "spherical":
            Ave = getattr(self.regmesh, "aveCC2F{}".format(self.orientation))

            if getattr(self, "model", None) is None:
                R = utils.speye(self.cellDiffStencil.shape[0])

            else:
                r = self.R(self.f_m)
                R = utils.sdiag(r)

            if self.cell_weights is not None:
                W = utils.sdiag(((Ave * (self.scale * self.cell_weights))) ** 0.5) * R

            else:
                W = utils.sdiag((Ave * (self.scale * self.regmesh.vol)) ** 0.5) * R

            theta = self.cellDiffStencil * (self.mapping * model)
            dmdx = utils.mat_utils.coterminal(theta)

            r = W * dmdx

        else:
            r = self.W * (self.mapping * model)

        mD = self.mapping.deriv(model)
        return mD.T * (self.W.T * r)

    @property
    def _multiplier_pair(self):
        return "alpha_{orientation}".format(orientation=self.orientation)

    @property
    def f_m(self):

        if self.mrefInSmooth:

            f_m = self._delta_m(self.model)

        else:
            f_m = self.model

        if self.space == "spherical":
            theta = self.cellDiffStencil * (self.mapping * f_m)
            dmdx = utils.mat_utils.coterminal(theta)

        else:

            if self.gradientType == "total":
                Ave = getattr(self.regmesh, "aveCC2F{}".format(self.orientation))

                dmdx = np.abs(
                    self.regmesh.aveFx2CC
                    * self.regmesh.cellDiffxStencil
                    * (self.mapping * f_m)
                )

                if self.regmesh.dim > 1:

                    dmdx += np.abs(
                        self.regmesh.aveFy2CC
                        * self.regmesh.cellDiffyStencil
                        * (self.mapping * f_m)
                    )

                if self.regmesh.dim > 2:

                    dmdx += np.abs(
                        self.regmesh.aveFz2CC
                        * self.regmesh.cellDiffzStencil
                        * (self.mapping * f_m)
                    )

                dmdx = Ave * dmdx

            else:
                dmdx = self.cellDiffStencil * (self.mapping * f_m)

        return dmdx

    @property
    def cellDiffStencil(self):
        return utils.sdiag(self.length_scales) * getattr(
            self.regmesh, "cellDiff{}Stencil".format(self.orientation)
        )

    @property
    def W(self):

        Ave = getattr(self.regmesh, "aveCC2F{}".format(self.orientation))

        if getattr(self, "model", None) is None:
            R = utils.speye(self.cellDiffStencil.shape[0])

        else:
            r = self.R(self.f_m)
            R = utils.sdiag(r)
        if self.scale is None:
            self.scale = np.ones(self.mapping.shape[0])
        if self.cell_weights is not None:
            return (
                utils.sdiag((Ave * (self.scale * self.cell_weights)) ** 0.5)
                * R
                * self.cellDiffStencil
            )
        else:
            return (
                utils.sdiag((Ave * (self.scale * self.regmesh.vol)) ** 0.5)
                * R
                * self.cellDiffStencil
            )

    @property
    def length_scales(self):
        """
            Normalized cell based weighting

        """
        Ave = getattr(self.regmesh, "aveCC2F{}".format(self.orientation))

        if getattr(self, "_length_scales", None) is None:
            index = "xyz".index(self.orientation)

            length_scales = Ave * (
                self.regmesh.Pac.T * self.regmesh.mesh.h_gridded[:, index]
            )

            self._length_scales = length_scales.min() / length_scales

        return self._length_scales

    @length_scales.setter
    def length_scales(self, value):
        self._length_scales = value


class Sparse(BaseComboRegularization):
    """
    The regularization is:

    .. math::

        R(m) = \\frac{1}{2}\mathbf{(m-m_\\text{ref})^\\top W^\\top R^\\top R
        W(m-m_\\text{ref})}

    where the IRLS weight

    .. math::

        R = \eta TO FINISH LATER!!!

    So the derivative is straight forward:

    .. math::

        R(m) = \mathbf{W^\\top R^\\top R W (m-m_\\text{ref})}

    The IRLS weights are recomputed after each beta solves.
    It is strongly recommended to do a few Gauss-Newton iterations
    before updating.
    """

    def __init__(
        self, mesh, alpha_s=1.0, alpha_x=1.0, alpha_y=1.0, alpha_z=1.0, **kwargs
    ):

        objfcts = [
            SparseSmall(mesh=mesh, **kwargs),
            SparseDeriv(mesh=mesh, orientation="x", **kwargs),
        ]

        if mesh.dim > 1:
            objfcts.append(SparseDeriv(mesh=mesh, orientation="y", **kwargs))

        if mesh.dim > 2:
            objfcts.append(SparseDeriv(mesh=mesh, orientation="z", **kwargs))

        super(Sparse, self).__init__(
            mesh=mesh,
            objfcts=objfcts,
            alpha_s=alpha_s,
            alpha_x=alpha_x,
            alpha_y=alpha_y,
            alpha_z=alpha_z,
            **kwargs
        )

        # Utils.setKwargs(self, **kwargs)

    # Properties
    norms = properties.Array(
        "Norms used to create the sparse regularization",
        default=np.c_[2.0, 2.0, 2.0, 2.0],
        shape={("*", "*")},
    )

    eps_p = properties.Float("Threshold value for the model norm", required=True)

    eps_q = properties.Float(
        "Threshold value for the model gradient norm", required=True
    )

    model = properties.Array("current model", dtype=float)

    space = properties.String("type of model", default="linear")

    gradientType = properties.String("type of gradient", default="components")

    scales = properties.Array(
        "General nob for scaling", default=np.c_[1.0, 1.0, 1.0, 1.0], shape={("*", "*")}
    )
    # Give the option to scale or not
    scaledIRLS = properties.Bool("Scale the gradients of the IRLS norms", default=True)
    # Save the l2 result during the IRLS
    l2model = None

    @properties.validator("norms")
    def _validate_norms(self, change):
        if change["value"].shape[0] == 1:
            change["value"] = np.kron(
                np.ones((self.regmesh.Pac.shape[1], 1)), change["value"]
            )
        elif change["value"].shape[0] > 1:
            assert change["value"].shape[0] == self.regmesh.Pac.shape[1], (
                "Vector of norms must be the size"
                " of active model parameters ({})"
                "The provided vector has length "
                "{}".format(self.regmesh.Pac.shape[0], len(change["value"]))
            )

    # Observers
    @properties.observer("norms")
    def _mirror_norms_to_objfcts(self, change):

        self.objfcts[0].norm = change["value"][:, 0]
        for i, objfct in enumerate(self.objfcts[1:]):
            Ave = getattr(objfct.regmesh, "aveCC2F{}".format(objfct.orientation))
            objfct.norm = Ave * change["value"][:, i + 1]

    @properties.observer("model")
    def _mirror_model_to_objfcts(self, change):
        for objfct in self.objfcts:
            objfct.model = change["value"]

    @properties.observer("eps_p")
    def _mirror_eps_p_to_smallness(self, change):
        for objfct in self.objfcts:
            if isinstance(objfct, SparseSmall):
                objfct.epsilon = change["value"]

    @properties.observer("eps_q")
    def _mirror_eps_q_to_derivs(self, change):
        for objfct in self.objfcts:
            if isinstance(objfct, SparseDeriv):
                objfct.epsilon = change["value"]

    @properties.observer("space")
    def _mirror_space_to_objfcts(self, change):
        for objfct in self.objfcts:
            objfct.space = change["value"]

    @properties.observer("gradientType")
    def _mirror_gradientType_to_objfcts(self, change):
        for objfct in self.objfcts:
            objfct.gradientType = change["value"]

    @properties.observer("scaledIRLS")
    def _mirror_scaledIRLS_to_objfcts(self, change):
        for objfct in self.objfcts:
            objfct.scaledIRLS = change["value"]

    @properties.validator("scales")
    def _validate_scales(self, change):
        if change["value"].shape[0] == 1:
            change["value"] = np.kron(
                np.ones((self.regmesh.Pac.shape[1], 1)), change["value"]
            )
        elif change["value"].shape[0] > 1:
            assert change["value"].shape[0] == self.regmesh.Pac.shape[1], (
                "Vector of scales must be the size"
                " of active model parameters ({})"
                "The provided vector has length "
                "{}".format(self.regmesh.Pac.shape[0], len(change["value"]))
            )

    # Observers
    @properties.observer("scales")
    def _mirror_scale_to_objfcts(self, change):
        for i, objfct in enumerate(self.objfcts):
            objfct.scale = change["value"][:, i]
