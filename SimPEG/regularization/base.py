import numpy as np
import scipy.sparse as sp
import warnings
import properties

from .. import props
from .. import maps
from ..objective_function import BaseObjectiveFunction, ComboObjectiveFunction
from .. import utils
from .regularization_mesh import RegularizationMesh

###############################################################################
#                                                                             #
#                          Base Regularization                                #
#                                                                             #
###############################################################################


class BaseRegularization(BaseObjectiveFunction):
    """
    Base class for regularization. Inherit this for building your own
    regularization. The base regularization assumes a weighted l2 style of
    regularization. However, if you wish to employ a different norm, the
    methods :meth:`__call__`, :meth:`deriv` and :meth:`deriv2` can be
    over-written

    :param discretize.base.BaseMesh mesh: SimPEG mesh

    """

    def __init__(self, mesh=None, **kwargs):
        super(BaseRegularization, self).__init__()
        self.regmesh = RegularizationMesh(mesh)
        if "indActive" in kwargs.keys():
            indActive = kwargs.pop("indActive")
            self.regmesh.indActive = indActive
        utils.setKwargs(self, **kwargs)

    counter = None

    # Properties
    mref = props.Array("reference model")
    indActive = properties.Array(
        "indices of active cells in the mesh", dtype=(bool, int)
    )
    cell_weights = properties.Array(
        "regularization weights applied at cell centers", dtype=float
    )
    regmesh = properties.Instance(
        "regularization mesh", RegularizationMesh, required=True
    )
    mapping = properties.Instance(
        "mapping which is applied to model in the regularization",
        maps.IdentityMap,
        default=maps.IdentityMap(),
    )

    # Observers and Validators
    @properties.validator("indActive")
    def _cast_to_bool(self, change):
        value = change["value"]
        if value is not None:
            if value.dtype != "bool":  # cast it to a bool otherwise
                tmp = value
                value = np.zeros(self.regmesh.nC, dtype=bool)
                value[tmp] = True
                change["value"] = value

        # update regmesh indActive
        if getattr(self, "regmesh", None) is not None:
            self.regmesh.indActive = utils.mkvc(value)

    @properties.observer("indActive")
    def _update_regmesh_indActive(self, change):
        # update regmesh indActive
        if getattr(self, "regmesh", None) is not None:
            self.regmesh.indActive = change["value"]

    @properties.validator("cell_weights")
    def _validate_cell_weights(self, change):
        if change["value"] is not None:
            # todo: residual size? we need to know the expected end shape
            if self._nC_residual != "*":
                assert (
                    len(change["value"]) == self._nC_residual
                ), "cell_weights must be length {} not {}".format(
                    self._nC_residual, len(change["value"])
                )

    # Other properties and methods
    @property
    def nP(self):
        """
        number of model parameters
        """
        if getattr(self.mapping, "nP") != "*":
            return self.mapping.nP
        elif getattr(self.regmesh, "nC") != "*":
            return self.regmesh.nC
        else:
            return "*"

    @property
    def _nC_residual(self):
        """
        Shape of the residual
        """

        nC = getattr(self.regmesh, "nC", None)
        mapping = getattr(self, "mapping", None)

        if nC != "*" and nC is not None:
            return self.regmesh.nC
        elif mapping is not None and mapping.shape[0] != "*":
            return self.mapping.shape[0]
        else:
            return self.nP

    def _delta_m(self, m):
        if self.mref is None:
            return m
        return -self.mref + m  # in case self.mref is Zero, returns type m

    @utils.timeIt
    def __call__(self, m):
        """
        We use a weighted 2-norm objective function

        .. math::

            r(m) = \\frac{1}{2}
        """
        r = self.W * (self.mapping * (self._delta_m(m)))
        return 0.5 * r.dot(r)

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

    @utils.timeIt
    def deriv2(self, m, v=None):
        """
        Second derivative

        :param numpy.ndarray m: geophysical model
        :param numpy.ndarray v: vector to multiply
        :rtype: scipy.sparse.csr_matrix
        :return: WtW, or if v is supplied WtW*v (numpy.ndarray)

        The regularization is:

        .. math::

            R(m) = \\frac{1}{2}\mathbf{(m-m_\\text{ref})^\\top W^\\top
            W(m-m_\\text{ref})}

        So the second derivative is straight forward:

        .. math::

            R(m) = \\mathbf{W^\\top W}

        """

        mD = self.mapping.deriv(self._delta_m(m))
        if v is None:
            return mD.T * self.W.T * self.W * mD

        return mD.T * (self.W.T * (self.W * (mD * v)))


###############################################################################
#                                                                             #
#                        Base Combo Regularization                            #
#                                                                             #
###############################################################################


class BaseComboRegularization(ComboObjectiveFunction):
    def __init__(self, mesh, objfcts=[], **kwargs):

        super(BaseComboRegularization, self).__init__(objfcts=objfcts, multipliers=None)
        self.regmesh = RegularizationMesh(mesh)
        if "indActive" in kwargs.keys():
            indActive = kwargs.pop("indActive")
            self.regmesh.indActive = indActive
        utils.setKwargs(self, **kwargs)

        # link these attributes
        linkattrs = ["regmesh", "indActive", "cell_weights", "mapping"]

        for attr in linkattrs:
            val = getattr(self, attr)
            if val is not None:
                [setattr(fct, attr, val) for fct in self.objfcts]

    # Properties
    alpha_s = props.Float("smallness weight")
    alpha_x = props.Float("weight for the first x-derivative")
    alpha_y = props.Float("weight for the first y-derivative")
    alpha_z = props.Float("weight for the first z-derivative")
    alpha_xx = props.Float("weight for the second x-derivative")
    alpha_yy = props.Float("weight for the second y-derivative")
    alpha_zz = props.Float("weight for the second z-derivative")

    counter = None

    mref = props.Array("reference model")
    mrefInSmooth = properties.Bool(
        "include mref in the smoothness calculation?", default=False
    )
    indActive = properties.Array(
        "indices of active cells in the mesh", dtype=(bool, int)
    )
    cell_weights = properties.Array(
        "regularization weights applied at cell centers", dtype=float
    )
    scale = properties.Float("function scaling applied inside the norm", default=1.0)
    regmesh = properties.Instance(
        "regularization mesh", RegularizationMesh, required=True
    )
    mapping = properties.Instance(
        "mapping which is applied to model in the regularization",
        maps.IdentityMap,
        default=maps.IdentityMap(),
    )

    # Other properties and methods
    @property
    def nP(self):
        """
        number of model parameters
        """
        if getattr(self.mapping, "nP") != "*":
            return self.mapping.nP
        elif getattr(self.regmesh, "nC") != "*":
            return self.regmesh.nC
        else:
            return "*"

    @property
    def _nC_residual(self):
        """
        Shape of the residual
        """
        nC = getattr(self.regmesh, "nC", None)
        mapping = getattr(self, "mapping", None)

        if nC != "*" and nC is not None:
            return self.regmesh.nC
        elif mapping is not None and mapping.shape[0] != "*":
            return self.mapping.shape[0]
        else:
            return self.nP

    def _delta_m(self, m):
        if self.mref is None:
            return m
        return -self.mref + m  # in case self.mref is Zero, returns type m

    @property
    def multipliers(self):
        """
        Factors that multiply the objective functions that are summed together
        to build to composite regularization
        """
        return [
            getattr(self, "{alpha}".format(alpha=objfct._multiplier_pair))
            for objfct in self.objfcts
        ]

    # Observers and Validators
    @properties.validator("indActive")
    def _cast_to_bool(self, change):
        value = change["value"]
        if value is not None:
            if value.dtype != "bool":  # cast it to a bool otherwise
                tmp = value
                value = np.zeros(self.regmesh.nC, dtype=bool)
                value[tmp] = True
                change["value"] = value

        # update regmesh indActive
        if getattr(self, "regmesh", None) is not None:
            self.regmesh.indActive = utils.mkvc(value)

    @properties.observer("indActive")
    def _update_regmesh_indActive(self, change):
        # update regmesh indActive
        if getattr(self, "regmesh", None) is not None:
            self.regmesh.indActive = change["value"]

    @properties.validator("cell_weights")
    def _validate_cell_weights(self, change):
        if change["value"] is not None:
            # todo: residual size? we need to know the expected end shape
            if self._nC_residual != "*":
                assert (
                    len(change["value"]) == self._nC_residual
                ), "cell_weights must be length {} not {}".format(
                    self._nC_residual, len(change["value"])
                )

    @properties.observer("mref")
    def _mirror_mref_to_objfctlist(self, change):
        for fct in self.objfcts:
            if getattr(fct, "mrefInSmooth", None) is not None:
                if self.mrefInSmooth is False:
                    fct.mref = utils.Zero()
                else:
                    fct.mref = change["value"]
            else:
                fct.mref = change["value"]

    @properties.observer("mrefInSmooth")
    def _mirror_mrefInSmooth_to_objfctlist(self, change):
        for fct in self.objfcts:
            if getattr(fct, "mrefInSmooth", None) is not None:
                fct.mrefInSmooth = change["value"]

    @properties.observer("indActive")
    def _mirror_indActive_to_objfctlist(self, change):
        value = change["value"]
        if value is not None:
            if value.dtype != "bool":
                tmp = value
                value = np.zeros(self.mesh.nC, dtype=bool)
                value[tmp] = True
                change["value"] = value

        if getattr(self, "regmesh", None) is not None:
            self.regmesh.indActive = value

        for fct in self.objfcts:
            fct.indActive = value

    @properties.observer("cell_weights")
    def _mirror_cell_weights_to_objfctlist(self, change):
        for fct in self.objfcts:
            fct.cell_weights = change["value"]

    @properties.observer("mapping")
    def _mirror_mapping_to_objfctlist(self, change):
        for fct in self.objfcts:
            fct.mapping = change["value"]
