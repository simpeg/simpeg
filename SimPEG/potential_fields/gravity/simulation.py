from SimPEG.utils import mkvc, sdiag, setKwargs
from SimPEG import props
from ...simulation import BaseSimulation
from ...base import BasePDESimulation
from ..base import BasePFSimulation, BaseEquivalentSourceLayerSimulation
from .survey import Survey
import scipy.constants as constants
from scipy.constants import G as NewtG
import numpy as np


class Simulation3DIntegral(BasePFSimulation):
    """
    Gravity simulation in integral form.

    """

    rho, rhoMap, rhoDeriv = props.Invertible("Physical property", default=1.0)

    def __init__(self, mesh, **kwargs):
        super().__init__(mesh, **kwargs)
        self._G = None
        self._gtg_diagonal = None
        self.modelMap = self.rhoMap

    def fields(self, m):
        self.model = m

        if self.store_sensitivities == "forward_only":
            self.model = m
            # Compute the linear operation without forming the full dense G
            fields = mkvc(self.linear_operator())
        else:
            fields = self.G @ (self.rhoMap @ m).astype(np.float32)

        return np.asarray(fields)

    def getJtJdiag(self, m, W=None):
        """
        Return the diagonal of JtJ
        """
        self.model = m

        if W is None:
            W = np.ones(self.survey.nD)
        else:
            W = W.diagonal() ** 2
        if getattr(self, "_gtg_diagonal", None) is None:

            diag = np.zeros(self.G.shape[1])
            for i in range(len(W)):
                diag += W[i] * (self.G[i] * self.G[i])
            self._gtg_diagonal = diag
        else:
            diag = self._gtg_diagonal
        return mkvc((sdiag(np.sqrt(diag)) @ self.rhoDeriv).power(2).sum(axis=0))

    def getJ(self, m, f=None):
        """
        Sensitivity matrix
        """
        return self.G.dot(self.rhoDeriv)

    def Jvec(self, m, v, f=None):
        """
        Sensitivity times a vector
        """
        dmu_dm_v = self.rhoDeriv @ v
        return self.G @ dmu_dm_v.astype(np.float32)

    def Jtvec(self, m, v, f=None):
        """
        Sensitivity transposed times a vector
        """
        Jtvec = self.G.T @ v.astype(np.float32)
        return np.asarray(self.rhoDeriv.T @ Jtvec)

    @property
    def G(self):
        """
        Gravity forward operator
        """
        if getattr(self, "_G", None) is None:
            self._G = self.linear_operator()

        return self._G

    @property
    def gtg_diagonal(self):
        """
        Diagonal of GtG
        """
        if getattr(self, "_gtg_diagonal", None) is None:

            return None

        return self._gtg_diagonal

    def evaluate_integral(self, receiver_location, components, tolerance=1e-4):
        """
        Compute the forward linear relationship between the model and the physics at a point
        and for all components of the survey.

        :param numpy.ndarray receiver_location:  array with shape (n_receivers, 3)
            Array of receiver locations as x, y, z columns.
        :param list[str] components: List of gravity components chosen from:
            'gx', 'gy', 'gz', 'gxx', 'gxy', 'gxz', 'gyy', 'gyz', 'gzz', 'guv'
        :param float tolerance: Small constant to avoid singularity near nodes and edges.
        :rtype numpy.ndarray: rows
        :returns: ndarray with shape (n_components, n_cells)
            Dense array mapping of the contribution of all active cells to data components::

                rows =
                    g_1 = [g_1x g_1y g_1z]
                    g_2 = [g_2x g_2y g_2z]
                           ...
                    g_c = [g_cx g_cy g_cz]

        """
        # base cell dimensions
        min_hx, min_hy = self.mesh.h[0].min(), self.mesh.h[1].min()
        if len(self.mesh.h) < 3:
            # Allow for 2D quadtree representations by using a dummy cell height.
            # Actually cell heights will come from externally defined ``self.Zn``
            min_hz = np.minimum(min_hx, min_hy) / 10.0
        else:
            min_hz = self.mesh.h[2].min()

        # comp. pos. differences for tne, bsw nodes. Adjust if location within
        # tolerance of a node or edge
        dx = self.Xn - receiver_location[0]
        dx[np.abs(dx) / min_hx < tolerance] = tolerance * min_hx
        dy = self.Yn - receiver_location[1]
        dy[np.abs(dy) / min_hy < tolerance] = tolerance * min_hy
        dz = self.Zn - receiver_location[2]
        dz[np.abs(dz) / min_hz < tolerance] = tolerance * min_hz

        rows = {component: np.zeros(self.Xn.shape[0]) for component in components}

        gxx = np.zeros(self.Xn.shape[0])
        gyy = np.zeros(self.Xn.shape[0])

        for aa in range(2):
            for bb in range(2):
                for cc in range(2):

                    r = (
                        mkvc(dx[:, aa]) ** 2
                        + mkvc(dy[:, bb]) ** 2
                        + mkvc(dz[:, cc]) ** 2
                    ) ** (0.50)

                    dz_r = dz[:, cc] + r
                    dy_r = dy[:, bb] + r
                    dx_r = dx[:, aa] + r

                    dxr = dx[:, aa] * r
                    dyr = dy[:, bb] * r
                    dzr = dz[:, cc] * r

                    dydz = dy[:, bb] * dz[:, cc]
                    dxdy = dx[:, aa] * dy[:, bb]
                    dxdz = dx[:, aa] * dz[:, cc]

                    if "gx" in components:
                        rows["gx"] += (
                            (-1) ** aa
                            * (-1) ** bb
                            * (-1) ** cc
                            * (
                                dy[:, bb] * np.log(dz_r)
                                + dz[:, cc] * np.log(dy_r)
                                - dx[:, aa] * np.arctan(dydz / dxr)
                            )
                        )

                    if "gy" in components:
                        rows["gy"] += (
                            (-1) ** aa
                            * (-1) ** bb
                            * (-1) ** cc
                            * (
                                dx[:, aa] * np.log(dz_r)
                                + dz[:, cc] * np.log(dx_r)
                                - dy[:, bb] * np.arctan(dxdz / dyr)
                            )
                        )

                    if "gz" in components:
                        rows["gz"] += (
                            (-1) ** aa
                            * (-1) ** bb
                            * (-1) ** cc
                            * (
                                dx[:, aa] * np.log(dy_r)
                                + dy[:, bb] * np.log(dx_r)
                                - dz[:, cc] * np.arctan(dxdy / dzr)
                            )
                        )

                    arg = dy[:, bb] * dz[:, cc] / dxr

                    if (
                        ("gxx" in components)
                        or ("gzz" in components)
                        or ("guv" in components)
                    ):
                        gxx -= (
                            (-1) ** aa
                            * (-1) ** bb
                            * (-1) ** cc
                            * (
                                dxdy / (r * dz_r)
                                + dxdz / (r * dy_r)
                                - np.arctan(arg)
                                + dx[:, aa]
                                * (1.0 / (1 + arg ** 2.0))
                                * dydz
                                / dxr ** 2.0
                                * (r + dx[:, aa] ** 2.0 / r)
                            )
                        )

                    if "gxy" in components:
                        rows["gxy"] -= (
                            (-1) ** aa
                            * (-1) ** bb
                            * (-1) ** cc
                            * (
                                np.log(dz_r)
                                + dy[:, bb] ** 2.0 / (r * dz_r)
                                + dz[:, cc] / r
                                - 1.0
                                / (1 + arg ** 2.0)
                                * (dz[:, cc] / r ** 2)
                                * (r - dy[:, bb] ** 2.0 / r)
                            )
                        )

                    if "gxz" in components:
                        rows["gxz"] -= (
                            (-1) ** aa
                            * (-1) ** bb
                            * (-1) ** cc
                            * (
                                np.log(dy_r)
                                + dz[:, cc] ** 2.0 / (r * dy_r)
                                + dy[:, bb] / r
                                - 1.0
                                / (1 + arg ** 2.0)
                                * (dy[:, bb] / (r ** 2))
                                * (r - dz[:, cc] ** 2.0 / r)
                            )
                        )

                    arg = dx[:, aa] * dz[:, cc] / dyr

                    if (
                        ("gyy" in components)
                        or ("gzz" in components)
                        or ("guv" in components)
                    ):
                        gyy -= (
                            (-1) ** aa
                            * (-1) ** bb
                            * (-1) ** cc
                            * (
                                dxdy / (r * dz_r)
                                + dydz / (r * dx_r)
                                - np.arctan(arg)
                                + dy[:, bb]
                                * (1.0 / (1 + arg ** 2.0))
                                * dxdz
                                / dyr ** 2.0
                                * (r + dy[:, bb] ** 2.0 / r)
                            )
                        )

                    if "gyz" in components:
                        rows["gyz"] -= (
                            (-1) ** aa
                            * (-1) ** bb
                            * (-1) ** cc
                            * (
                                np.log(dx_r)
                                + dz[:, cc] ** 2.0 / (r * (dx_r))
                                + dx[:, aa] / r
                                - 1.0
                                / (1 + arg ** 2.0)
                                * (dx[:, aa] / (r ** 2))
                                * (r - dz[:, cc] ** 2.0 / r)
                            )
                        )

        if "gyy" in components:
            rows["gyy"] = gyy

        if "gxx" in components:
            rows["gxx"] = gxx

        if "gzz" in components:
            rows["gzz"] = -gxx - gyy

        if "guv" in components:
            rows["guv"] = -0.5 * (gxx - gyy)

        for component in components:
            if len(component) == 3:
                rows[component] *= constants.G * 1e12  # conversion for Eotvos
            else:
                rows[component] *= constants.G * 1e8  # conversion for mGal

        return np.vstack([rows[component] for component in components])


class SimulationEquivalentSourceLayer(
    BaseEquivalentSourceLayerSimulation, Simulation3DIntegral
):
    """
    Equivalent source layer simulations

    Parameters
    ----------
    mesh : discretize.BaseMesh
        A 2D tensor or tree mesh defining discretization along the x and y directions
    cell_z_top : numpy.ndarray or float
        Define the elevations for the top face of all cells in the layer
    cell_z_bottom : numpy.ndarray or float
        Define the elevations for the bottom face of all cells in the layer
    """


class Simulation3DDifferential(BasePDESimulation):
    r"""Finite volume simulation class for gravity.

    Notes
    -----
    From Blakely (1996), the scalar potential :math:`\phi` outside the source region
    is obtained by solving a Poisson's equation:

    .. math::
        \nabla^2 \phi = 4 \pi \gamma \rho

    where :math:`\gamma` is the gravitational constant and :math:`\rho` defines the
    distribution of density within the source region.

    Applying the finite volumn method, we can solve the Poisson's equation on a
    3D voxel grid according to:

    .. math::
        \big [ \mathbf{D M_f D^T} \big ] \mathbf{u} = - \mathbf{M_c \, \rho}
    """

    _deprecate_main_map = "rhoMap"

    rho, rhoMap, rhoDeriv = props.Invertible("Specific density (g/cc)", default=1.0)

    solver = None

    def __init__(self, mesh, **kwargs):
        BaseSimulation.__init__(self, mesh, **kwargs)

        self._Div = self.mesh.face_divergence

    def getRHS(self):
        """Return right-hand side for the linear system"""
        Mc = self.Mcc
        rho = self.rho
        return -Mc * rho

    def getA(self):
        r"""
        GetA creates and returns the A matrix for the Gravity nodal problem

        The A matrix has the form:

        .. math ::

            \mathbf{A} =  \Div(\Mf Mui)^{-1}\Div^{T}
        """
        # Constructs A with 0 dirichlet
        if getattr(self, "_A", None) is None:
            self._A = self._Div * self.Mf * self._Div.T
        return self._A

    def fields(self, m=None):
        r"""Compute fields

        **INCOMPLETE**

        Parameters
        ----------
        m: (nP) np.ndarray
            The model

        Returns
        -------
        dict
            The fields
        """
        if m is not None:
            self.model = m

        A = self.getA()
        RHS = self.getRHS()

        Ainv = self.solver(A)
        u = Ainv * RHS

        gField = 4.0 * np.pi * NewtG * 1e8 * self._Div * u

        return {"G": gField, "u": u}
