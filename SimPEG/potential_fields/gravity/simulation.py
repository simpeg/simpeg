from SimPEG.utils import mkvc, sdiag, setKwargs
from SimPEG import props
from ...simulation import BaseSimulation
from ...base import BasePDESimulation
from ..base import BasePFSimulation, BaseEquivalentSourceLayerSimulation
import scipy.constants as constants
from scipy.constants import G as NewtG
import numpy as np
from geoana.kernels import (
    prism_fz,
    prism_fzz,
    prism_fzx,
    prism_fzy,
)


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
        dr = self._nodes - receiver_location
        dx = dr[..., 0]
        dy = dr[..., 1]
        dz = dr[..., 2]

        node_evals = {component: np.zeros(len(dx)) for component in components}

        gxx = np.zeros(len(dx))
        gyy = np.zeros(len(dx))
        if "gx" in components:
            node_evals["gx"] = prism_fz(dy, dz, dx)
        if "gy" in components:
            node_evals["gy"] = prism_fz(dz, dx, dy)
        if "gz" in components:
            node_evals["gz"] = prism_fz(dx, dy, dz)
        if "gxy" in components:
            node_evals["gxy"] = prism_fzx(dy, dz, dx)
        if "gxz" in components:
            node_evals["gxz"] = prism_fzx(dx, dy, dz)
        if "gyz" in components:
            node_evals["gyz"] = prism_fzy(dx, dy, dz)
        gxx = None
        if "gxx" in components or "guv" in components:
            gxx = prism_fzz(dy, dz, dx)
            if "gxx" in components:
                node_evals["gxx"] = gxx
        gyy = None
        if "gyy" in components or "guv" in components:
            gyy = prism_fzz(dz, dx, dy)
            if "gyy" in components:
                node_evals["gyy"] = gyy
            if "guv" in components:
                node_evals["guv"] = (gyy - gxx) * 0.5  # (NN - EE) / 2
        inside_adjust = False
        if "gzz" in components:
            if gxx is None or gyy is None:
                node_evals["gzz"] = prism_fzz(dx, dy, dz)
            else:
                inside_adjust = True
                # The below need to be adjusted for observation points within a cell.
                # because `gxx + gyy + gzz = -4 * pi * G * rho`
                # gzz = - gxx - gyy - 4 * np.pi * G * rho[in_cell]
                node_evals["gzz"] = -gxx - gyy

        rows = {}
        for component in components:
            vals = node_evals[component]
            if self.mesh.dim > 2:
                vals = vals[self._unique_inv]
            cell_vals = (
                vals[:, 7]
                - vals[:, 6]
                - vals[:, 5]
                + vals[:, 4]
                - vals[:, 3]
                + vals[:, 2]
                + vals[:, 1]
                - vals[:, 0]
            )
            if inside_adjust and component == "gzz":
                # should subtract 4 * pi to the cell containing the observation point
                # just need a little logic to find the containing cell
                # cell_vals[inside_cell] += 4 * np.pi
                pass
            rows[component] = cell_vals
            if len(component) == 3:
                rows[component] *= -constants.G * 1e12  # conversion for Eotvos
            else:
                rows[component] *= -constants.G * 1e8  # conversion for mGal

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
    """
    Gravity in differential equations!
    """

    _deprecate_main_map = "rhoMap"

    rho, rhoMap, rhoDeriv = props.Invertible("Specific density (g/cc)", default=1.0)

    solver = None

    def __init__(self, mesh, **kwargs):
        BaseSimulation.__init__(self, mesh, **kwargs)

        self._Div = self.mesh.face_divergence

    def getRHS(self):
        """"""
        Mc = self.Mcc
        rho = self.rho
        return -Mc * rho

    def getA(self):
        """
        GetA creates and returns the A matrix for the Gravity nodal problem

        The A matrix has the form:

        .. math ::

            \mathbf{A} =  \Div(\MfMui)^{-1}\Div^{T}
        """
        # Constructs A with 0 dirichlet
        if getattr(self, "_A", None) is None:
            self._A = self._Div * self.Mf * self._Div.T
        return self._A

    def fields(self, m=None):
        """
        Return magnetic potential (u) and flux (B)
        u: defined on the cell nodes [nC x 1]
        gField: defined on the cell faces [nF x 1]

        After we compute u, then we update B.

        .. math ::

            \mathbf{B}_s = (\MfMui)^{-1}\mathbf{M}^f_{\mu_0^{-1}}\mathbf{B}_0-\mathbf{B}_0 -(\MfMui)^{-1}\Div^T \mathbf{u}

        """
        if m is not None:
            self.model = m

        A = self.getA()
        RHS = self.getRHS()

        Ainv = self.solver(A)
        u = Ainv * RHS

        gField = 4.0 * np.pi * NewtG * 1e8 * self._Div * u

        return {"G": gField, "u": u}
