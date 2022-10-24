from SimPEG.utils import mkvc, sdiag
from SimPEG import props
from ...base import BasePDESimulation
from ..base import BasePFSimulation, BaseEquivalentSourceLayerSimulation
from .survey import Survey
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

    rho, rhoMap, rhoDeriv = props.Invertible("Density")

    def __init__(self, mesh, rho=None, rhoMap=None, **kwargs):
        super().__init__(mesh, **kwargs)
        self.rho = rho
        self.rhoMap = rhoMap
        self._G = None
        self._gtg_diagonal = None
        self.modelMap = self.rhoMap

    def fields(self, m):
        self.model = m

        if self.store_sensitivities == "forward_only":
            # Compute the linear operation without forming the full dense G
            fields = mkvc(self.linear_operator())
        else:
            fields = self.G @ (self.rho).astype(np.float32)

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

    def evaluate_integral(self, receiver_location, components):
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

        node_evals = {}
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
        if "gxx" in components or "guv" in components:
            node_evals["gxx"] = prism_fzz(dy, dz, dx)
        if "gyy" in components or "guv" in components:
            node_evals["gyy"] = prism_fzz(dz, dx, dy)
            if "guv" in components:
                node_evals["guv"] = (node_evals["gyy"] - node_evals["gxx"]) * 0.5
                # (NN - EE) / 2
        inside_adjust = False
        if "gzz" in components:
            if "gxx" not in node_evals or "gyy" not in node_evals:
                node_evals["gzz"] = prism_fzz(dx, dy, dz)
            else:
                inside_adjust = True
                # The below need to be adjusted for observation points within a cell.
                # because `gxx + gyy + gzz = -4 * pi * G * rho`
                # gzz = - gxx - gyy - 4 * np.pi * G * rho[in_cell]
                node_evals["gzz"] = -node_evals["gxx"] - node_evals["gyy"]

        rows = {}
        for component in set(components):
            vals = node_evals[component]
            if self._unique_inv is not None:
                vals = vals[self._unique_inv]
            cell_vals = (
                vals[0]
                - vals[1]
                - vals[2]
                + vals[3]
                - vals[4]
                + vals[5]
                + vals[6]
                - vals[7]
            )
            if inside_adjust and component == "gzz":
                # should subtract 4 * pi to the cell containing the observation point
                # just need a little logic to find the containing cell
                # cell_vals[inside_cell] += 4 * np.pi
                pass
            if self.store_sensitivities == "forward_only":
                rows[component] = cell_vals @ self.rho
            else:
                rows[component] = cell_vals
            if len(component) == 3:
                rows[component] *= constants.G * 1e12  # conversion for Eotvos
            else:
                rows[component] *= constants.G * 1e8  # conversion for mGal

        return np.stack([rows[component] for component in components])


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

    rho, rhoMap, rhoDeriv = props.Invertible("Specific density (g/cc)")

    def __init__(self, mesh, rho=1.0, rhoMap=None, **kwargs):
        super().__init__(mesh, **kwargs)
        self.rho = rho
        self.rhoMap = rhoMap

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
            self._A = self._Div * self.Mf * self._Div.T.tocsr()
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
