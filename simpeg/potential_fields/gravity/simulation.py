import warnings
import numpy as np
import scipy.constants as constants
from geoana.kernels import prism_fz, prism_fzx, prism_fzy, prism_fzz
from scipy.constants import G as NewtG

from simpeg import props
from simpeg.utils import mkvc, sdiag

from ...base import BasePDESimulation
from ..base import BaseEquivalentSourceLayerSimulation, BasePFSimulation

from ._numba_functions import (
    choclo,
    _sensitivity_gravity_serial,
    _sensitivity_gravity_parallel,
    _forward_gravity_serial,
    _forward_gravity_parallel,
)

if choclo is not None:
    from numba import jit

    @jit(nopython=True)
    def kernel_uv(easting, northing, upward, radius):
        """Kernel for Guv gradiometry component."""
        result = 0.5 * (
            choclo.prism.kernel_nn(easting, northing, upward, radius)
            - choclo.prism.kernel_ee(easting, northing, upward, radius)
        )
        return result

    CHOCLO_KERNELS = {
        "gx": choclo.prism.kernel_e,
        "gy": choclo.prism.kernel_n,
        "gz": choclo.prism.kernel_u,
        "gxx": choclo.prism.kernel_ee,
        "gyy": choclo.prism.kernel_nn,
        "gzz": choclo.prism.kernel_uu,
        "gxy": choclo.prism.kernel_en,
        "gxz": choclo.prism.kernel_eu,
        "gyz": choclo.prism.kernel_nu,
        "guv": kernel_uv,
    }


def _get_conversion_factor(component):
    """
    Return conversion factor for the given component
    """
    if component in ("gx", "gy", "gz"):
        conversion_factor = 1e8
    elif component in ("gxx", "gyy", "gzz", "gxy", "gxz", "gyz", "guv"):
        conversion_factor = 1e12
    else:
        raise ValueError(f"Invalid component '{component}'.")
    return conversion_factor


class Simulation3DIntegral(BasePFSimulation):
    """
    Gravity simulation in integral form.

    .. important::

        Density model is assumed to be in g/cc.

    .. important::

        Acceleration components ("gx", "gy", "gz") are returned in mgal
        (:math:`10^{-5} m/s^2`).

    .. important::

        Gradient components ("gxx", "gyy", "gzz", "gxy", "gxz", "gyz") are
        returned in Eotvos (:math:`10^{-9} s^{-2}`).

    Parameters
    ----------
    mesh : discretize.TreeMesh or discretize.TensorMesh
        Mesh use to run the gravity simulation.
    survey : simpeg.potential_fields.gravity.Survey
        Gravity survey with information of the receivers.
    ind_active : (n_cells) numpy.ndarray, optional
        Array that indicates which cells in ``mesh`` are active cells.
    rho : numpy.ndarray, optional
        Density array for the active cells in the mesh.
    rhoMap : Mapping, optional
        Model mapping.
    sensitivity_dtype : numpy.dtype, optional
        Data type that will be used to build the sensitivity matrix.
    store_sensitivities : {"ram", "disk", "forward_only"}
        Options for storing sensitivity matrix. There are 3 options

        - 'ram': sensitivities are stored in the computer's RAM
        - 'disk': sensitivities are written to a directory
        - 'forward_only': you intend only do perform a forward simulation and
          sensitivities do not need to be stored

    sensitivity_path : str, optional
        Path to store the sensitivity matrix if ``store_sensitivities`` is set
        to ``"disk"``. Default to "./sensitivities".
    engine : {"geoana", "choclo"}, optional
       Choose which engine should be used to run the forward model.
    numba_parallel : bool, optional
        If True, the simulation will run in parallel. If False, it will
        run in serial. If ``engine`` is not ``"choclo"`` this argument will be
        ignored.
    """

    rho, rhoMap, rhoDeriv = props.Invertible("Density")

    def __init__(
        self,
        mesh,
        rho=None,
        rhoMap=None,
        engine="geoana",
        numba_parallel=True,
        **kwargs,
    ):
        super().__init__(mesh, engine=engine, numba_parallel=numba_parallel, **kwargs)
        self.rho = rho
        self.rhoMap = rhoMap
        self._G = None
        self._gtg_diagonal = None
        self.modelMap = self.rhoMap

        # Warn if n_processes has been passed
        if self.engine == "choclo" and "n_processes" in kwargs:
            warnings.warn(
                "The 'n_processes' will be ignored when selecting 'choclo' as the "
                "engine in the gravity simulation.",
                UserWarning,
                stacklevel=1,
            )
            self.n_processes = None

        # Define jit functions
        if self.engine == "choclo":
            if self.numba_parallel:
                self._sensitivity_gravity = _sensitivity_gravity_parallel
                self._forward_gravity = _forward_gravity_parallel
            else:
                self._sensitivity_gravity = _sensitivity_gravity_serial
                self._forward_gravity = _forward_gravity_serial

    def fields(self, m):
        """
        Forward model the gravity field of the mesh on the receivers in the survey

        Parameters
        ----------
        m : (n_active_cells,) numpy.ndarray
            Array with values for the model.

        Returns
        -------
        (nD,) numpy.ndarray
            Gravity fields generated by the given model on every receiver
            location.
        """
        self.model = m
        if self.store_sensitivities == "forward_only":
            # Compute the linear operation without forming the full dense G
            if self.engine == "choclo":
                fields = self._forward(self.rho)
            else:
                fields = mkvc(self.linear_operator())
        else:
            fields = self.G @ (self.rho).astype(self.sensitivity_dtype, copy=False)
        return np.asarray(fields)

    def getJtJdiag(self, m, W=None, f=None):
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
        return self.G @ dmu_dm_v.astype(self.sensitivity_dtype, copy=False)

    def Jtvec(self, m, v, f=None):
        """
        Sensitivity transposed times a vector
        """
        Jtvec = self.G.T @ v.astype(self.sensitivity_dtype, copy=False)
        return np.asarray(self.rhoDeriv.T @ Jtvec)

    @property
    def G(self):
        """
        Gravity forward operator
        """
        if getattr(self, "_G", None) is None:
            if self.engine == "choclo":
                self._G = self._sensitivity_matrix()
            else:
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
            node_evals["gzz"] = prism_fzz(dx, dy, dz)
            # The below should be uncommented when we are able to give the index of a
            # containing cell.
            # if "gxx" not in node_evals or "gyy" not in node_evals:
            #     node_evals["gzz"] = prism_fzz(dx, dy, dz)
            # else:
            #     inside_adjust = True
            #     # The below need to be adjusted for observation points within a cell.
            #     # because `gxx + gyy + gzz = -4 * pi * G * rho`
            #     # gzz = - gxx - gyy - 4 * np.pi * G * rho[in_cell]
            #     node_evals["gzz"] = -node_evals["gxx"] - node_evals["gyy"]

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

        return np.stack(
            [
                rows[component].astype(self.sensitivity_dtype, copy=False)
                for component in components
            ]
        )

    def _forward(self, densities):
        """
        Forward model the fields of active cells in the mesh on receivers.

        Parameters
        ----------
        densities : (n_active_cells) numpy.ndarray
            Array containing the densities of the active cells in the mesh, in
            g/cc.

        Returns
        -------
        (nD,) numpy.ndarray
            Always return a ``np.float64`` array.
        """
        # Gather active nodes and the indices of the nodes for each active cell
        active_nodes, active_cell_nodes = self._get_active_nodes()
        # Allocate fields array
        fields = np.zeros(self.survey.nD, dtype=self.sensitivity_dtype)
        # Compute fields
        index_offset = 0
        for components, receivers in self._get_components_and_receivers():
            n_components = len(components)
            n_elements = n_components * receivers.shape[0]
            for i, component in enumerate(components):
                kernel_func = CHOCLO_KERNELS[component]
                conversion_factor = _get_conversion_factor(component)
                vector_slice = slice(
                    index_offset + i, index_offset + n_elements, n_components
                )
                self._forward_gravity(
                    receivers,
                    active_nodes,
                    densities,
                    fields[vector_slice],
                    active_cell_nodes,
                    kernel_func,
                    constants.G * conversion_factor,
                )
            index_offset += n_elements
        return fields

    def _sensitivity_matrix(self):
        """
        Compute the sensitivity matrix G

        Returns
        -------
        (nD, n_active_cells) numpy.ndarray
        """
        # Gather active nodes and the indices of the nodes for each active cell
        active_nodes, active_cell_nodes = self._get_active_nodes()
        # Allocate sensitivity matrix
        shape = (self.survey.nD, self.nC)
        if self.store_sensitivities == "disk":
            sensitivity_matrix = np.memmap(
                self.sensitivity_path,
                shape=shape,
                dtype=self.sensitivity_dtype,
                order="C",  # it's more efficient to write in row major
                mode="w+",
            )
        else:
            sensitivity_matrix = np.empty(shape, dtype=self.sensitivity_dtype)
        # Start filling the sensitivity matrix
        index_offset = 0
        for components, receivers in self._get_components_and_receivers():
            n_components = len(components)
            n_rows = n_components * receivers.shape[0]
            for i, component in enumerate(components):
                kernel_func = CHOCLO_KERNELS[component]
                conversion_factor = _get_conversion_factor(component)
                matrix_slice = slice(
                    index_offset + i, index_offset + n_rows, n_components
                )
                self._sensitivity_gravity(
                    receivers,
                    active_nodes,
                    sensitivity_matrix[matrix_slice, :],
                    active_cell_nodes,
                    kernel_func,
                    constants.G * conversion_factor,
                )
            index_offset += n_rows
        return sensitivity_matrix


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
        Define the elevations for the top face of all cells in the layer. If an array it should be the same size as
        the active cell set.
    cell_z_bottom : numpy.ndarray or float
        Define the elevations for the bottom face of all cells in the layer. If an array it should be the same size as
        the active cell set.
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
