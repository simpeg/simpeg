from __future__ import annotations
import hashlib
import warnings
import numpy as np
from numpy.typing import NDArray
import scipy.constants as constants
from geoana.kernels import prism_fz, prism_fzx, prism_fzy, prism_fzz
from scipy.constants import G as NewtG
from scipy.sparse.linalg import LinearOperator, aslinearoperator

from simpeg import props
from simpeg.utils import mkvc, sdiag

from ...base import BasePDESimulation
from ..base import BaseEquivalentSourceLayerSimulation, BasePFSimulation

from ._numba import choclo, NUMBA_FUNCTIONS_3D, NUMBA_FUNCTIONS_2D

try:
    from warnings import deprecated
except ImportError:
    # Use the deprecated decorator provided by typing_extensions (which
    # supports older versions of Python) if it cannot be imported from
    # warnings.
    from typing_extensions import deprecated

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

    @jit(nopython=True)
    def gravity_uv(
        easting,
        northing,
        upward,
        prism_west,
        prism_east,
        prism_south,
        prism_north,
        prism_bottom,
        prism_top,
        density,
    ):
        """Forward model the Guv gradiometry component."""
        result = 0.5 * (
            choclo.prism.gravity_nn(
                easting,
                northing,
                upward,
                prism_west,
                prism_east,
                prism_south,
                prism_north,
                prism_bottom,
                prism_top,
                density,
            )
            - choclo.prism.gravity_ee(
                easting,
                northing,
                upward,
                prism_west,
                prism_east,
                prism_south,
                prism_north,
                prism_bottom,
                prism_top,
                density,
            )
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

    CHOCLO_FORWARD_FUNCS = {
        "gx": choclo.prism.gravity_e,
        "gy": choclo.prism.gravity_n,
        "gz": choclo.prism.gravity_u,
        "gxx": choclo.prism.gravity_ee,
        "gyy": choclo.prism.gravity_nn,
        "gzz": choclo.prism.gravity_uu,
        "gxy": choclo.prism.gravity_en,
        "gxz": choclo.prism.gravity_eu,
        "gyz": choclo.prism.gravity_nu,
        "guv": gravity_uv,
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
    r"""
    Gravity simulation in integral form.

    .. note::

        The gravity simulation assumes the following units for its inputs and outputs:

        - Density model is assumed to be in gram per cubic centimeter (g/cc).
        - Acceleration components (``"gx"``, ``"gy"``, ``"gz"``) are returned in mgal
          (:math:`10^{-5} \text{m}/\text{s}^2`).
        - Gradient components (``"gxx"``, ``"gyy"``, ``"gzz"``, ``"gxy"``, ``"gxz"``,
          ``"gyz"``, ``"guv"``) are returned in Eotvos (:math:`10^{-9} s^{-2}`).

    .. important::

        Following SimPEG convention for the right-handed xyz coordinate system, the
        z axis points *upwards*. Therefore, the ``"gz"`` component corresponds to the
        **upward** component of the gravity acceleration vector.


    Parameters
    ----------
    mesh : discretize.TreeMesh or discretize.TensorMesh
        Mesh use to run the gravity simulation.
    survey : simpeg.potential_fields.gravity.Survey
        Gravity survey with information of the receivers.
    active_cells : (n_cells) numpy.ndarray, optional
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
          sensitivities do not need to be stored. The sensitivity matrix ``G``
          is never created, but it'll be defined as
          a :class:`~scipy.sparse.linalg.LinearOperator`.

    sensitivity_path : str, optional
        Path to store the sensitivity matrix if ``store_sensitivities`` is set
        to ``"disk"``. Default to "./sensitivities".
    engine : {"geoana", "choclo"}, optional
       Choose which engine should be used to run the forward model.
    numba_parallel : bool, optional
        If True, the simulation will run in parallel. If False, it will
        run in serial. If ``engine`` is not ``"choclo"`` this argument will be
        ignored.
    ind_active : np.ndarray of int or bool

        .. deprecated:: 0.23.0

           Argument ``ind_active`` is deprecated in favor of
           ``active_cells`` and will be removed in SimPEG v0.24.0.
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

    def fields(self, m):
        """
        Forward model the gravity field of the mesh on the receivers in the survey

        Parameters
        ----------
        m : (n_param,) numpy.ndarray
            The model parameters.

        Returns
        -------
        (nD,) numpy.ndarray
            Gravity fields generated by the given model on every receiver
            location.
        """
        # Need to assign the model, so the rho property can be accessed.
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
        r"""
        Compute diagonal of :math:`\mathbf{J}^T \mathbf{J}``.

        Parameters
        ----------
        m : (n_param,) numpy.ndarray
            The model parameters.
        W : (nD, nD) np.ndarray or scipy.sparse.sparray, optional
            Diagonal matrix with the square root of the weights. If not None,
            the function returns the diagonal of
            :math:`\mathbf{J}^T \mathbf{W}^T \mathbf{W} \mathbf{J}``.
        f : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        (n_active_cells) np.ndarray
            Array with the diagonal of ``J.T @ J``.

        Notes
        -----
        If ``store_sensitivities`` is ``"forward_only"``, the ``G`` matrix is
        never allocated in memory, and the diagonal is obtained by
        accumulation, computing each element of the ``G`` matrix on the fly.

        This method caches the diagonal ``G.T @ W.T @ W @ G`` and the sha256
        hash of the diagonal of the ``W`` matrix. This way, if same weights are
        passed to it, it reuses the cached diagonal so it doesn't need to be
        recomputed.
        If new weights are passed, the cache is updated with the latest
        diagonal of ``G.T @ W.T @ W @ G``.
        """
        # Need to assign the model, so the rhoDeriv can be computed (if the
        # model is None, the rhoDeriv is going to be Zero).
        self.model = m

        # We should probably check that W is diagonal. Let's assume it for now.
        weights = (
            W.diagonal() ** 2
            if W is not None
            else np.ones(self.survey.nD, dtype=np.float64)
        )

        # Compute gtg (G.T @ W.T @ W @ G) if it's not cached, or if the
        # weights are not the same.
        weights_sha256 = hashlib.sha256(weights)
        use_cached_gtg = (
            hasattr(self, "_gtg_diagonal")
            and hasattr(self, "_weights_sha256")
            and self._weights_sha256.digest() == weights_sha256.digest()
        )
        if not use_cached_gtg:
            self._gtg_diagonal = self._get_gtg_diagonal(weights)
            self._weights_sha256 = weights_sha256

        # Multiply the gtg_diagonal by the derivative of the mapping
        diagonal = mkvc(
            (sdiag(np.sqrt(self._gtg_diagonal)) @ self.rhoDeriv).power(2).sum(axis=0)
        )
        return diagonal

    def _get_gtg_diagonal(self, weights: NDArray) -> NDArray:
        """
        Compute the diagonal of ``G.T @ W.T @ W @ G``.

        Parameters
        ----------
        weights : np.ndarray
            Weights array: diagonal of ``W.T @ W``.

        Returns
        -------
        np.ndarray
        """
        match self.store_sensitivities, self.engine:
            case ("forward_only", "geoana"):
                msg = (
                    "Computing the diagonal of G.T @ G with "
                    'store_sensitivities="forward_only" and engine="geoana" '
                    "hasn't been implemented yet. "
                    'Choose store_sensitivities="ram" or "disk", '
                    'or another engine, like "choclo".'
                )
                raise NotImplementedError(msg)
            case ("forward_only", "choclo"):
                gtg_diagonal = self._gtg_diagonal_without_building_g(weights)
            case (_, _):
                # In Einstein notation, the j-th element of the diagonal is:
                #   d_j = w_i * G_{ij} * G_{ij}
                gtg_diagonal = np.asarray(
                    np.einsum("i,ij,ij->j", weights, self.G, self.G)
                )
        return gtg_diagonal

    def getJ(self, m, f=None) -> NDArray[np.float64 | np.float32] | LinearOperator:
        r"""
        Sensitivity matrix :math:`\mathbf{J}`.

        Parameters
        ----------
        m : (n_param,) numpy.ndarray
            The model parameters.
        f : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        (nD, n_active_cells) np.ndarray or scipy.sparse.linalg.LinearOperator.
            Array or :class:`~scipy.sparse.linalg.LinearOperator` for the
            :math:`\mathbf{J}` matrix.
            A :class:`~scipy.sparse.linalg.LinearOperator` will be returned if
            ``store_sensitivities`` is ``"forward_only"``, otherwise a dense
            array will be returned.

        Notes
        -----
        If ``store_sensitivities`` is ``"ram"`` or ``"disk"``, a dense array
        for the ``J`` matrix is returned.
        A :class:`~scipy.sparse.linalg.LinearOperator` is returned if
        ``store_sensitivities`` is ``"forward_only"``. This object can perform
        operations like ``J @ m`` or ``J.T @ v`` without allocating the full
        ``J`` matrix in memory.
        """
        # Need to assign the model, so the rhoDeriv can be computed (if the
        # model is None, the rhoDeriv is going to be Zero).
        self.model = m
        rhoDeriv = (
            self.rhoDeriv
            if not isinstance(self.G, LinearOperator)
            else aslinearoperator(self.rhoDeriv)
        )
        return self.G @ rhoDeriv

    def Jvec(self, m, v, f=None):
        """
        Dot product between sensitivity matrix and a vector.

        Parameters
        ----------
        m : (n_param,) numpy.ndarray
            The model parameters. This array is used to compute the ``J``
            matrix.
        v : (n_param,) numpy.ndarray
            Vector used in the matrix-vector multiplication.
        f : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        (nD,) numpy.ndarray

        Notes
        -----
        If ``store_sensitivities`` is set to ``"forward_only"``, then the
        matrix `G` is never fully constructed, and the dot product is computed
        by accumulation, computing the matrix elements on the fly. Otherwise,
        the full matrix ``G`` is constructed and stored either in memory or
        disk.
        """
        # Need to assign the model, so the rhoDeriv can be computed (if the
        # model is None, the rhoDeriv is going to be Zero).
        self.model = m
        dmu_dm_v = self.rhoDeriv @ v
        return self.G @ dmu_dm_v.astype(self.sensitivity_dtype, copy=False)

    def Jtvec(self, m, v, f=None):
        """
        Dot product between transposed sensitivity matrix and a vector.

        Parameters
        ----------
        m : (n_param,) numpy.ndarray
            The model parameters. This array is used to compute the ``J``
            matrix.
        v : (nD,) numpy.ndarray
            Vector used in the matrix-vector multiplication.
        f : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        (nD,) numpy.ndarray

        Notes
        -----
        If ``store_sensitivities`` is set to ``"forward_only"``, then the
        matrix `G` is never fully constructed, and the dot product is computed
        by accumulation, computing the matrix elements on the fly. Otherwise,
        the full matrix ``G`` is constructed and stored either in memory or
        disk.
        """
        # Need to assign the model, so the rhoDeriv can be computed (if the
        # model is None, the rhoDeriv is going to be Zero).
        self.model = m
        Jtvec = self.G.T @ v.astype(self.sensitivity_dtype, copy=False)
        return np.asarray(self.rhoDeriv.T @ Jtvec)

    @property
    def G(self) -> NDArray | np.memmap | LinearOperator:
        """
        Gravity forward operator.
        """
        if not hasattr(self, "_G"):
            match self.engine, self.store_sensitivities:
                case ("choclo", "forward_only"):
                    self._G = self._sensitivity_matrix_as_operator()
                case ("choclo", _):
                    self._G = self._sensitivity_matrix()
                case ("geoana", "forward_only"):
                    msg = (
                        "Accessing matrix G with "
                        'store_sensitivities="forward_only" and engine="geoana" '
                        "hasn't been implemented yet. "
                        'Choose store_sensitivities="ram" or "disk", '
                        'or another engine, like "choclo".'
                    )
                    raise NotImplementedError(msg)
                case ("geoana", _):
                    self._G = self.linear_operator()
        return self._G

    @property
    @deprecated(
        "The `gtg_diagonal` property has been deprecated. "
        "It will be removed in SimPEG v0.25.0.",
        category=FutureWarning,
    )
    def gtg_diagonal(self):
        """
        Diagonal of GtG
        """
        return getattr(self, "_gtg_diagonal", None)

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
        # Get Numba function
        forward_func = NUMBA_FUNCTIONS_3D["forward"][self.numba_parallel]
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
                forward_func(
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
        Compute the sensitivity matrix ``G``.

        Returns
        -------
        (nD, n_active_cells) numpy.ndarray
        """
        # Get Numba function
        sensitivity_func = NUMBA_FUNCTIONS_3D["sensitivity"][self.numba_parallel]
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
                sensitivity_func(
                    receivers,
                    active_nodes,
                    sensitivity_matrix[matrix_slice, :],
                    active_cell_nodes,
                    kernel_func,
                    constants.G * conversion_factor,
                )
            index_offset += n_rows
        return sensitivity_matrix

    def _sensitivity_matrix_transpose_dot_vec(self, vector):
        """
        Compute ``G.T @ v`` without building ``G``.

        Parameters
        ----------
        vector : (nD) numpy.ndarray
            Vector used in the dot product.

        Returns
        -------
        (n_active_cells) numpy.ndarray
        """
        # Get Numba function
        sensitivity_t_dot_v_func = NUMBA_FUNCTIONS_3D["gt_dot_v"][self.numba_parallel]
        # Gather active nodes and the indices of the nodes for each active cell
        active_nodes, active_cell_nodes = self._get_active_nodes()
        # Allocate resulting array
        result = np.zeros(self.nC)
        # Start filling the result array
        index_offset = 0
        for components, receivers in self._get_components_and_receivers():
            n_components = len(components)
            n_rows = n_components * receivers.shape[0]
            for i, component in enumerate(components):
                kernel_func = CHOCLO_KERNELS[component]
                conversion_factor = _get_conversion_factor(component)
                vector_slice = slice(
                    index_offset + i, index_offset + n_rows, n_components
                )
                sensitivity_t_dot_v_func(
                    receivers,
                    active_nodes,
                    active_cell_nodes,
                    kernel_func,
                    constants.G * conversion_factor,
                    vector[vector_slice],
                    result,
                )
            index_offset += n_rows
        return result

    def _sensitivity_matrix_as_operator(self):
        """
        Create a LinearOperator for the sensitivity matrix G.

        Returns
        -------
        scipy.sparse.linalg.LinearOperator
        """
        shape = (self.survey.nD, self.nC)
        linear_op = LinearOperator(
            shape=shape,
            matvec=self._forward,
            rmatvec=self._sensitivity_matrix_transpose_dot_vec,
            dtype=np.float64,
        )
        return linear_op

    def _gtg_diagonal_without_building_g(self, weights):
        """
        Compute the diagonal of ``G.T @ G`` without building the ``G`` matrix.

        Parameters
        -----------
        weights : (nD,) array
            Array with data weights. It should be the diagonal of the ``W``
            matrix, squared.

        Returns
        -------
        (n_active_cells) numpy.ndarray
        """
        # Get Numba function
        diagonal_gtg_func = NUMBA_FUNCTIONS_3D["diagonal_gtg"][self.numba_parallel]
        # Gather active nodes and the indices of the nodes for each active cell
        active_nodes, active_cell_nodes = self._get_active_nodes()
        # Allocate array for the diagonal of G.T @ G
        diagonal = np.zeros(self.nC, dtype=np.float64)
        # Start filling the diagonal array
        for components, receivers in self._get_components_and_receivers():
            for component in components:
                kernel_func = CHOCLO_KERNELS[component]
                conversion_factor = _get_conversion_factor(component)
                diagonal_gtg_func(
                    receivers,
                    active_nodes,
                    active_cell_nodes,
                    kernel_func,
                    constants.G * conversion_factor,
                    weights,
                    diagonal,
                )
        return diagonal


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
        Define the elevations for the top face of all cells in the layer.
        If an array it should be the same size as the active cell set.
    cell_z_bottom : numpy.ndarray or float
        Define the elevations for the bottom face of all cells in the layer.
        If an array it should be the same size as the active cell set.
    engine : {"geoana", "choclo"}, optional
        Choose which engine should be used to run the forward model.
    numba_parallel : bool, optional
        If True, the simulation will run in parallel. If False, it will
        run in serial. If ``engine`` is not ``"choclo"`` this argument will be
        ignored.
    """

    def __init__(
        self,
        mesh,
        cell_z_top,
        cell_z_bottom,
        engine="geoana",
        numba_parallel=True,
        **kwargs,
    ):
        super().__init__(
            mesh,
            cell_z_top,
            cell_z_bottom,
            engine=engine,
            numba_parallel=numba_parallel,
            **kwargs,
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
        # Get Numba function
        forward_func = NUMBA_FUNCTIONS_2D["forward"][self.numba_parallel]
        # Get cells in the 2D mesh and keep only active cells
        cells_bounds_active = self.mesh.cell_bounds[self.active_cells]
        # Allocate fields array
        fields = np.zeros(self.survey.nD, dtype=self.sensitivity_dtype)
        # Compute fields
        index_offset = 0
        for components, receivers in self._get_components_and_receivers():
            n_components = len(components)
            n_elements = n_components * receivers.shape[0]
            for i, component in enumerate(components):
                choclo_forward_func = CHOCLO_FORWARD_FUNCS[component]
                conversion_factor = _get_conversion_factor(component)
                vector_slice = slice(
                    index_offset + i, index_offset + n_elements, n_components
                )
                forward_func(
                    receivers,
                    cells_bounds_active,
                    self.cell_z_top,
                    self.cell_z_bottom,
                    densities,
                    fields[vector_slice],
                    choclo_forward_func,
                    conversion_factor,
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
        # Get Numba function
        sensitivity_func = NUMBA_FUNCTIONS_2D["sensitivity"][self.numba_parallel]
        # Get cells in the 2D mesh and keep only active cells
        cells_bounds_active = self.mesh.cell_bounds[self.active_cells]
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
                choclo_forward_func = CHOCLO_FORWARD_FUNCS[component]
                conversion_factor = _get_conversion_factor(component)
                matrix_slice = slice(
                    index_offset + i, index_offset + n_rows, n_components
                )
                sensitivity_func(
                    receivers,
                    cells_bounds_active,
                    self.cell_z_top,
                    self.cell_z_bottom,
                    sensitivity_matrix[matrix_slice, :],
                    choclo_forward_func,
                    conversion_factor,
                )
            index_offset += n_rows
        return sensitivity_matrix

    def _sensitivity_matrix_transpose_dot_vec(self, vector):
        """
        Compute ``G.T @ v`` without building ``G``.

        Parameters
        ----------
        vector : (nD) numpy.ndarray
            Vector used in the dot product.

        Returns
        -------
        (n_active_cells) numpy.ndarray
        """
        # Get Numba function
        g_t_dot_v_func = NUMBA_FUNCTIONS_2D["gt_dot_v"][self.numba_parallel]
        # Get cells in the 2D mesh and keep only active cells
        cells_bounds_active = self.mesh.cell_bounds[self.active_cells]
        # Allocate resulting array
        result = np.zeros(self.nC)
        # Start filling the result array
        index_offset = 0
        for components, receivers in self._get_components_and_receivers():
            n_components = len(components)
            n_rows = n_components * receivers.shape[0]
            for i, component in enumerate(components):
                choclo_forward_func = CHOCLO_FORWARD_FUNCS[component]
                conversion_factor = _get_conversion_factor(component)
                vector_slice = slice(
                    index_offset + i, index_offset + n_rows, n_components
                )
                g_t_dot_v_func(
                    receivers,
                    cells_bounds_active,
                    self.cell_z_top,
                    self.cell_z_bottom,
                    choclo_forward_func,
                    conversion_factor,
                    vector[vector_slice],
                    result,
                )
            index_offset += n_rows
        return result

    def _gtg_diagonal_without_building_g(self, weights):
        """
        Compute the diagonal of ``G.T @ G`` without building the ``G`` matrix.

        Parameters
        -----------
        weights : (nD,) array
            Array with data weights. It should be the diagonal of the ``W``
            matrix, squared.

        Returns
        -------
        (n_active_cells) numpy.ndarray
        """
        # Get Numba function
        diagonal_gtg_func = NUMBA_FUNCTIONS_2D["diagonal_gtg"][self.numba_parallel]
        # Get cells in the 2D mesh and keep only active cells
        cells_bounds_active = self.mesh.cell_bounds[self.active_cells]
        # Allocate array for the diagonal of G.T @ G
        diagonal = np.zeros(self.nC, dtype=np.float64)
        # Start filling the diagonal array
        for components, receivers in self._get_components_and_receivers():
            for component in components:
                choclo_forward_func = CHOCLO_FORWARD_FUNCS[component]
                conversion_factor = _get_conversion_factor(component)
                diagonal_gtg_func(
                    receivers,
                    cells_bounds_active,
                    self.cell_z_top,
                    self.cell_z_bottom,
                    choclo_forward_func,
                    conversion_factor,
                    weights,
                    diagonal,
                )
        return diagonal


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
