import warnings
import numpy as np
import scipy.sparse as sp
from geoana.kernels import (
    prism_fxxy,
    prism_fxxz,
    prism_fxyz,
    prism_fzx,
    prism_fzy,
    prism_fzz,
    prism_fzzz,
)
from scipy.constants import mu_0

from simpeg import Solver, props, utils
from simpeg.utils import mat_utils, mkvc, sdiag
from simpeg.utils.code_utils import deprecate_property, validate_string, validate_type

from ...base import BaseMagneticPDESimulation
from ..base import BaseEquivalentSourceLayerSimulation, BasePFSimulation
from .analytics import CongruousMagBC
from .survey import Survey

from ._numba_functions import (
    choclo,
    _sensitivity_tmi_parallel,
    _sensitivity_tmi_serial,
    _sensitivity_mag_parallel,
    _sensitivity_mag_serial,
    _forward_tmi_parallel,
    _forward_tmi_serial,
    _forward_mag_parallel,
    _forward_mag_serial,
)

if choclo is not None:
    CHOCLO_SUPPORTED_COMPONENTS = {"tmi", "bx", "by", "bz"}
    CHOCLO_KERNELS = {
        "bx": (choclo.prism.kernel_ee, choclo.prism.kernel_en, choclo.prism.kernel_eu),
        "by": (choclo.prism.kernel_en, choclo.prism.kernel_nn, choclo.prism.kernel_nu),
        "bz": (choclo.prism.kernel_eu, choclo.prism.kernel_nu, choclo.prism.kernel_uu),
    }


class Simulation3DIntegral(BasePFSimulation):
    """
    Magnetic simulation in integral form.

    Parameters
    ----------
    mesh : discretize.TreeMesh or discretize.TensorMesh
        Mesh use to run the magnetic simulation.
    survey : simpeg.potential_fields.magnetics.Survey
        Magnetic survey with information of the receivers.
    ind_active : (n_cells) numpy.ndarray, optional
        Array that indicates which cells in ``mesh`` are active cells.
    chi : numpy.ndarray, optional
        Susceptibility array for the active cells in the mesh.
    chiMap : Mapping, optional
        Model mapping.
    model_type : str, optional
        Whether the model are susceptibilities of the cells (``"scalar"``),
        or effective susceptibilities (``"vector"``).
    is_amplitude_data : bool, optional
        If True, the returned fields will be the amplitude of the magnetic
        field. If False, the fields will be returned unmodified.
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

    chi, chiMap, chiDeriv = props.Invertible("Magnetic Susceptibility (SI)")

    def __init__(
        self,
        mesh,
        chi=None,
        chiMap=None,
        model_type="scalar",
        is_amplitude_data=False,
        engine="geoana",
        numba_parallel=True,
        **kwargs,
    ):
        self.model_type = model_type
        super().__init__(mesh, engine=engine, numba_parallel=numba_parallel, **kwargs)
        self.chi = chi
        self.chiMap = chiMap

        self._G = None
        self._M = None
        self._gtg_diagonal = None
        self.is_amplitude_data = is_amplitude_data
        self.modelMap = self.chiMap

        # Warn if n_processes has been passed
        if self.engine == "choclo" and "n_processes" in kwargs:
            warnings.warn(
                "The 'n_processes' will be ignored when selecting 'choclo' as the "
                "engine in the magnetic simulation.",
                UserWarning,
                stacklevel=1,
            )
            self.n_processes = None

        if self.engine == "choclo":
            if self.numba_parallel:
                self._sensitivity_tmi = _sensitivity_tmi_parallel
                self._sensitivity_mag = _sensitivity_mag_parallel
                self._forward_tmi = _forward_tmi_parallel
                self._forward_mag = _forward_mag_parallel
            else:
                self._sensitivity_tmi = _sensitivity_tmi_serial
                self._sensitivity_mag = _sensitivity_mag_serial
                self._forward_tmi = _forward_tmi_serial
                self._forward_mag = _forward_mag_serial

    @property
    def model_type(self):
        """Type of magnetization model

        Returns
        -------
        str
            A string defining the model type for the simulation.
            One of {'scalar', 'vector'}.
        """
        return self._model_type

    @model_type.setter
    def model_type(self, value):
        self._model_type = validate_string("model_type", value, ["scalar", "vector"])

    @property
    def is_amplitude_data(self):
        return self._is_amplitude_data

    @is_amplitude_data.setter
    def is_amplitude_data(self, value):
        self._is_amplitude_data = validate_type("is_amplitude_data", value, bool)

    @property
    def M(self):
        """
        M: ndarray
            Magnetization matrix
        """
        if self.model_type == "vector":
            return None
        if getattr(self, "_M", None) is None:
            mag = self.survey.source_field.b0
            self._M = np.ones((self.nC, 3)) * mag
        return self._M

    @M.setter
    def M(self, M):
        """
        Create magnetization matrix from unit vector orientation
        :parameter
        M: array (3*nC,) or (nC, 3)
        """
        if self.model_type == "vector":
            self._M = sdiag(mkvc(M) * self.survey.source_field.amplitude)
        else:
            M = np.asarray(M)
            self._M = M.reshape((self.nC, 3))

    def fields(self, model):
        self.model = model
        # model = self.chiMap * model
        if self.store_sensitivities == "forward_only":
            if self.engine == "choclo":
                fields = self._forward(self.chi)
            else:
                fields = mkvc(self.linear_operator())
        else:
            fields = np.asarray(
                self.G @ self.chi.astype(self.sensitivity_dtype, copy=False)
            )

        if self.is_amplitude_data:
            fields = self.compute_amplitude(fields)

        return fields

    @property
    def G(self):
        if getattr(self, "_G", None) is None:
            if self.engine == "choclo":
                self._G = self._sensitivity_matrix()
            else:
                self._G = self.linear_operator()

        return self._G

    modelType = deprecate_property(
        model_type, "modelType", "model_type", removal_version="0.18.0", error=True
    )

    @property
    def nD(self):
        """
        Number of data
        """
        self._nD = self.survey.receiver_locations.shape[0]

        return self._nD

    @property
    def tmi_projection(self):
        if getattr(self, "_tmi_projection", None) is None:
            # Convert from north to cartesian
            self._tmi_projection = mat_utils.dip_azimuth2cartesian(
                self.survey.source_field.inclination,
                self.survey.source_field.declination,
            ).squeeze()

        return self._tmi_projection

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
            if not self.is_amplitude_data:
                for i in range(len(W)):
                    diag += W[i] * (self.G[i] * self.G[i])
            else:
                ampDeriv = self.ampDeriv
                Gx = self.G[::3]
                Gy = self.G[1::3]
                Gz = self.G[2::3]
                for i in range(len(W)):
                    row = (
                        ampDeriv[0, i] * Gx[i]
                        + ampDeriv[1, i] * Gy[i]
                        + ampDeriv[2, i] * Gz[i]
                    )
                    diag += W[i] * (row * row)
            self._gtg_diagonal = diag
        else:
            diag = self._gtg_diagonal
        return mkvc((sdiag(np.sqrt(diag)) @ self.chiDeriv).power(2).sum(axis=0))

    def Jvec(self, m, v, f=None):
        self.model = m
        dmu_dm_v = self.chiDeriv @ v

        Jvec = self.G @ dmu_dm_v.astype(self.sensitivity_dtype, copy=False)

        if self.is_amplitude_data:
            # dask doesn't support an "order" argument to reshape...
            Jvec = Jvec.reshape((-1, 3)).T  # reshape((3, -1), order="F")
            ampDeriv_Jvec = self.ampDeriv * Jvec
            return ampDeriv_Jvec[0] + ampDeriv_Jvec[1] + ampDeriv_Jvec[2]
        else:
            return Jvec

    def Jtvec(self, m, v, f=None):
        self.model = m

        if self.is_amplitude_data:
            v = self.ampDeriv * v
            # dask doesn't support and "order" argument to reshape...
            v = v.T.reshape(-1)  # .reshape(-1, order="F")
        Jtvec = self.G.T @ v.astype(self.sensitivity_dtype, copy=False)
        return np.asarray(self.chiDeriv.T @ Jtvec)

    @property
    def ampDeriv(self):
        if getattr(self, "_ampDeriv", None) is None:
            fields = np.asarray(
                self.G.dot(self.chi).astype(self.sensitivity_dtype, copy=False)
            )
            self._ampDeriv = self.normalized_fields(fields)

        return self._ampDeriv

    @classmethod
    def normalized_fields(cls, fields):
        """
        Return the normalized B fields
        """

        # Get field amplitude
        amp = cls.compute_amplitude(fields.astype(np.float64))

        return fields.reshape((3, -1), order="F") / amp[None, :]

    @classmethod
    def compute_amplitude(cls, b_xyz):
        """
        Compute amplitude of the magnetic field
        """

        amplitude = np.linalg.norm(b_xyz.reshape((3, -1), order="F"), axis=0)

        return amplitude

    def evaluate_integral(self, receiver_location, components):
        """
        Load in the active nodes of a tensor mesh and computes the magnetic
        forward relation between a cuboid and a given observation
        location outside the Earth [obsx, obsy, obsz]

        INPUT:
        receiver_location:  [obsx, obsy, obsz] nC x 3 Array

        components: list[str]
            List of magnetic components chosen from:
            'tmi', 'bx', 'by', 'bz', 'bxx', 'bxy', 'bxz', 'byy', 'byz', 'bzz', 'tmi_x', 'tmi_y', 'tmi_z'

        OUTPUT:
        Tx = [Txx Txy Txz]
        Ty = [Tyx Tyy Tyz]
        Tz = [Tzx Tzy Tzz]
        """
        dr = self._nodes - receiver_location
        dx = dr[..., 0]
        dy = dr[..., 1]
        dz = dr[..., 2]

        node_evals = {}
        if "bx" in components or "tmi" in components:
            node_evals["gxx"] = prism_fzz(dy, dz, dx)
            node_evals["gxy"] = prism_fzx(dy, dz, dx)
            node_evals["gxz"] = prism_fzy(dy, dz, dx)
        if "by" in components or "tmi" in components:
            if "gxy" not in node_evals:
                node_evals["gxy"] = prism_fzx(dy, dz, dx)
            node_evals["gyy"] = prism_fzz(dz, dx, dy)
            node_evals["gyz"] = prism_fzy(dx, dy, dz)
        if "bz" in components or "tmi" in components:
            if "gxz" not in node_evals:
                node_evals["gxz"] = prism_fzy(dy, dz, dx)
            if "gyz" not in node_evals:
                node_evals["gyz"] = prism_fzy(dx, dy, dz)
            node_evals["gzz"] = prism_fzz(dx, dy, dz)
            # the below will be uncommented when we give the containing cell index
            # for interior observations.
            # if "gxx" not in node_evals or "gyy" not in node_evals:
            #     node_evals["gzz"] = prism_fzz(dx, dy, dz)
            # else:
            #     # This is the one that would need to be adjusted if the observation is
            #     # inside an active cell.
            #     node_evals["gzz"] = -node_evals["gxx"] - node_evals["gyy"]

        if "bxx" in components or "tmi_x" in components:
            node_evals["gxxx"] = prism_fzzz(dy, dz, dx)
            node_evals["gxxy"] = prism_fxxy(dx, dy, dz)
            node_evals["gxxz"] = prism_fxxz(dx, dy, dz)
        if "bxy" in components or "tmi_x" in components or "tmi_y" in components:
            if "gxxy" not in node_evals:
                node_evals["gxxy"] = prism_fxxy(dx, dy, dz)
            node_evals["gyyx"] = prism_fxxz(dy, dz, dx)
            node_evals["gxyz"] = prism_fxyz(dx, dy, dz)
        if "bxz" in components or "tmi_x" in components or "tmi_z" in components:
            if "gxxz" not in node_evals:
                node_evals["gxxz"] = prism_fxxz(dx, dy, dz)
            if "gxyz" not in node_evals:
                node_evals["gxyz"] = prism_fxyz(dx, dy, dz)
            node_evals["gzzx"] = prism_fxxy(dz, dx, dy)
        if "byy" in components or "tmi_y" in components:
            if "gyyx" not in node_evals:
                node_evals["gyyx"] = prism_fxxz(dy, dz, dx)
            node_evals["gyyy"] = prism_fzzz(dz, dx, dy)
            node_evals["gyyz"] = prism_fxxy(dy, dz, dx)
        if "byz" in components or "tmi_y" in components or "tmi_z" in components:
            if "gxyz" not in node_evals:
                node_evals["gxyz"] = prism_fxyz(dx, dy, dz)
            if "gyyz" not in node_evals:
                node_evals["gyyz"] = prism_fxxy(dy, dz, dx)
            node_evals["gzzy"] = prism_fxxz(dz, dx, dy)
        if "bzz" in components or "tmi_z" in components:
            if "gzzx" not in node_evals:
                node_evals["gzzx"] = prism_fxxy(dz, dx, dy)
            if "gzzy" not in node_evals:
                node_evals["gzzy"] = prism_fxxz(dz, dx, dy)
            node_evals["gzzz"] = prism_fzzz(dx, dy, dz)

        ## Hxx = gxxx * m_x + gxxy * m_y + gxxz * m_z
        ## Hxy = gxxy * m_x + gyyx * m_y + gxyz * m_z
        ## Hxz = gxxz * m_x + gxyz * m_y + gzzx * m_z
        ## Hyy = gyyx * m_x + gyyy * m_y + gyyz * m_z
        ## Hyz = gxyz * m_x + gyyz * m_y + gzzy * m_z
        ## Hzz = gzzx * m_x + gzzy * m_y + gzzz * m_z

        rows = {}
        M = self.M
        for component in set(components):
            if component == "bx":
                vals_x = node_evals["gxx"]
                vals_y = node_evals["gxy"]
                vals_z = node_evals["gxz"]
            elif component == "by":
                vals_x = node_evals["gxy"]
                vals_y = node_evals["gyy"]
                vals_z = node_evals["gyz"]
            elif component == "bz":
                vals_x = node_evals["gxz"]
                vals_y = node_evals["gyz"]
                vals_z = node_evals["gzz"]
            elif component == "tmi":
                tmi = self.tmi_projection
                vals_x = (
                    tmi[0] * node_evals["gxx"]
                    + tmi[1] * node_evals["gxy"]
                    + tmi[2] * node_evals["gxz"]
                )
                vals_y = (
                    tmi[0] * node_evals["gxy"]
                    + tmi[1] * node_evals["gyy"]
                    + tmi[2] * node_evals["gyz"]
                )
                vals_z = (
                    tmi[0] * node_evals["gxz"]
                    + tmi[1] * node_evals["gyz"]
                    + tmi[2] * node_evals["gzz"]
                )
            elif component == "tmi_x":
                tmi = self.tmi_projection
                vals_x = (
                    tmi[0] * node_evals["gxxx"]
                    + tmi[1] * node_evals["gxxy"]
                    + tmi[2] * node_evals["gxxz"]
                )
                vals_y = (
                    tmi[0] * node_evals["gxxy"]
                    + tmi[1] * node_evals["gyyx"]
                    + tmi[2] * node_evals["gxyz"]
                )
                vals_z = (
                    tmi[0] * node_evals["gxxz"]
                    + tmi[1] * node_evals["gxyz"]
                    + tmi[2] * node_evals["gzzx"]
                )
            elif component == "tmi_y":
                tmi = self.tmi_projection
                vals_x = (
                    tmi[0] * node_evals["gxxy"]
                    + tmi[1] * node_evals["gyyx"]
                    + tmi[2] * node_evals["gxyz"]
                )
                vals_y = (
                    tmi[0] * node_evals["gyyx"]
                    + tmi[1] * node_evals["gyyy"]
                    + tmi[2] * node_evals["gyyz"]
                )
                vals_z = (
                    tmi[0] * node_evals["gxyz"]
                    + tmi[1] * node_evals["gyyz"]
                    + tmi[2] * node_evals["gzzy"]
                )
            elif component == "tmi_z":
                tmi = self.tmi_projection
                vals_x = (
                    tmi[0] * node_evals["gxxz"]
                    + tmi[1] * node_evals["gxyz"]
                    + tmi[2] * node_evals["gzzx"]
                )
                vals_y = (
                    tmi[0] * node_evals["gxyz"]
                    + tmi[1] * node_evals["gyyz"]
                    + tmi[2] * node_evals["gzzy"]
                )
                vals_z = (
                    tmi[0] * node_evals["gzzx"]
                    + tmi[1] * node_evals["gzzy"]
                    + tmi[2] * node_evals["gzzz"]
                )
            elif component == "bxx":
                vals_x = node_evals["gxxx"]
                vals_y = node_evals["gxxy"]
                vals_z = node_evals["gxxz"]
            elif component == "bxy":
                vals_x = node_evals["gxxy"]
                vals_y = node_evals["gyyx"]
                vals_z = node_evals["gxyz"]
            elif component == "bxz":
                vals_x = node_evals["gxxz"]
                vals_y = node_evals["gxyz"]
                vals_z = node_evals["gzzx"]
            elif component == "byy":
                vals_x = node_evals["gyyx"]
                vals_y = node_evals["gyyy"]
                vals_z = node_evals["gyyz"]
            elif component == "byz":
                vals_x = node_evals["gxyz"]
                vals_y = node_evals["gyyz"]
                vals_z = node_evals["gzzy"]
            elif component == "bzz":
                vals_x = node_evals["gzzx"]
                vals_y = node_evals["gzzy"]
                vals_z = node_evals["gzzz"]
            if self._unique_inv is not None:
                vals_x = vals_x[self._unique_inv]
                vals_y = vals_y[self._unique_inv]
                vals_z = vals_z[self._unique_inv]

            cell_eval_x = (
                vals_x[0]
                - vals_x[1]
                - vals_x[2]
                + vals_x[3]
                - vals_x[4]
                + vals_x[5]
                + vals_x[6]
                - vals_x[7]
            )
            cell_eval_y = (
                vals_y[0]
                - vals_y[1]
                - vals_y[2]
                + vals_y[3]
                - vals_y[4]
                + vals_y[5]
                + vals_y[6]
                - vals_y[7]
            )
            cell_eval_z = (
                vals_z[0]
                - vals_z[1]
                - vals_z[2]
                + vals_z[3]
                - vals_z[4]
                + vals_z[5]
                + vals_z[6]
                - vals_z[7]
            )
            if self.model_type == "vector":
                cell_vals = (
                    np.r_[cell_eval_x, cell_eval_y, cell_eval_z]
                ) * self.survey.source_field.amplitude
            else:
                cell_vals = (
                    cell_eval_x * M[:, 0]
                    + cell_eval_y * M[:, 1]
                    + cell_eval_z * M[:, 2]
                )

            if self.store_sensitivities == "forward_only":
                rows[component] = cell_vals @ self.chi
            else:
                rows[component] = cell_vals

            rows[component] /= 4 * np.pi

        return np.stack(
            [
                rows[component].astype(self.sensitivity_dtype, copy=False)
                for component in components
            ]
        )

    @property
    def deleteTheseOnModelUpdate(self):
        deletes = super().deleteTheseOnModelUpdate
        if self.is_amplitude_data:
            deletes = deletes + ["_gtg_diagonal", "_ampDeriv"]
        return deletes

    def _forward(self, model):
        """
        Forward model the fields of active cells in the mesh on receivers.

        Parameters
        ----------
        model : (n_active_cells) or (3 * n_active_cells) array
            Array containing the susceptibilities (scalar) or effective
            susceptibilities (vector) of the active cells in the mesh, in SI
            units.
            Susceptibilities are expected if ``model_type`` is ``"scalar"``,
            and the array should have ``n_active_cells`` elements.
            Effective susceptibilities are expected if ``model_type`` is
            ``"vector"``, and the array should have ``3 * n_active_cells``
            elements.

        Returns
        -------
        (nD, ) array
            Always return a ``np.float64`` array.
        """
        # Gather active nodes and the indices of the nodes for each active cell
        active_nodes, active_cell_nodes = self._get_active_nodes()
        # Get regional field
        regional_field = self.survey.source_field.b0
        # Allocate fields array
        fields = np.zeros(self.survey.nD, dtype=self.sensitivity_dtype)
        # Define the constant factor
        constant_factor = 1 / 4 / np.pi
        # Start computing the fields
        index_offset = 0
        scalar_model = self.model_type == "scalar"
        for components, receivers in self._get_components_and_receivers():
            if not CHOCLO_SUPPORTED_COMPONENTS.issuperset(components):
                raise NotImplementedError(
                    f"Other components besides {CHOCLO_SUPPORTED_COMPONENTS} "
                    "aren't implemented yet."
                )
            n_components = len(components)
            n_rows = n_components * receivers.shape[0]
            for i, component in enumerate(components):
                vector_slice = slice(
                    index_offset + i, index_offset + n_rows, n_components
                )
                if component == "tmi":
                    self._forward_tmi(
                        receivers,
                        active_nodes,
                        model,
                        fields[vector_slice],
                        active_cell_nodes,
                        regional_field,
                        constant_factor,
                        scalar_model,
                    )
                else:
                    kernel_x, kernel_y, kernel_z = CHOCLO_KERNELS[component]
                    self._forward_mag(
                        receivers,
                        active_nodes,
                        model,
                        fields[vector_slice],
                        active_cell_nodes,
                        regional_field,
                        kernel_x,
                        kernel_y,
                        kernel_z,
                        constant_factor,
                        scalar_model,
                    )
            index_offset += n_rows
        return fields

    def _sensitivity_matrix(self):
        """
        Compute the sensitivity matrix G

        Returns
        -------
        (nD, n_active_cells) array
        """
        # Gather active nodes and the indices of the nodes for each active cell
        active_nodes, active_cell_nodes = self._get_active_nodes()
        # Get regional field
        regional_field = self.survey.source_field.b0
        # Allocate sensitivity matrix
        if self.model_type == "scalar":
            n_columns = self.nC
        else:
            n_columns = 3 * self.nC
        shape = (self.survey.nD, n_columns)
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
        # Define the constant factor
        constant_factor = 1 / 4 / np.pi
        # Start filling the sensitivity matrix
        index_offset = 0
        scalar_model = self.model_type == "scalar"
        for components, receivers in self._get_components_and_receivers():
            if not CHOCLO_SUPPORTED_COMPONENTS.issuperset(components):
                raise NotImplementedError(
                    f"Other components besides {CHOCLO_SUPPORTED_COMPONENTS} "
                    "aren't implemented yet."
                )
            n_components = len(components)
            n_rows = n_components * receivers.shape[0]
            for i, component in enumerate(components):
                matrix_slice = slice(
                    index_offset + i, index_offset + n_rows, n_components
                )
                if component == "tmi":
                    self._sensitivity_tmi(
                        receivers,
                        active_nodes,
                        sensitivity_matrix[matrix_slice, :],
                        active_cell_nodes,
                        regional_field,
                        constant_factor,
                        scalar_model,
                    )
                else:
                    kernel_x, kernel_y, kernel_z = CHOCLO_KERNELS[component]
                    self._sensitivity_mag(
                        receivers,
                        active_nodes,
                        sensitivity_matrix[matrix_slice, :],
                        active_cell_nodes,
                        regional_field,
                        kernel_x,
                        kernel_y,
                        kernel_z,
                        constant_factor,
                        scalar_model,
                    )
            index_offset += n_rows
        return sensitivity_matrix


class SimulationEquivalentSourceLayer(
    BaseEquivalentSourceLayerSimulation, Simulation3DIntegral
):
    """
    Equivalent source layer simulation

    Parameters
    ----------
    mesh : discretize.BaseMesh
        A 2D tensor or tree mesh defining discretization along the x and y directions
    cell_z_top : numpy.ndarray or float
        Define the elevations for the top face of all cells in the layer
    cell_z_bottom : numpy.ndarray or float
        Define the elevations for the bottom face of all cells in the layer

    """


class Simulation3DDifferential(BaseMagneticPDESimulation):
    """
    Secondary field approach using differential equations!
    """

    def __init__(self, mesh, survey=None, **kwargs):
        super().__init__(mesh, survey=survey, **kwargs)

        Pbc, Pin, self._Pout = self.mesh.get_BC_projections(
            "neumann", discretization="CC"
        )

        Dface = self.mesh.face_divergence
        Mc = sdiag(self.mesh.cell_volumes)
        self._Div = Mc * Dface * Pin.T.tocsr() * Pin

    @property
    def survey(self):
        """The survey for this simulation.

        Returns
        -------
        simpeg.potential_fields.magnetics.survey.Survey
        """
        if self._survey is None:
            raise AttributeError("Simulation must have a survey")
        return self._survey

    @survey.setter
    def survey(self, obj):
        if obj is not None:
            obj = validate_type("survey", obj, Survey, cast=False)
        self._survey = obj

    @property
    def MfMuI(self):
        return self._MfMuI

    @property
    def MfMui(self):
        return self._MfMui

    @property
    def MfMu0(self):
        return self._MfMu0

    def makeMassMatrices(self, m):
        mu = self.muMap * m
        self._MfMui = self.mesh.get_face_inner_product(1.0 / mu) / self.mesh.dim
        # self._MfMui = self.mesh.get_face_inner_product(1./mu)
        # TODO: this will break if tensor mu
        self._MfMuI = sdiag(1.0 / self._MfMui.diagonal())
        self._MfMu0 = self.mesh.get_face_inner_product(1.0 / mu_0) / self.mesh.dim
        # self._MfMu0 = self.mesh.get_face_inner_product(1/mu_0)

    @utils.requires("survey")
    def getB0(self):
        b0 = self.survey.source_field.b0
        B0 = np.r_[
            b0[0] * np.ones(self.mesh.nFx),
            b0[1] * np.ones(self.mesh.nFy),
            b0[2] * np.ones(self.mesh.nFz),
        ]
        return B0

    def getRHS(self, m):
        r"""

        .. math ::

            \mathbf{rhs} =
                \Div(\MfMui)^{-1}\mathbf{M}^f_{\mu_0^{-1}}\mathbf{B}_0
                - \Div\mathbf{B}_0
                +\diag(v)\mathbf{D} \mathbf{P}_{out}^T \mathbf{B}_{sBC}

        """
        B0 = self.getB0()

        mu = self.muMap * m
        chi = mu / mu_0 - 1

        # Temporary fix
        Bbc, Bbc_const = CongruousMagBC(self.mesh, self.survey.source_field.b0, chi)
        self.Bbc = Bbc
        self.Bbc_const = Bbc_const
        # return self._Div*self.MfMuI*self.MfMu0*B0 - self._Div*B0 +
        # Mc*Dface*self._Pout.T*Bbc
        return self._Div * self.MfMuI * self.MfMu0 * B0 - self._Div * B0

    def getA(self, m):
        r"""
        GetA creates and returns the A matrix for the Magnetics problem

        The A matrix has the form:

        .. math ::

            \mathbf{A} =  \Div(\MfMui)^{-1}\Div^{T}

        """
        return self._Div * self.MfMuI * self._Div.T.tocsr()

    def fields(self, m):
        r"""
        Return magnetic potential (u) and flux (B)

        u: defined on the cell center [nC x 1]
        B: defined on the cell center [nG x 1]

        After we compute u, then we update B.

        .. math ::

            \mathbf{B}_s =
                (\MfMui)^{-1}\mathbf{M}^f_{\mu_0^{-1}}\mathbf{B}_0
                - \mathbf{B}_0
                - (\MfMui)^{-1}\Div^T \mathbf{u}

        """
        self.makeMassMatrices(m)
        A = self.getA(m)
        rhs = self.getRHS(m)
        Ainv = self.solver(A, **self.solver_opts)
        u = Ainv * rhs
        B0 = self.getB0()
        B = self.MfMuI * self.MfMu0 * B0 - B0 - self.MfMuI * self._Div.T * u
        Ainv.clean()

        return {"B": B, "u": u}

    @utils.timeIt
    def Jvec(self, m, v, u=None):
        r"""
        Computing Jacobian multiplied by vector

        By setting our problem as

        .. math ::

            \mathbf{C}(\mathbf{m}, \mathbf{u}) = \mathbf{A}\mathbf{u} - \mathbf{rhs} = 0

        And taking derivative w.r.t m

        .. math ::

            \nabla \mathbf{C}(\mathbf{m}, \mathbf{u}) =
                \nabla_m \mathbf{C}(\mathbf{m}) \delta \mathbf{m} +
                \nabla_u \mathbf{C}(\mathbf{u}) \delta \mathbf{u} = 0

            \frac{\delta \mathbf{u}}{\delta \mathbf{m}} =
                - [\nabla_u \mathbf{C}(\mathbf{u})]^{-1}\nabla_m \mathbf{C}(\mathbf{m})

        With some linear algebra we can have

        .. math ::

            \nabla_u \mathbf{C}(\mathbf{u}) = \mathbf{A}

            \nabla_m \mathbf{C}(\mathbf{m}) =
                \frac{\partial \mathbf{A}} {\partial \mathbf{m}} (\mathbf{m}) \mathbf{u}
                - \frac{\partial \mathbf{rhs}(\mathbf{m})}{\partial \mathbf{m}}

        .. math ::

            \frac{\partial \mathbf{A}}{\partial \mathbf{m}}(\mathbf{m})\mathbf{u} =
                \frac{\partial \mathbf{\mu}}{\partial \mathbf{m}}
                \left[\Div \diag (\Div^T \mathbf{u}) \dMfMuI \right]

            \dMfMuI =
                \diag(\MfMui)^{-1}_{vec}
                \mathbf{Av}_{F2CC}^T\diag(\mathbf{v})\diag(\frac{1}{\mu^2})

            \frac{\partial \mathbf{rhs}(\mathbf{m})}{\partial \mathbf{m}} =
                \frac{\partial \mathbf{\mu}}{\partial \mathbf{m}}
                \left[
                    \Div \diag(\M^f_{\mu_{0}^{-1}}\mathbf{B}_0) \dMfMuI
                \right]
                - \diag(\mathbf{v}) \mathbf{D} \mathbf{P}_{out}^T
                    \frac{\partial B_{sBC}}{\partial \mathbf{m}}

        In the end,

        .. math ::

            \frac{\delta \mathbf{u}}{\delta \mathbf{m}} =
            - [ \mathbf{A} ]^{-1}
            \left[
                \frac{\partial \mathbf{A}}{\partial \mathbf{m}}(\mathbf{m})\mathbf{u}
                - \frac{\partial \mathbf{rhs}(\mathbf{m})}{\partial \mathbf{m}}
            \right]

        A little tricky point here is we are not interested in potential (u), but interested in magnetic flux (B).
        Thus, we need sensitivity for B. Now we take derivative of B w.r.t m and have

        .. math ::

            \frac{\delta \mathbf{B}} {\delta \mathbf{m}} =
            \frac{\partial \mathbf{\mu} } {\partial \mathbf{m} }
            \left[
                \diag(\M^f_{\mu_{0}^{-1} } \mathbf{B}_0) \dMfMuI  \
                 - \diag (\Div^T\mathbf{u})\dMfMuI
            \right ]

             -  (\MfMui)^{-1}\Div^T\frac{\delta\mathbf{u}}{\delta \mathbf{m}}

        Finally we evaluate the above, but we should remember that

        .. note ::

            We only want to evaluate

            .. math ::

                \mathbf{J}\mathbf{v} =
                    \frac{\delta \mathbf{P}\mathbf{B}} {\delta \mathbf{m}}\mathbf{v}

            Since forming sensitivity matrix is very expensive in that this
            monster is "big" and "dense" matrix!!

        """
        if u is None:
            u = self.fields(m)

        B, u = u["B"], u["u"]
        mu = self.muMap * (m)
        dmu_dm = self.muDeriv
        # dchidmu = sdiag(1 / mu_0 * np.ones(self.mesh.nC))

        vol = self.mesh.cell_volumes
        Div = self._Div
        P = self.survey.projectFieldsDeriv(B)  # Projection matrix
        B0 = self.getB0()

        MfMuIvec = 1 / self.MfMui.diagonal()
        dMfMuI = sdiag(MfMuIvec**2) * self.mesh.aveF2CC.T * sdiag(vol * 1.0 / mu**2)

        # A = self._Div*self.MfMuI*self._Div.T
        # RHS = Div*MfMuI*MfMu0*B0 - Div*B0 + Mc*Dface*Pout.T*Bbc
        # C(m,u) = A*m-rhs
        # dudm = -(dCdu)^(-1)dCdm

        dCdu = self.getA(m)  # = A
        dCdm_A = Div * (sdiag(Div.T * u) * dMfMuI * dmu_dm)
        dCdm_RHS1 = Div * (sdiag(self.MfMu0 * B0) * dMfMuI)
        # temp1 = (Dface * (self._Pout.T * self.Bbc_const * self.Bbc))
        # dCdm_RHS2v = (sdiag(vol) * temp1) * \
        #    np.inner(vol, dchidmu * dmu_dm * v)

        # dCdm_RHSv =  dCdm_RHS1*(dmu_dm*v) +  dCdm_RHS2v
        dCdm_RHSv = dCdm_RHS1 * (dmu_dm * v)
        dCdm_v = dCdm_A * v - dCdm_RHSv

        Ainv = self.solver(dCdu, **self.solver_opts)
        sol = Ainv * dCdm_v

        dudm = -sol
        dBdmv = (
            sdiag(self.MfMu0 * B0) * (dMfMuI * (dmu_dm * v))
            - sdiag(Div.T * u) * (dMfMuI * (dmu_dm * v))
            - self.MfMuI * (Div.T * (dudm))
        )

        Ainv.clean()

        return mkvc(P * dBdmv)

    @utils.timeIt
    def Jtvec(self, m, v, u=None):
        r"""
        Computing Jacobian^T multiplied by vector.

        .. math ::

            (\frac{\delta \mathbf{P}\mathbf{B}} {\delta \mathbf{m}})^{T} =
                \left[
                    \mathbf{P}_{deriv}\frac{\partial \mathbf{\mu} } {\partial \mathbf{m} }
                    \left[
                        \diag(\M^f_{\mu_{0}^{-1} } \mathbf{B}_0) \dMfMuI
                         - \diag (\Div^T\mathbf{u})\dMfMuI
                    \right ]
                \right]^{T}
                 -
                 \left[
                     \mathbf{P}_{deriv}(\MfMui)^{-1} \Div^T
                     \frac{\delta\mathbf{u}}{\delta \mathbf{m}}
                 \right]^{T}

        where

        .. math ::

            \mathbf{P}_{derv} = \frac{\partial \mathbf{P}}{\partial\mathbf{B}}

        .. note ::

            Here we only want to compute

            .. math ::

                \mathbf{J}^{T}\mathbf{v} =
                (\frac{\delta \mathbf{P}\mathbf{B}} {\delta \mathbf{m}})^{T} \mathbf{v}

        """
        if u is None:
            u = self.fields(m)

        B, u = u["B"], u["u"]
        mu = self.mapping * (m)
        dmu_dm = self.mapping.deriv(m)
        # dchidmu = sdiag(1 / mu_0 * np.ones(self.mesh.nC))

        vol = self.mesh.cell_volumes
        Div = self._Div
        P = self.survey.projectFieldsDeriv(B)  # Projection matrix
        B0 = self.getB0()

        MfMuIvec = 1 / self.MfMui.diagonal()
        dMfMuI = sdiag(MfMuIvec**2) * self.mesh.aveF2CC.T * sdiag(vol * 1.0 / mu**2)

        # A = self._Div*self.MfMuI*self._Div.T
        # RHS = Div*MfMuI*MfMu0*B0 - Div*B0 + Mc*Dface*Pout.T*Bbc
        # C(m,u) = A*m-rhs
        # dudm = -(dCdu)^(-1)dCdm

        dCdu = self.getA(m)
        s = Div * (self.MfMuI.T * (P.T * v))

        Ainv = self.solver(dCdu.T, **self.solver_opts)
        sol = Ainv * s

        Ainv.clean()

        # dCdm_A = Div * ( sdiag( Div.T * u )* dMfMuI *dmu_dm  )
        # dCdm_Atsol = ( dMfMuI.T*( sdiag( Div.T * u ) * (Div.T * dmu_dm)) ) * sol
        dCdm_Atsol = (dmu_dm.T * dMfMuI.T * (sdiag(Div.T * u) * Div.T)) * sol

        # dCdm_RHS1 = Div * (sdiag( self.MfMu0*B0  ) * dMfMuI)
        # dCdm_RHS1tsol = (dMfMuI.T*( sdiag( self.MfMu0*B0  ) ) * Div.T * dmu_dm) * sol
        dCdm_RHS1tsol = (dmu_dm.T * dMfMuI.T * (sdiag(self.MfMu0 * B0)) * Div.T) * sol

        # temp1 = (Dface*(self._Pout.T*self.Bbc_const*self.Bbc))
        # temp1sol = (Dface.T * (sdiag(vol) * sol))
        # temp2 = self.Bbc_const * (self._Pout.T * self.Bbc).T
        # dCdm_RHS2v  = (sdiag(vol)*temp1)*np.inner(vol, dchidmu*dmu_dm*v)
        # dCdm_RHS2tsol = (dmu_dm.T * dchidmu.T * vol) * np.inner(temp2, temp1sol)

        # dCdm_RHSv =  dCdm_RHS1*(dmu_dm*v) +  dCdm_RHS2v

        # temporary fix
        # dCdm_RHStsol = dCdm_RHS1tsol - dCdm_RHS2tsol
        dCdm_RHStsol = dCdm_RHS1tsol

        # dCdm_RHSv =  dCdm_RHS1*(dmu_dm*v) +  dCdm_RHS2v
        # dCdm_v = dCdm_A*v - dCdm_RHSv

        Ctv = dCdm_Atsol - dCdm_RHStsol

        # B = self.MfMuI*self.MfMu0*B0-B0-self.MfMuI*self._Div.T*u
        # dBdm = d\mudm*dBd\mu
        # dPBdm^T*v = Atemp^T*P^T*v - Btemp^T*P^T*v - Ctv

        Atemp = sdiag(self.MfMu0 * B0) * (dMfMuI * (dmu_dm))
        Btemp = sdiag(Div.T * u) * (dMfMuI * (dmu_dm))
        Jtv = Atemp.T * (P.T * v) - Btemp.T * (P.T * v) - Ctv

        return mkvc(Jtv)

    @property
    def Qfx(self):
        if getattr(self, "_Qfx", None) is None:
            self._Qfx = self.mesh.get_interpolation_matrix(
                self.survey.receiver_locations, "Fx"
            )
        return self._Qfx

    @property
    def Qfy(self):
        if getattr(self, "_Qfy", None) is None:
            self._Qfy = self.mesh.get_interpolation_matrix(
                self.survey.receiver_locations, "Fy"
            )
        return self._Qfy

    @property
    def Qfz(self):
        if getattr(self, "_Qfz", None) is None:
            self._Qfz = self.mesh.get_interpolation_matrix(
                self.survey.receiver_locations, "Fz"
            )
        return self._Qfz

    def projectFields(self, u):
        r"""
        This function projects the fields onto the data space.
        Especially, here for we use total magnetic intensity (TMI) data,
        which is common in practice.
        First we project our B on to data location

        .. math::

            \mathbf{B}_{rec} = \mathbf{P} \mathbf{B}

        then we take the dot product between B and b_0

        .. math ::

            \text{TMI} = \vec{B}_s \cdot \hat{B}_0

        """
        # TODO: There can be some different tyes of data like |B| or B
        components = self.survey.components

        fields = {}
        if "bx" in components or "tmi" in components:
            fields["bx"] = self.Qfx * u["B"]
        if "by" in components or "tmi" in components:
            fields["by"] = self.Qfy * u["B"]
        if "bz" in components or "tmi" in components:
            fields["bz"] = self.Qfz * u["B"]

        if "tmi" in components:
            bx = fields["bx"]
            by = fields["by"]
            bz = fields["bz"]
            # Generate unit vector
            B0 = self.survey.source_field.b0
            Bot = np.sqrt(B0[0] ** 2 + B0[1] ** 2 + B0[2] ** 2)
            box = B0[0] / Bot
            boy = B0[1] / Bot
            boz = B0[2] / Bot
            fields["tmi"] = bx * box + by * boy + bz * boz

        return np.concatenate([fields[comp] for comp in components])

    @utils.count
    def projectFieldsDeriv(self, B):
        r"""
        This function projects the fields onto the data space.

        .. math::

            \frac{\partial d_\text{pred}}{\partial \mathbf{B}} = \mathbf{P}

        Especially, this function is for TMI data type
        """

        components = self.survey.components

        fields = {}
        if "bx" in components or "tmi" in components:
            fields["bx"] = self.Qfx
        if "by" in components or "tmi" in components:
            fields["by"] = self.Qfy
        if "bz" in components or "tmi" in components:
            fields["bz"] = self.Qfz

        if "tmi" in components:
            bx = fields["bx"]
            by = fields["by"]
            bz = fields["bz"]
            # Generate unit vector
            B0 = self.survey.source_field.b0
            Bot = np.sqrt(B0[0] ** 2 + B0[1] ** 2 + B0[2] ** 2)
            box = B0[0] / Bot
            boy = B0[1] / Bot
            boz = B0[2] / Bot
            fields["tmi"] = bx * box + by * boy + bz * boz

        return sp.vstack([fields[comp] for comp in components])

    def projectFieldsAsVector(self, B):
        bfx = self.Qfx * B
        bfy = self.Qfy * B
        bfz = self.Qfz * B

        return np.r_[bfx, bfy, bfz]


def MagneticsDiffSecondaryInv(mesh, model, data, **kwargs):
    """
    Inversion module for MagneticsDiffSecondary

    """
    from simpeg import (
        directives,
        inversion,
        objective_function,
        optimization,
        regularization,
    )

    prob = Simulation3DDifferential(mesh, survey=data, mu=model)

    miter = kwargs.get("maxIter", 10)

    # Create an optimization program
    opt = optimization.InexactGaussNewton(maxIter=miter)
    opt.bfgsH0 = Solver(sp.identity(model.nP), flag="D")
    # Create a regularization program
    reg = regularization.WeightedLeastSquares(model)
    # Create an objective function
    beta = directives.BetaSchedule(beta0=1e0)
    obj = objective_function.BaseObjFunction(prob, reg, beta=beta)
    # Create an inversion object
    inv = inversion.BaseInversion(obj, opt)

    return inv, reg
