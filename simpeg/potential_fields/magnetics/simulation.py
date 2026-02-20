import hashlib
import warnings
import numpy as np
from numpy.typing import NDArray
import scipy.sparse as sp
from functools import cached_property
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
from scipy.sparse.linalg import LinearOperator, aslinearoperator

from simpeg import props, utils
from simpeg.utils import mat_utils, mkvc, sdiag
from simpeg.utils.code_utils import validate_string, validate_type

from ...base import BaseMagneticPDESimulation
from ..base import BaseEquivalentSourceLayerSimulation, BasePFSimulation
from .survey import Survey

from ._numba import choclo, NUMBA_FUNCTIONS_3D, NUMBA_FUNCTIONS_2D

if choclo is not None:
    CHOCLO_SUPPORTED_COMPONENTS = {
        "tmi",
        "bx",
        "by",
        "bz",
        "bxx",
        "byy",
        "bzz",
        "bxy",
        "bxz",
        "byz",
        "tmi_x",
        "tmi_y",
        "tmi_z",
    }
    CHOCLO_KERNELS = {
        "bx": (choclo.prism.kernel_ee, choclo.prism.kernel_en, choclo.prism.kernel_eu),
        "by": (choclo.prism.kernel_en, choclo.prism.kernel_nn, choclo.prism.kernel_nu),
        "bz": (choclo.prism.kernel_eu, choclo.prism.kernel_nu, choclo.prism.kernel_uu),
        "bxx": (
            choclo.prism.kernel_eee,
            choclo.prism.kernel_een,
            choclo.prism.kernel_eeu,
        ),
        "byy": (
            choclo.prism.kernel_enn,
            choclo.prism.kernel_nnn,
            choclo.prism.kernel_nnu,
        ),
        "bzz": (
            choclo.prism.kernel_euu,
            choclo.prism.kernel_nuu,
            choclo.prism.kernel_uuu,
        ),
        "bxy": (
            choclo.prism.kernel_een,
            choclo.prism.kernel_enn,
            choclo.prism.kernel_enu,
        ),
        "bxz": (
            choclo.prism.kernel_eeu,
            choclo.prism.kernel_enu,
            choclo.prism.kernel_euu,
        ),
        "byz": (
            choclo.prism.kernel_enu,
            choclo.prism.kernel_nnu,
            choclo.prism.kernel_nuu,
        ),
        "tmi_x": (
            choclo.prism.kernel_eee,
            choclo.prism.kernel_enn,
            choclo.prism.kernel_euu,
            choclo.prism.kernel_een,
            choclo.prism.kernel_eeu,
            choclo.prism.kernel_enu,
        ),
        "tmi_y": (
            choclo.prism.kernel_een,
            choclo.prism.kernel_nnn,
            choclo.prism.kernel_nuu,
            choclo.prism.kernel_enn,
            choclo.prism.kernel_enu,
            choclo.prism.kernel_nnu,
        ),
        "tmi_z": (
            choclo.prism.kernel_eeu,
            choclo.prism.kernel_nnu,
            choclo.prism.kernel_uuu,
            choclo.prism.kernel_enu,
            choclo.prism.kernel_euu,
            choclo.prism.kernel_nuu,
        ),
    }
    CHOCLO_FORWARD_FUNCS = {
        "bx": choclo.prism.magnetic_e,
        "by": choclo.prism.magnetic_n,
        "bz": choclo.prism.magnetic_u,
        "bxx": choclo.prism.magnetic_ee,
        "byy": choclo.prism.magnetic_nn,
        "bzz": choclo.prism.magnetic_uu,
        "bxy": choclo.prism.magnetic_en,
        "bxz": choclo.prism.magnetic_eu,
        "byz": choclo.prism.magnetic_nu,
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
    active_cells : (n_cells) numpy.ndarray, optional
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

        self._M = None
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
    def G(self) -> NDArray | np.memmap | LinearOperator:
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
    def nD(self):
        """
        Number of data
        """
        self._nD = self.survey.nD if not self.is_amplitude_data else self.survey.nD // 3
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
        (nD, n_params) np.ndarray or scipy.sparse.linalg.LinearOperator.
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
        if self.is_amplitude_data:
            msg = (
                "The `getJ` method is not yet implemented to work with "
                "`is_amplitude_data`."
            )
            raise NotImplementedError(msg)

        # Need to assign the model, so the chiDeriv can be computed (if the
        # model is None, the chiDeriv is going to be Zero).
        self.model = m
        chiDeriv = (
            self.chiDeriv
            if not isinstance(self.G, LinearOperator)
            else aslinearoperator(self.chiDeriv)
        )
        return self.G @ chiDeriv

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
        (nparam) np.ndarray
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
        # Need to assign the model, so the chiDeriv can be computed (if the
        # model is None, the chiDeriv is going to be Zero).
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
            (sdiag(np.sqrt(self._gtg_diagonal)) @ self.chiDeriv).power(2).sum(axis=0)
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
        match (self.engine, self.store_sensitivities, self.is_amplitude_data):
            case ("geoana", "forward_only", _):
                msg = (
                    "Computing the diagonal of `G.T @ G` using "
                    "`'forward_only'` and `'geoana'` as engine hasn't been "
                    "implemented yet."
                )
                raise NotImplementedError(msg)
            case ("choclo", "forward_only", True):
                msg = (
                    "Computing the diagonal of `G.T @ G` using "
                    "`'forward_only'` and `is_amplitude_data` hasn't been "
                    "implemented yet."
                )
                raise NotImplementedError(msg)
            case ("choclo", "forward_only", False):
                gtg_diagonal = self._gtg_diagonal_without_building_g(weights)
            case (_, _, False):
                # In Einstein notation, the j-th element of the diagonal is:
                #   d_j = w_i * G_{ij} * G_{ij}
                gtg_diagonal = np.asarray(
                    np.einsum("i,ij,ij->j", weights, self.G, self.G)
                )
            case (_, _, True):
                ampDeriv = self.ampDeriv
                Gx = self.G[::3]
                Gy = self.G[1::3]
                Gz = self.G[2::3]
                gtg_diagonal = np.zeros(self.G.shape[1])
                for i in range(weights.size):
                    row = (
                        ampDeriv[0, i] * Gx[i]
                        + ampDeriv[1, i] * Gy[i]
                        + ampDeriv[2, i] * Gz[i]
                    )
                    gtg_diagonal += weights[i] * (row * row)
        return gtg_diagonal

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
        # Need to assign the model, so the chiDeriv can be computed (if the
        # model is None, the chiDeriv is going to be Zero).
        self.model = m
        dmu_dm_v = self.chiDeriv @ v
        Jvec = self.G @ dmu_dm_v.astype(self.sensitivity_dtype, copy=False)

        if self.is_amplitude_data:
            # dask doesn't support an "order" argument to reshape...
            Jvec = Jvec.reshape((-1, 3)).T  # reshape((3, -1), order="F")
            ampDeriv_Jvec = self.ampDeriv * Jvec
            return ampDeriv_Jvec[0] + ampDeriv_Jvec[1] + ampDeriv_Jvec[2]

        return Jvec

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
        # Need to assign the model, so the chiDeriv can be computed (if the
        # model is None, the chiDeriv is going to be Zero).
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
    def _delete_on_model_update(self):
        deletes = super()._delete_on_model_update
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
                    forward_func = NUMBA_FUNCTIONS_3D["forward"]["tmi"][
                        self.numba_parallel
                    ]
                    forward_func(
                        receivers,
                        active_nodes,
                        model,
                        fields[vector_slice],
                        active_cell_nodes,
                        regional_field,
                        constant_factor,
                        scalar_model,
                    )
                elif component in ("tmi_x", "tmi_y", "tmi_z"):
                    kernel_xx, kernel_yy, kernel_zz, kernel_xy, kernel_xz, kernel_yz = (
                        CHOCLO_KERNELS[component]
                    )
                    forward_func = NUMBA_FUNCTIONS_3D["forward"]["tmi_derivative"][
                        self.numba_parallel
                    ]
                    forward_func(
                        receivers,
                        active_nodes,
                        model,
                        fields[vector_slice],
                        active_cell_nodes,
                        regional_field,
                        kernel_xx,
                        kernel_yy,
                        kernel_zz,
                        kernel_xy,
                        kernel_xz,
                        kernel_yz,
                        constant_factor,
                        scalar_model,
                    )
                else:
                    kernel_x, kernel_y, kernel_z = CHOCLO_KERNELS[component]
                    forward_func = NUMBA_FUNCTIONS_3D["forward"]["magnetic_component"][
                        self.numba_parallel
                    ]
                    forward_func(
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
        scalar_model = self.model_type == "scalar"
        n_columns = self.nC if scalar_model else 3 * self.nC
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
                    sensitivity_func = NUMBA_FUNCTIONS_3D["sensitivity"]["tmi"][
                        self.numba_parallel
                    ]
                    sensitivity_func(
                        receivers,
                        active_nodes,
                        sensitivity_matrix[matrix_slice, :],
                        active_cell_nodes,
                        regional_field,
                        constant_factor,
                        scalar_model,
                    )
                elif component in ("tmi_x", "tmi_y", "tmi_z"):
                    sensitivity_func = NUMBA_FUNCTIONS_3D["sensitivity"][
                        "tmi_derivative"
                    ][self.numba_parallel]
                    kernel_xx, kernel_yy, kernel_zz, kernel_xy, kernel_xz, kernel_yz = (
                        CHOCLO_KERNELS[component]
                    )
                    sensitivity_func(
                        receivers,
                        active_nodes,
                        sensitivity_matrix[matrix_slice, :],
                        active_cell_nodes,
                        regional_field,
                        kernel_xx,
                        kernel_yy,
                        kernel_zz,
                        kernel_xy,
                        kernel_xz,
                        kernel_yz,
                        constant_factor,
                        scalar_model,
                    )
                else:
                    sensitivity_func = NUMBA_FUNCTIONS_3D["sensitivity"][
                        "magnetic_component"
                    ][self.numba_parallel]
                    kernel_x, kernel_y, kernel_z = CHOCLO_KERNELS[component]
                    sensitivity_func(
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

    def _sensitivity_matrix_as_operator(self):
        """
        Create a LinearOperator for the sensitivity matrix G.

        Returns
        -------
        scipy.sparse.linalg.LinearOperator
        """
        n_columns = self.nC if self.model_type == "scalar" else self.nC * 3
        shape = (self.survey.nD, n_columns)
        linear_op = LinearOperator(
            shape=shape,
            matvec=self._forward,
            rmatvec=self._sensitivity_matrix_transpose_dot_vec,
            dtype=np.float64,
        )
        return linear_op

    def _sensitivity_matrix_transpose_dot_vec(self, vector):
        """
        Compute ``G.T @ v`` without building ``G``.

        Parameters
        ----------
        vector : (nD) numpy.ndarray
            Vector used in the dot product.

        Returns
        -------
        (n_active_cells) or (3 * n_active_cells) numpy.ndarray
        """
        # Gather active nodes and the indices of the nodes for each active cell
        active_nodes, active_cell_nodes = self._get_active_nodes()
        # Get regional field
        regional_field = self.survey.source_field.b0

        # Allocate resulting array.
        scalar_model = self.model_type == "scalar"
        result = np.zeros(self.nC if scalar_model else 3 * self.nC)

        # Define the constant factor
        constant_factor = 1 / 4 / np.pi

        # Fill the result array
        index_offset = 0
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
                    gt_dot_v_func = NUMBA_FUNCTIONS_3D["gt_dot_v"]["tmi"][
                        self.numba_parallel
                    ]
                    gt_dot_v_func(
                        receivers,
                        active_nodes,
                        active_cell_nodes,
                        regional_field,
                        constant_factor,
                        scalar_model,
                        vector[vector_slice],
                        result,
                    )
                elif component in ("tmi_x", "tmi_y", "tmi_z"):
                    kernel_xx, kernel_yy, kernel_zz, kernel_xy, kernel_xz, kernel_yz = (
                        CHOCLO_KERNELS[component]
                    )
                    gt_dot_v_func = NUMBA_FUNCTIONS_3D["gt_dot_v"]["tmi_derivative"][
                        self.numba_parallel
                    ]
                    gt_dot_v_func(
                        receivers,
                        active_nodes,
                        active_cell_nodes,
                        regional_field,
                        kernel_xx,
                        kernel_yy,
                        kernel_zz,
                        kernel_xy,
                        kernel_xz,
                        kernel_yz,
                        constant_factor,
                        scalar_model,
                        vector[vector_slice],
                        result,
                    )
                else:
                    kernel_x, kernel_y, kernel_z = CHOCLO_KERNELS[component]
                    gt_dot_v_func = NUMBA_FUNCTIONS_3D["gt_dot_v"][
                        "magnetic_component"
                    ][self.numba_parallel]
                    gt_dot_v_func(
                        receivers,
                        active_nodes,
                        active_cell_nodes,
                        regional_field,
                        kernel_x,
                        kernel_y,
                        kernel_z,
                        constant_factor,
                        scalar_model,
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
        # Gather active nodes and the indices of the nodes for each active cell
        active_nodes, active_cell_nodes = self._get_active_nodes()
        # Get regional field
        regional_field = self.survey.source_field.b0
        # Define the constant factor
        constant_factor = 1 / 4 / np.pi

        # Allocate array for the diagonal
        scalar_model = self.model_type == "scalar"
        n_columns = self.nC if scalar_model else 3 * self.nC
        diagonal = np.zeros(n_columns, dtype=np.float64)

        # Start filling the diagonal array
        for components, receivers in self._get_components_and_receivers():
            if not CHOCLO_SUPPORTED_COMPONENTS.issuperset(components):
                raise NotImplementedError(
                    f"Other components besides {CHOCLO_SUPPORTED_COMPONENTS} "
                    "aren't implemented yet."
                )
            for component in components:
                if component == "tmi":
                    diagonal_gtg_func = NUMBA_FUNCTIONS_3D["diagonal_gtg"]["tmi"][
                        self.numba_parallel
                    ]
                    diagonal_gtg_func(
                        receivers,
                        active_nodes,
                        active_cell_nodes,
                        regional_field,
                        constant_factor,
                        scalar_model,
                        weights,
                        diagonal,
                    )
                elif component in ("tmi_x", "tmi_y", "tmi_z"):
                    kernel_xx, kernel_yy, kernel_zz, kernel_xy, kernel_xz, kernel_yz = (
                        CHOCLO_KERNELS[component]
                    )
                    diagonal_gtg_func = NUMBA_FUNCTIONS_3D["diagonal_gtg"][
                        "tmi_derivative"
                    ][self.numba_parallel]
                    diagonal_gtg_func(
                        receivers,
                        active_nodes,
                        active_cell_nodes,
                        regional_field,
                        kernel_xx,
                        kernel_yy,
                        kernel_zz,
                        kernel_xy,
                        kernel_xz,
                        kernel_yz,
                        constant_factor,
                        scalar_model,
                        weights,
                        diagonal,
                    )
                else:
                    kernel_x, kernel_y, kernel_z = CHOCLO_KERNELS[component]
                    diagonal_gtg_func = NUMBA_FUNCTIONS_3D["diagonal_gtg"][
                        "magnetic_component"
                    ][self.numba_parallel]
                    diagonal_gtg_func(
                        receivers,
                        active_nodes,
                        active_cell_nodes,
                        regional_field,
                        kernel_x,
                        kernel_y,
                        kernel_z,
                        constant_factor,
                        scalar_model,
                        weights,
                        diagonal,
                    )
        return diagonal


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
        # Get cells in the 2D mesh and keep only active cells
        cells_bounds_active = self.mesh.cell_bounds[self.active_cells]
        # Get regional field
        regional_field = self.survey.source_field.b0
        # Allocate fields array
        fields = np.zeros(self.survey.nD, dtype=self.sensitivity_dtype)
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
                    forward_func = NUMBA_FUNCTIONS_2D["forward"]["tmi"][
                        self.numba_parallel
                    ]
                    forward_func(
                        receivers,
                        cells_bounds_active,
                        self.cell_z_top,
                        self.cell_z_bottom,
                        model,
                        fields[vector_slice],
                        regional_field,
                        scalar_model,
                    )
                elif component in ("tmi_x", "tmi_y", "tmi_z"):
                    kernel_xx, kernel_yy, kernel_zz, kernel_xy, kernel_xz, kernel_yz = (
                        CHOCLO_KERNELS[component]
                    )
                    forward_func = NUMBA_FUNCTIONS_2D["forward"]["tmi_derivative"][
                        self.numba_parallel
                    ]
                    forward_func(
                        receivers,
                        cells_bounds_active,
                        self.cell_z_top,
                        self.cell_z_bottom,
                        model,
                        fields[vector_slice],
                        regional_field,
                        kernel_xx,
                        kernel_yy,
                        kernel_zz,
                        kernel_xy,
                        kernel_xz,
                        kernel_yz,
                        scalar_model,
                    )
                else:
                    choclo_forward_func = CHOCLO_FORWARD_FUNCS[component]
                    forward_func = NUMBA_FUNCTIONS_2D["forward"]["magnetic_component"][
                        self.numba_parallel
                    ]
                    forward_func(
                        receivers,
                        cells_bounds_active,
                        self.cell_z_top,
                        self.cell_z_bottom,
                        model,
                        fields[vector_slice],
                        regional_field,
                        choclo_forward_func,
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
        # Get cells in the 2D mesh and keep only active cells
        cells_bounds_active = self.mesh.cell_bounds[self.active_cells]
        # Get regional field
        regional_field = self.survey.source_field.b0
        # Allocate sensitivity matrix
        scalar_model = self.model_type == "scalar"
        n_columns = self.nC if scalar_model else 3 * self.nC
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
        # Start filling the sensitivity matrix
        index_offset = 0
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
                    sensitivity_func = NUMBA_FUNCTIONS_2D["sensitivity"]["tmi"][
                        self.numba_parallel
                    ]
                    sensitivity_func(
                        receivers,
                        cells_bounds_active,
                        self.cell_z_top,
                        self.cell_z_bottom,
                        sensitivity_matrix[matrix_slice, :],
                        regional_field,
                        scalar_model,
                    )
                elif component in ("tmi_x", "tmi_y", "tmi_z"):
                    kernel_xx, kernel_yy, kernel_zz, kernel_xy, kernel_xz, kernel_yz = (
                        CHOCLO_KERNELS[component]
                    )
                    sensitivity_func = NUMBA_FUNCTIONS_2D["sensitivity"][
                        "tmi_derivative"
                    ][self.numba_parallel]
                    sensitivity_func(
                        receivers,
                        cells_bounds_active,
                        self.cell_z_top,
                        self.cell_z_bottom,
                        sensitivity_matrix[matrix_slice, :],
                        regional_field,
                        kernel_xx,
                        kernel_yy,
                        kernel_zz,
                        kernel_xy,
                        kernel_xz,
                        kernel_yz,
                        scalar_model,
                    )
                else:
                    kernel_x, kernel_y, kernel_z = CHOCLO_KERNELS[component]
                    sensitivity_func = NUMBA_FUNCTIONS_2D["sensitivity"][
                        "magnetic_component"
                    ][self.numba_parallel]
                    sensitivity_func(
                        receivers,
                        cells_bounds_active,
                        self.cell_z_top,
                        self.cell_z_bottom,
                        sensitivity_matrix[matrix_slice, :],
                        regional_field,
                        kernel_x,
                        kernel_y,
                        kernel_z,
                        scalar_model,
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
        (n_active_cells) or (3 * n_active_cells) numpy.ndarray
        """
        # Get regional field
        regional_field = self.survey.source_field.b0
        # Get cells in the 2D mesh and keep only active cells
        cells_bounds_active = self.mesh.cell_bounds[self.active_cells]
        # Allocate resulting array
        scalar_model = self.model_type == "scalar"
        result = np.zeros(self.nC if scalar_model else 3 * self.nC)
        # Start filling the result array
        index_offset = 0
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
                    gt_dot_v_func = NUMBA_FUNCTIONS_2D["gt_dot_v"]["tmi"][
                        self.numba_parallel
                    ]
                    gt_dot_v_func(
                        receivers,
                        cells_bounds_active,
                        self.cell_z_top,
                        self.cell_z_bottom,
                        regional_field,
                        scalar_model,
                        vector[vector_slice],
                        result,
                    )
                elif component in ("tmi_x", "tmi_y", "tmi_z"):
                    kernel_xx, kernel_yy, kernel_zz, kernel_xy, kernel_xz, kernel_yz = (
                        CHOCLO_KERNELS[component]
                    )
                    gt_dot_v_func = NUMBA_FUNCTIONS_2D["gt_dot_v"]["tmi_derivative"][
                        self.numba_parallel
                    ]
                    gt_dot_v_func(
                        receivers,
                        cells_bounds_active,
                        self.cell_z_top,
                        self.cell_z_bottom,
                        regional_field,
                        kernel_xx,
                        kernel_yy,
                        kernel_zz,
                        kernel_xy,
                        kernel_xz,
                        kernel_yz,
                        scalar_model,
                        vector[vector_slice],
                        result,
                    )
                else:
                    kernel_x, kernel_y, kernel_z = CHOCLO_KERNELS[component]
                    gt_dot_v_func = NUMBA_FUNCTIONS_2D["gt_dot_v"][
                        "magnetic_component"
                    ][self.numba_parallel]
                    gt_dot_v_func(
                        receivers,
                        cells_bounds_active,
                        self.cell_z_top,
                        self.cell_z_bottom,
                        regional_field,
                        kernel_x,
                        kernel_y,
                        kernel_z,
                        scalar_model,
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
        # Get regional field
        regional_field = self.survey.source_field.b0
        # Get cells in the 2D mesh and keep only active cells
        cells_bounds_active = self.mesh.cell_bounds[self.active_cells]
        # Define the constant factor
        constant_factor = 1 / 4 / np.pi
        # Allocate array for the diagonal
        scalar_model = self.model_type == "scalar"
        n_columns = self.nC if scalar_model else 3 * self.nC
        diagonal = np.zeros(n_columns, dtype=np.float64)
        # Start filling the diagonal array
        for components, receivers in self._get_components_and_receivers():
            if not CHOCLO_SUPPORTED_COMPONENTS.issuperset(components):
                raise NotImplementedError(
                    f"Other components besides {CHOCLO_SUPPORTED_COMPONENTS} "
                    "aren't implemented yet."
                )
            for component in components:
                if component == "tmi":
                    diagonal_gtg_func = NUMBA_FUNCTIONS_2D["diagonal_gtg"]["tmi"][
                        self.numba_parallel
                    ]
                    diagonal_gtg_func(
                        receivers,
                        cells_bounds_active,
                        self.cell_z_top,
                        self.cell_z_bottom,
                        regional_field,
                        constant_factor,
                        scalar_model,
                        weights,
                        diagonal,
                    )
                elif component in ("tmi_x", "tmi_y", "tmi_z"):
                    kernel_xx, kernel_yy, kernel_zz, kernel_xy, kernel_xz, kernel_yz = (
                        CHOCLO_KERNELS[component]
                    )
                    diagonal_gtg_func = NUMBA_FUNCTIONS_2D["diagonal_gtg"][
                        "tmi_derivative"
                    ][self.numba_parallel]
                    diagonal_gtg_func(
                        receivers,
                        cells_bounds_active,
                        self.cell_z_top,
                        self.cell_z_bottom,
                        regional_field,
                        kernel_xx,
                        kernel_yy,
                        kernel_zz,
                        kernel_xy,
                        kernel_xz,
                        kernel_yz,
                        constant_factor,
                        scalar_model,
                        weights,
                        diagonal,
                    )
                else:
                    kernel_x, kernel_y, kernel_z = CHOCLO_KERNELS[component]
                    diagonal_gtg_func = NUMBA_FUNCTIONS_2D["diagonal_gtg"][
                        "magnetic_component"
                    ][self.numba_parallel]
                    diagonal_gtg_func(
                        receivers,
                        cells_bounds_active,
                        self.cell_z_top,
                        self.cell_z_bottom,
                        regional_field,
                        kernel_x,
                        kernel_y,
                        kernel_z,
                        constant_factor,
                        scalar_model,
                        weights,
                        diagonal,
                    )
        return diagonal


class Simulation3DDifferential(BaseMagneticPDESimulation):
    r"""A secondary field simulation for magnetic data.

    Parameters
    ----------
    mesh : discretize.base.BaseMesh
    survey : magnetics.survey.Survey
    mu : float, array_like
        Magnetic Permeability Model (H/ m). Set this for forward
        modeling or to fix while inverting for remanence. This is used if
        ``muMap`` is None.
    muMap : simpeg.maps.IdentityMap, optional
        The mapping used to go from the simulation model to ``mu``. Set this
        to invert for ``mu``.
    rem : float, array_like
        Magnetic Polarization :math:`\mu_0 \mathbf{M}` (nT). Set this for forward
        modeling or to fix remanent magnetization while inverting for permeability.
        This is used if ``remMap`` is None.
    remMap : simpeg.maps.IdentityMap, optional
        The mapping used to go from the simulation model to :math:`\mu_0 \mathbf{M}`.
        Set this to invert for :math:`\mu_0 \mathbf{M}`.
    storeJ: bool
        Whether to store the sensitivity matrix. If set to True
    solver_dtype: dtype, optional
        Data type to use for the matrix that gets passed to the ``solver``.
        Default to `numpy.float64`.


    Notes
    -----
    This simulation solves for the magnetostatic PDE:

    .. math::

        \nabla \cdot \mathbf{B} = 0

    where the constitutive relation is specified as:

    .. math::

        \mathbf{B} = \mu\mathbf{H} + \mu_0\mathbf{M_r}

    where :math:`\mathbf{M_r}` is a fixed magnetization unaffected by the inducing field
    and :math:`\mu\mathbf{H}` is the induced magnetization.
    """

    _Ainv = None

    rem, remMap, remDeriv = props.Invertible(
        "Magnetic Polarization (nT)", optional=True
    )

    _supported_components = ("tmi", "bx", "by", "bz")

    def __init__(
        self,
        mesh,
        survey=None,
        mu=None,
        muMap=None,
        rem=None,
        remMap=None,
        storeJ=False,
        solver_dtype=np.float64,
        **kwargs,
    ):
        if mu is None:
            mu = mu_0

        super().__init__(mesh=mesh, survey=survey, mu=mu, muMap=muMap, **kwargs)

        self.rem = rem
        self.remMap = remMap

        self.storeJ = storeJ
        self.solver_dtype = solver_dtype

        self._MfMu0i = self.mesh.get_face_inner_product(1.0 / mu_0)
        self._Div = self.Mcc * self.mesh.face_divergence
        self._DivT = self._Div.T.tocsr()
        self._Mf_vec_deriv = self.mesh.get_face_inner_product_deriv(
            np.ones(self.mesh.n_cells * 3)
        )(np.ones(self.mesh.n_faces))

        self.solver_opts = {"is_symmetric": True, "is_positive_definite": True}

        self._Jmatrix = None
        self._stored_fields = None

    @property
    def survey(self):
        """The magnetic survey object.

        Returns
        -------
        simpeg.potential_fields.magnetics.Survey
        """
        if self._survey is None:
            raise AttributeError("Simulation must have a survey")
        return self._survey

    @survey.setter
    def survey(self, value):
        if value is not None:
            value = validate_type("survey", value, Survey, cast=False)
            unsupported_components = {
                component
                for source in value.source_list
                for receiver in source.receiver_list
                for component in receiver.components
                if component not in self._supported_components
            }
            if unsupported_components:
                msg = (
                    f"Found unsupported magnetic components "
                    f"'{', '.join(c for c in unsupported_components)}' in the survey."
                    f"The {type(self).__name__} currently supports the following "
                    f"components: {', '.join(c for c in self._supported_components)}"
                )
                raise NotImplementedError(msg)
        self._survey = value

    @property
    def storeJ(self):
        """Whether to store the sensitivity matrix

        Returns
        -------
        bool
        """
        return self._storeJ

    @storeJ.setter
    def storeJ(self, value):
        self._storeJ = validate_type("storeJ", value, bool)

    @property
    def solver_dtype(self):
        """
        Data type used by the solver.

        Returns
        -------
        numpy.dtype
            Either np.float32 or np.float64
        """
        return self._solver_dtype

    @solver_dtype.setter
    def solver_dtype(self, value):
        """
        Set the solver dtype. Must be np.float32 or np.float64.
        """
        if value not in (np.float32, np.float64):
            msg = (
                f"Invalid `solver_dtype` '{value}'. "
                "It must be np.float32 or np.float64."
            )
            raise ValueError(msg)
        self._solver_dtype = value

    @cached_property
    @utils.requires("survey")
    def _b0(self):
        # Todo: Experiment with avoiding array of constants
        b0 = self.survey.source_field.b0
        b0 = np.r_[
            b0[0] * np.ones(self.mesh.nFx),
            b0[1] * np.ones(self.mesh.nFy),
            b0[2] * np.ones(self.mesh.nFz),
        ]
        return b0

    @property
    def _stored_fields(self):
        return self.__stored_fields

    @_stored_fields.setter
    def _stored_fields(self, value):
        self.__stored_fields = value

    @_stored_fields.deleter
    def _stored_fields(self):
        self.__stored_fields = None

    def _getRHS(self, m):
        self.model = m

        rhs = 0

        if not np.isscalar(self.mu) or not np.allclose(self.mu, mu_0):
            rhs += (
                self._Div * self.MfMuiI * self._MfMu0i * self._b0 - self._Div * self._b0
            )

        if self.rem is not None:
            rhs += (
                self._Div
                * (
                    self.MfMuiI
                    * self.mesh.get_face_inner_product(
                        self.rem
                        / np.tile(self.mu * np.ones(self.mesh.n_cells), self.mesh.dim)
                    )
                ).diagonal()
            )

        return rhs

    def _getA(self):
        A = self._Div * self.MfMuiI * self._DivT

        A = A.astype(self.solver_dtype)

        return A

    def fields(self, m):
        self.model = m

        if self._stored_fields is None:

            if self._Ainv is None:
                self._Ainv = self.solver(self._getA(), **self.solver_opts)

            rhs = self._getRHS(m)

            u = self._Ainv * rhs
            b_field = -self.MfMuiI * self._DivT * u

            if not np.isscalar(self.mu) or not np.allclose(self.mu, mu_0):
                b_field += self._MfMu0i * self.MfMuiI * self._b0 - self._b0

            if self.rem is not None:
                b_field += (
                    self.MfMuiI
                    * self.mesh.get_face_inner_product(
                        self.rem
                        / np.tile(self.mu * np.ones(self.mesh.n_cells), self.mesh.dim)
                    )
                ).diagonal()

            fields = {"b": b_field, "u": u}
            self._stored_fields = fields

        else:
            fields = self._stored_fields

        return fields

    def dpred(self, m=None, f=None):
        self.model = m
        if f is not None:
            return self._projectFields(f)

        if f is None:
            f = self.fields(m)

        dpred = self._projectFields(f)

        return dpred

    def magnetic_polarization(self, m=None):
        r"""
        Computes the total magnetic polarization :math:`\mu_0\mathbf{M}`.

        Parameters
        ----------
        m : (n_param,) numpy.ndarray
            The model parameters.

        Returns
        -------
        mu0_m : np.ndarray
            The magnetic polarization :math:`\mu_0 \mathbf{M}` in nanoteslas (nT), defined on the mesh faces.
            The result is ordered as a concatenation of the x, y, and z face components
            (i.e., ``[Mx_faces, My_faces, Mz_faces]``).


        """
        self.model = m
        f = self.fields(m)
        b_field, u = f["b"], f["u"]
        MfMu0iI = self.mesh.get_face_inner_product(1.0 / mu_0, invert_matrix=True)

        mu0_h = -MfMu0iI * self._DivT * u
        mu0_m = b_field - mu0_h

        return mu0_m

    def Jvec(self, m, v, f=None):
        self.model = m

        if f is None:
            f = self.fields(m)

        if self.storeJ:
            J = self.getJ(m, f=f)
            return J.dot(v)

        return self._Jvec(m, v, f)

    def Jtvec(self, m, v, f=None):
        self.model = m

        if f is None:
            f = self.fields(m)

        if self.storeJ:
            J = self.getJ(m, f=f)
            return np.asarray(J.T.dot(v))

        return self._Jtvec(m, v, f)

    def getJ(self, m, f=None):
        self.model = m
        if self._Jmatrix:
            return self._Jmatrix
        if f is None:
            f = self.fields(m)
        if m.size < self.survey.nD:
            J = self._Jvec(m, v=None, f=f)
        else:
            J = self._Jtvec(m, v=None, f=f).T

        if self.storeJ:
            self._Jmatrix = J
        return J

    def _Jtvec(self, m, v, f):
        b_field, u = f["b"], f["u"]

        Q = self._projectFieldsDeriv(b_field)

        if v is None:
            v = np.eye(Q.shape[0])
            divt_solve_q = (
                self._DivT * (self._Ainv * ((Q * self.MfMuiI * -self._DivT).T * v))
                + Q.T * v
            )
            del v
        else:
            divt_solve_q = (
                self._DivT * (self._Ainv * ((-self._Div * (self.MfMuiI.T * (Q.T * v)))))
                + Q.T * v
            )

        mu_vec = np.tile(self.mu * np.ones(self.mesh.n_cells), self.mesh.dim)

        Jtv = 0

        if self.remMap is not None:
            Mf_rem_deriv = self._Mf_vec_deriv * sp.diags(1 / mu_vec) * self.remDeriv
            Jtv += (self.MfMuiI * Mf_rem_deriv).T * (divt_solve_q)

        if self.muMap is not None:
            Jtv += self.MfMuiIDeriv(self._DivT * u, -divt_solve_q, adjoint=True)
            Jtv += self.MfMuiIDeriv(
                self._b0, self._MfMu0i.T * (divt_solve_q), adjoint=True
            )

            if self.rem is not None:
                Mf_r_mui = self.mesh.get_face_inner_product(
                    self.rem / mu_vec
                ).diagonal()
                mu_vec_i_deriv = sp.vstack(
                    (self.muiDeriv, self.muiDeriv, self.muiDeriv)
                )

                Mf_r_mui_deriv = (
                    self._Mf_vec_deriv * sp.diags(self.rem) * mu_vec_i_deriv
                )

                Jtv += (
                    self.MfMuiIDeriv(Mf_r_mui, divt_solve_q, adjoint=True)
                    + (Mf_r_mui_deriv.T * self.MfMuiI.T) * divt_solve_q
                )

        return Jtv

    def _Jvec(self, m, v, f):

        if v is None:
            v = np.eye(m.shape[0])

        b_field, u = f["b"], f["u"]

        Q = self._projectFieldsDeriv(b_field)
        C = -self.MfMuiI * self._DivT

        db_dm = 0
        dCmu_dm = 0

        mu_vec = np.tile(self.mu * np.ones(self.mesh.n_cells), self.mesh.dim)

        if self.remMap is not None:
            Mf_rem_deriv = self._Mf_vec_deriv * sp.diags(1 / mu_vec) * self.remDeriv
            db_dm += self.MfMuiI * Mf_rem_deriv * v

        if self.muMap is not None:
            dCmu_dm += self.MfMuiIDeriv(self._DivT @ u, v, adjoint=False)
            db_dm += self._MfMu0i * self.MfMuiIDeriv(self._b0, v, adjoint=False)

            if self.rem is not None:
                Mf_r_mui = self.mesh.get_face_inner_product(
                    self.rem / mu_vec
                ).diagonal()
                mu_vec_i_deriv = sp.vstack(
                    (self.muiDeriv, self.muiDeriv, self.muiDeriv)
                )
                Mf_r_mui_deriv = (
                    self._Mf_vec_deriv * sp.diags(self.rem) * mu_vec_i_deriv
                )
                db_dm += self.MfMuiIDeriv(Mf_r_mui, v, adjoint=False) + (
                    self.MfMuiI * Mf_r_mui_deriv * v
                )

        Ainv_Ddm = self._Ainv * (self._Div * (-dCmu_dm + db_dm))

        Jv = Q * (C * Ainv_Ddm + (-dCmu_dm + db_dm))

        return Jv

    @cached_property
    def _Qfx(self):
        Qfx = self.mesh.get_interpolation_matrix(self.survey.receiver_locations, "Fx")
        return Qfx

    @cached_property
    def _Qfy(self):
        Qfy = self.mesh.get_interpolation_matrix(self.survey.receiver_locations, "Fy")
        return Qfy

    @cached_property
    def _Qfz(self):
        Qfz = self.mesh.get_interpolation_matrix(self.survey.receiver_locations, "Fz")
        return Qfz

    def _projectFields(self, f):

        rx_list = self.survey.source_field.receiver_list
        components = []
        for rx in rx_list:
            components.extend(rx.components)
        components = set(components)

        if "bx" in components or "tmi" in components:
            bx = self._Qfx * f["b"]
        if "by" in components or "tmi" in components:
            by = self._Qfy * f["b"]
        if "bz" in components or "tmi" in components:
            bz = self._Qfz * f["b"]

        if "tmi" in components:
            b0 = self.survey.source_field.b0
            tmi = np.sqrt(
                (bx + b0[0]) ** 2 + (by + b0[1]) ** 2 + (bz + b0[2]) ** 2
            ) - np.sqrt(b0[0] ** 2 + b0[1] ** 2 + b0[2] ** 2)

        n_total = 0
        total_data_list = []
        for rx in rx_list:
            data = {}
            rx_n_locs = rx.locations.shape[0]
            if "bx" in rx.components:
                data["bx"] = bx[n_total : n_total + rx_n_locs]
            if "by" in rx.components:
                data["by"] = by[n_total : n_total + rx_n_locs]
            if "bz" in rx.components:
                data["bz"] = bz[n_total : n_total + rx_n_locs]
            if "tmi" in rx.components:
                data["tmi"] = tmi[n_total : n_total + rx_n_locs]

            n_total += rx_n_locs

            total_data_list.append(
                np.concatenate([data[comp] for comp in rx.components])
            )

        if len(total_data_list) == 1:
            return total_data_list[0]

        return np.concatenate(total_data_list, axis=0)

    @utils.count
    def _projectFieldsDeriv(self, bs):
        rx_list = self.survey.source_field.receiver_list
        components = []
        for rx in rx_list:
            components.extend(rx.components)
        components = set(components)

        if "tmi" in components:
            b0 = self.survey.source_field.b0
            bot = np.sqrt(b0[0] ** 2 + b0[1] ** 2 + b0[2] ** 2)

            bx = self._Qfx * bs
            by = self._Qfy * bs
            bz = self._Qfz * bs

            dpred = (
                np.sqrt((bx + b0[0]) ** 2 + (by + b0[1]) ** 2 + (bz + b0[2]) ** 2) - bot
            )

            dDhalf_dD = sdiag(1 / (dpred + bot))

            xterm = sdiag(b0[0] + bx) * self._Qfx
            yterm = sdiag(b0[1] + by) * self._Qfy
            zterm = sdiag(b0[2] + bz) * self._Qfz

            Qtmi = dDhalf_dD * (xterm + yterm + zterm)

        n_total = 0
        total_data_list = []
        for rx in rx_list:
            data = {}
            rx_n_locs = rx.locations.shape[0]
            if "bx" in rx.components:
                data["bx"] = self._Qfx[n_total : n_total + rx_n_locs][:]
            if "by" in rx.components:
                data["by"] = self._Qfy[n_total : n_total + rx_n_locs][:]
            if "bz" in rx.components:
                data["bz"] = self._Qfz[n_total : n_total + rx_n_locs][:]
            if "tmi" in rx.components:
                data["tmi"] = Qtmi[n_total : n_total + rx_n_locs][:]

            n_total += rx_n_locs

            total_data_list.append(sp.vstack([data[comp] for comp in rx.components]))

        if len(total_data_list) == 1:
            return total_data_list[0]

        return sp.vstack(total_data_list)

    @property
    def _delete_on_model_update(self):
        toDelete = super()._delete_on_model_update
        if self._stored_fields is not None:
            toDelete = toDelete + ["_stored_fields"]
        if self.muMap is not None:
            if self._Ainv is not None:
                toDelete = toDelete + ["_Ainv"]
            if self._Jmatrix is not None:
                toDelete = toDelete + ["_Jmatrix"]
        return toDelete
