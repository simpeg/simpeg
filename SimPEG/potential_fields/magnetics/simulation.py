import numpy as np
import scipy.sparse as sp
from scipy.constants import mu_0

from SimPEG import utils
from ..base import BasePFSimulation, BaseEquivalentSourceLayerSimulation
from ...base import BaseMagneticPDESimulation
from .survey import Survey

from SimPEG import props

from SimPEG.utils import mkvc, mat_utils, sdiag
from SimPEG.utils.code_utils import validate_string, deprecate_property, validate_type
from geoana.kernels import (
    prism_fzz,
    prism_fzx,
    prism_fzy,
    prism_fzzz,
    prism_fxxy,
    prism_fxxz,
    prism_fxyz,
)


class Simulation3DIntegral(BasePFSimulation):
    """
    magnetic simulation in integral form.

    """

    chi, chiMap, chiDeriv = props.Invertible("Magnetic Susceptibility (SI)")

    def __init__(
        self,
        mesh,
        chi=None,
        chiMap=None,
        model_type="scalar",
        is_amplitude_data=False,
        **kwargs
    ):
        self.model_type = model_type
        super().__init__(mesh, **kwargs)
        self.chi = chi
        self.chiMap = chiMap

        self._G = None
        self._M = None
        self._gtg_diagonal = None
        self.is_amplitude_data = is_amplitude_data
        self.modelMap = self.chiMap

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
            self._G = self.linear_operator()

        return self._G

    modelType = deprecate_property(
        model_type, "modelType", "model_type", removal_version="0.18.0"
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
            'bx', 'by', 'bz', 'bxx', 'bxy', 'bxz', 'byy', 'byz', 'bzz'

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

        if "bxx" in components:
            node_evals["gxxx"] = prism_fzzz(dy, dz, dx)
            node_evals["gxxy"] = prism_fxxy(dx, dy, dz)
            node_evals["gxxz"] = prism_fxxz(dx, dy, dz)
        if "bxy" in components:
            if "gxxy" not in node_evals:
                node_evals["gxxy"] = prism_fxxy(dx, dy, dz)
            node_evals["gyyx"] = prism_fxxz(dy, dz, dx)
            node_evals["gxyz"] = prism_fxyz(dx, dy, dz)
        if "bxz" in components:
            if "gxxz" not in node_evals:
                node_evals["gxxz"] = prism_fxxz(dx, dy, dz)
            if "gxyz" not in node_evals:
                node_evals["gxyz"] = prism_fxyz(dx, dy, dz)
            node_evals["gzzx"] = prism_fxxy(dz, dx, dy)
        if "byy" in components:
            if "gyyx" not in node_evals:
                node_evals["gyyx"] = prism_fxxz(dy, dz, dx)
            node_evals["gyyy"] = prism_fzzz(dz, dx, dy)
            node_evals["gyyz"] = prism_fxxy(dy, dz, dx)
        if "byz" in components:
            if "gxyz" not in node_evals:
                node_evals["gxyz"] = prism_fxyz(dx, dy, dz)
            if "gyyz" not in node_evals:
                node_evals["gyyz"] = prism_fxxy(dy, dz, dx)
            node_evals["gzzy"] = prism_fxxz(dz, dx, dy)
        if "bzz" in components:
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
    r"""A secondary field simulation for magnetic data.

    Parameters
    ----------
    mesh : discretize.base.BaseMesh
    survey : magnetics.suvey.Survey
    mu : float, array_like
        Magnetic Permeability Model (C/(s m^3)). Set this for forward
        modeling or to fix while inverting for remanence. This is used if
        muMap == None
    rem : float, array_like
        Magnetic Polarization \mu_0*M (nT). Set this for forward
        modeling or to fix remanent magnetization while inverting for permeability.
        This is used if remMap == None
    muMap : SimPEG.maps.IdentityMap, optional
        The mapping used to go from the simulation model to `mu`. Set this
        to invert for `mu`.
    remMap : SimPEG.maps.IdentityMap, optional
        The mapping used to go from the simulation model to `mu_0*M`. Set this
        to invert for `mu_0*M`.
    storeJ: bool
        Whether to store the sensitivity matrix. If set to True
    exact_TMI: bool
        Preforms an exact TMI projection if set to True and "tmi" is in
        survey.components


    Notes
    -----
    This simulation solves for the magnetostatic PDE:
    \nabla \cdot \Vec{B} = 0

    where the constitutive relation is defined as:
    \Vec{B} = \mu\Vec{H} + \mu_0\Vec{M_r}

    where \Vec{M_r} is a fixed magnetization unaffected by the inducing field
    and \mu\Vec{H} is the induced magnetization
    """

    _Jmatrix = None
    _Ainv = None

    rem, remMap, remDeriv = props.Invertible(
        "Magnetic Polarization (nT)", optional=True
    )

    def __init__(
        self,
        mesh,
        survey=None,
        mu=None,
        rem=None,
        remMap=None,
        muMap=None,
        storeJ=False,
        exact_TMI=True,
        **kwargs
    ):
        if mu is None:
            mu = mu_0

        super().__init__(mesh=mesh, survey=survey, mu=mu, muMap=muMap, **kwargs)

        if (
            muMap is None
            and np.isscalar(mu)
            and np.allclose(mu, mu_0)
            and storeJ is True
        ):
            self._update_J = False
        else:
            self._update_J = True

        self.rem = rem
        self.remMap = remMap

        self.storeJ = storeJ
        self.exact_TMI = exact_TMI

        self._MfMu0i = self.mesh.get_face_inner_product(1.0 / mu_0)
        self._Div = self.Mcc * self.mesh.face_divergence
        self._DivT = self._Div.T.tocsr()
        self._Mf_vec_deriv = self.mesh.get_face_inner_product_deriv(
            np.ones(self.mesh.n_cells * 3)
        )(np.ones(self.mesh.n_faces))

        self.solver_opts = {"is_symmetric": True, "is_positive_definite": True}

    @property
    def survey(self):
        """The magnetic survey object.

        Returns
        -------
        SimPEG.potential_fields.magnetics.Survey
        """
        if self._survey is None:
            raise AttributeError("Simulation must have a survey")
        return self._survey

    @survey.setter
    def survey(self, value):
        if value is not None:
            value = validate_type("survey", value, Survey, cast=False)
        self._survey = value

    @property
    def storeJ(self):
        return self._storeJ

    @storeJ.setter
    def storeJ(self, value):
        self._storeJ = validate_type("storeJ", value, bool)

    @property
    def exact_TMI(self):
        return self._exact_TMI

    @exact_TMI.setter
    def exact_TMI(self, value):
        self._exact_TMI = validate_type("exact_TMI", value, bool)

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
        self.model = m

        rhs = 0

        if not np.isscalar(self.mu) or not np.allclose(self.mu, mu_0):
            B0 = self.getB0()
            rhs += self._Div * self.MfMuiI * self._MfMu0i * B0 - self._Div * B0

        if self.rem is not None:
            mu = self.mu * np.ones(self.mesh.n_cells)
            mu_vec = np.hstack((mu, mu, mu))
            rhs += (
                self._Div
                * (
                    self.MfMuiI * self.mesh.get_face_inner_product(self.rem / mu_vec)
                ).diagonal()
            )

        return rhs

    def getA(self, m):
        return self._Div * self.MfMuiI * self._DivT

    def fields(self, m):
        self.model = m

        self._Ainv = self.solver(self.getA(m), **self.solver_opts)

        rhs = self.getRHS(m)

        u = self._Ainv * rhs
        B = -self.MfMuiI * self._DivT * u

        if not np.isscalar(self.mu) or not np.allclose(self.mu, mu_0):
            B0 = self.getB0()
            B += self._MfMu0i * self.MfMuiI * B0 - B0

        if self.rem is not None:
            mu = self.mu * np.ones(self.mesh.n_cells)
            mu_vec = np.hstack((mu, mu, mu))
            B += (
                self.MfMuiI * self.mesh.get_face_inner_product(self.rem / mu_vec)
            ).diagonal()

        return {"B": B, "u": u}

    def dpred(self, m=None, f=None):
        if f is not None:
            return self.projectFields(f)

        if self._Jmatrix is not None and self._update_J is False:
            if self.exact_TMI:
                dpred_fields = self._Jmatrix @ m
                return self.projectFields({"B": dpred_fields, "u": None})
            else:
                return self._Jmatrix @ m

        if f is None:
            f = self.fields(m)

        dpred = self.projectFields(f)

        return dpred

    def Jvec(self, m, v, f=None):
        self.model = m

        if self.storeJ:
            J = self.getJ(m, f=f)
            return J.dot(v)

        self.model = m

        if f is None:
            f = self.fields(m)

        B, u = f["B"], f["u"]

        Q = self.projectFieldsDeriv(B)
        B0 = self.getB0()
        C = -self.MfMuiI * self._DivT

        db_dm = 0
        dCmu_dm = 0

        mu = np.ones(self.mesh.n_cells) * self.mu
        mu_vec = np.hstack((mu, mu, mu))

        if self.remMap is not None:
            Mf_rem_deriv = self._Mf_vec_deriv * sp.diags(1 / mu_vec) * self.remDeriv
            db_dm += self.MfMuiI * Mf_rem_deriv * v

        if self.muMap is not None:
            dCmu_dm += self.MfMuiIDeriv(self._DivT @ u, v, adjoint=False)
            db_dm += self._MfMu0i * self.MfMuiIDeriv(B0, v, adjoint=False)

            if self.rem is not None:
                Mf_r_over_uvec = self.mesh.get_face_inner_product(
                    self.rem / mu_vec
                ).diagonal()
                mu_vec_i_deriv = sp.vstack(
                    (self.muiDeriv, self.muiDeriv, self.muiDeriv)
                )
                Mf_mu_vec_i_deriv = (
                    self._Mf_vec_deriv * sp.diags(self.rem) * mu_vec_i_deriv
                )
                db_dm += self.MfMuiIDeriv(Mf_r_over_uvec, v, adjoint=False) + (
                    self.MfMuiI * Mf_mu_vec_i_deriv * v
                )

        dq_dm_min_dAmu_dm = self._Div * (-dCmu_dm + db_dm)

        Ainv_dqmindAmu = self._Ainv * dq_dm_min_dAmu_dm

        Jv = Q * (C * Ainv_dqmindAmu + (-dCmu_dm + db_dm))

        return Jv

    def Jtvec(self, m, v, f=None):
        self.model = m

        if self.storeJ:
            J = self.getJ(m, f=f)
            return np.asarray(J.T.dot(v))

        self.model = m

        if f is None:
            f = self.fields(m)

        return self._Jtvec(m, v, f)

    def getJ(self, m, f=None):
        if self._Jmatrix is None:
            if f is None:
                f = self.fields(m)

            J = self._Jtvec(m, v=None, f=f).T

        else:
            J = self._Jmatrix

        if self.storeJ is True:
            self._Jmatrix = J

        if self._update_J is False and self.exact_TMI:
            projection_deriv = self.projectFieldsDeriv(J @ m)
            J = projection_deriv * J

        return J

    def _Jtvec(self, m, v, f=None):
        B, u = f["B"], f["u"]

        Q = self.projectFieldsDeriv(B)

        B0 = self.getB0()
        if v is None:
            v = np.eye(Q.shape[0])
            DivTatsol_p_QT = (
                self._DivT * (self._Ainv * ((Q * self.MfMuiI * -self._DivT).T * v))
                + Q.T * v
            )
        else:
            DivTatsol_p_QT = (
                self._DivT * (self._Ainv * ((-self._Div * (self.MfMuiI.T * (Q.T * v)))))
                + Q.T * v
            )

        del Q

        mu = np.ones(self.mesh.n_cells) * self.mu
        mu_vec = np.hstack((mu, mu, mu))

        Jtv = 0

        if self.remMap is not None:
            Mf_rem_deriv = self._Mf_vec_deriv * sp.diags(1 / mu_vec) * self.remDeriv
            Jtv += (self.MfMuiI * Mf_rem_deriv).T * (DivTatsol_p_QT)

        if self.muMap is not None:
            Jtv += self.MfMuiIDeriv(self._DivT * u, -DivTatsol_p_QT, adjoint=True)
            Jtv += self.MfMuiIDeriv(B0, self._MfMu0i.T * (DivTatsol_p_QT), adjoint=True)

            if self.rem is not None:
                Mf_r_over_uvec = self.mesh.get_face_inner_product(
                    self.rem / mu_vec
                ).diagonal()
                mu_vec_i_deriv = sp.vstack(
                    (self.muiDeriv, self.muiDeriv, self.muiDeriv)
                )

                Mf_mu_vec_i_deriv = (
                    self._Mf_vec_deriv * sp.diags(self.rem) * mu_vec_i_deriv
                )

                Jtv += (
                    self.MfMuiIDeriv(Mf_r_over_uvec, DivTatsol_p_QT, adjoint=True)
                    + (Mf_mu_vec_i_deriv.T * self.MfMuiI.T) * DivTatsol_p_QT
                )

        return Jtv

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
        components = self.survey.components

        fields = {}

        if u["u"] is None:
            fields["bx"], fields["by"], fields["bz"] = np.split(u["B"], 3)

        else:
            if "bx" in components or "tmi" in components:
                fields["bx"] = self.Qfx * u["B"]
            if "by" in components or "tmi" in components:
                fields["by"] = self.Qfy * u["B"]
            if "bz" in components or "tmi" in components:
                fields["bz"] = self.Qfz * u["B"]

        B0 = self.survey.source_field.b0

        if "tmi" in components:
            if self.exact_TMI:
                Bot = np.sqrt(B0[0] ** 2 + B0[1] ** 2 + B0[2] ** 2)
                box = B0[0]
                boy = B0[1]
                boz = B0[2]
                fields["tmi"] = (
                    np.sqrt(
                        (fields["bx"] + box) ** 2
                        + (fields["by"] + boy) ** 2
                        + (fields["bz"] + boz) ** 2
                    )
                    - Bot
                )
            else:
                Bot = np.sqrt(B0[0] ** 2 + B0[1] ** 2 + B0[2] ** 2)
                box = B0[0] / Bot
                boy = B0[1] / Bot
                boz = B0[2] / Bot
                fields["tmi"] = (
                    fields["bx"] * box + fields["by"] * boy + fields["bz"] * boz
                )

        return np.concatenate([fields[comp] for comp in components])

    @utils.count
    def projectFieldsDeriv(self, Bs):
        components = self.survey.components

        fields = {}

        B0 = self.survey.source_field.b0
        Bot = np.sqrt(B0[0] ** 2 + B0[1] ** 2 + B0[2] ** 2)

        box = B0[0]
        boy = B0[1]
        boz = B0[2]

        if self._update_J is False and self._Jmatrix is not None and self.exact_TMI:
            bx, by, bz = np.split(Bs, 3)

            total_field = np.sqrt((bx + box) ** 2 + (by + boy) ** 2 + (bz + boz) ** 2)
            dTMIe_dx = sp.diags((box + bx) / total_field)
            dTMIe_dy = sp.diags((boy + by) / total_field)
            dTMIe_dz = sp.diags((boz + bz) / total_field)

            fields["tmi"] = sp.hstack([dTMIe_dx, dTMIe_dy, dTMIe_dz])

            diag_b = sp.diags(np.ones_like(bx))

            if "bx" in components:
                fields["bx"] = sp.hstack([diag_b, 0 * diag_b, 0 * diag_b])
            if "by" in components:
                fields["by"] = sp.hstack([0 * diag_b, diag_b, 0 * diag_b])
            if "bz" in components:
                fields["bz"] = sp.hstack([0 * diag_b, 0 * diag_b, diag_b])

            return sp.vstack([fields[comp] for comp in components])

        if "bx" in components or "tmi" in components:
            fields["bx"] = self.Qfx
        if "by" in components or "tmi" in components:
            fields["by"] = self.Qfy
        if "bz" in components or "tmi" in components:
            fields["bz"] = self.Qfz

        if "tmi" in components:
            if self.exact_TMI:
                if not self._update_J and self._Jmatrix is None:
                    return sp.vstack((self.Qfx, self.Qfy, self.Qfz))

                box = B0[0]
                boy = B0[1]
                boz = B0[2]

                bx = self.Qfx * Bs
                by = self.Qfy * Bs
                bz = self.Qfz * Bs

                dpred = (
                    np.sqrt((bx + box) ** 2 + (by + boy) ** 2 + (bz + boz) ** 2) - Bot
                )

                dDhalf_dD = sdiag(1 / (dpred + Bot))

                xterm = sdiag(box + bx) * self.Qfx
                yterm = sdiag(boy + by) * self.Qfy
                zterm = sdiag(boz + bz) * self.Qfz

                fields["tmi"] = dDhalf_dD * (xterm + yterm + zterm)

            else:
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

    @property
    def deleteTheseOnModelUpdate(self):
        toDelete = super().deleteTheseOnModelUpdate
        if self._update_J:
            if self._Jmatrix is not None:
                toDelete = toDelete + ["_Jmatrix"]
        return toDelete

    @property
    def clean_on_model_update(self):
        toclean = super().clean_on_model_update
        if self.muMap is None:
            return toclean
        else:
            return toclean + ["_Ainv"]
