import numpy as np
from scipy.optimize import minimize
import warnings


from ....utils import (
    mkvc,
    sdiag,
    Zero,
    validate_type,
    validate_string,
    validate_integer,
    validate_active_indices,
)
from ....base import BaseElectricalPDESimulation
from ....data import Data

from .survey import Survey
from .fields_2d import Fields2D, Fields2DCellCentered, Fields2DNodal
from .fields import FieldsDC, Fields3DCellCentered, Fields3DNodal
from .utils import _mini_pole_pole
from scipy.special import k0e, k1e, k0
from discretize.utils import make_boundary_bool


class BaseDCSimulation2D(BaseElectricalPDESimulation):
    """
    Base 2.5D DC problem
    """

    fieldsPair = Fields2D  # SimPEG.EM.Static.Fields_2D
    fieldsPair_fwd = FieldsDC
    # there's actually nT+1 fields, so we don't need to store the last one
    _mini_survey = None

    def __init__(
        self,
        mesh,
        survey=None,
        nky=11,
        storeJ=False,
        miniaturize=False,
        do_trap=False,
        fix_Jmatrix=False,
        surface_faces=None,
        **kwargs,
    ):
        super().__init__(mesh=mesh, survey=survey, **kwargs)
        self.nky = nky
        self.storeJ = storeJ
        self.fix_Jmatrix = fix_Jmatrix
        self.surface_faces = surface_faces

        do_trap = validate_type("do_trap", do_trap, bool)
        if not do_trap:
            # try to find an optimal set of quadrature points and weights
            def get_phi(r):
                e = np.ones_like(r)

                def phi(k):
                    # use log10 transform to enforce positivity
                    k = 10**k
                    A = r[:, None] * k0(r[:, None] * k)
                    v_i = A @ np.linalg.solve(A.T @ A, A.T @ e)
                    dv = (e - v_i) / len(r)
                    return np.linalg.norm(dv)

                def g(k):
                    A = r[:, None] * k0(r[:, None] * k)
                    return np.linalg.solve(A.T @ A, A.T @ e)

                return phi, g

            # find the minimum cell spacing, and the maximum side of the mesh
            min_r = min(self.mesh.edge_lengths)
            max_r = max(
                np.max(self.mesh.nodes, axis=0) - np.min(self.mesh.nodes, axis=0)
            )
            # generate test points log spaced between these two end members
            rs = np.logspace(np.log10(min_r / 4), np.log10(max_r * 4), 100)

            min_rinv = -np.log10(rs).max()
            max_rinv = -np.log10(rs).min()
            # a decent initial guess of the k_i's for the optimization = 1/rs
            k_i = np.linspace(min_rinv, max_rinv, self.nky)

            # these functions depend on r, so grab them
            func, g_func = get_phi(rs)

            # just use scipy's minimize for ease
            out = minimize(func, k_i)
            if self.verbose:
                print(f"optimized ks converged? : {out['success']}")
                print(f"Estimated transform Error: {out['fun']}")
            # transform the solution back to normal points
            points = 10 ** out["x"]
            # transform has a 2/pi and we want 1/pi, so divide by 2
            weights = g_func(points) / 2
            if not out["success"]:
                warnings.warn(
                    "Falling back to trapezoidal for integration. "
                    "You may need to change nky.",
                    stacklevel=2,
                )
                do_trap = True
        if do_trap:
            if self.verbose:
                print("doing trap")
            y = 0.0

            points = np.logspace(-4, 1, self.nky)
            dky = np.diff(points) / 2
            weights = np.r_[dky, 0] + np.r_[0, dky]
            weights *= np.cos(points * y)  # *(1.0/np.pi)
            # assume constant value at 0 frequency?
            weights[0] += points[0] / 2 * (1.0 + np.cos(points[0] * y))
            weights /= np.pi

        self._quad_weights = weights
        self._quad_points = points

        self.Ainv = [None for i in range(self.nky)]
        self.nT = self.nky - 1  # Only for using TimeFields

        # Do stuff to simplify the forward and JTvec operation if number of dipole
        # sources is greater than the number of unique pole sources
        miniaturize = validate_type("miniaturize", miniaturize, bool)
        if miniaturize:
            self._dipoles, self._invs, self._mini_survey = _mini_pole_pole(self.survey)

    @property
    def survey(self):
        """The DC survey object.

        Returns
        -------
        SimPEG.electromagnetics.static.resistivity.survey.Survey
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
    def nky(self):
        """Number of kys to use in wavenumber space.

        Returns
        -------
        int
        """
        return self._nky

    @nky.setter
    def nky(self, value):
        self._nky = validate_integer("nky", value, min_val=3)

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
    def fix_Jmatrix(self):
        """Whether to fix the sensitivity matrix between iterations.

        Returns
        -------
        bool
        """
        return self._fix_Jmatrix

    @fix_Jmatrix.setter
    def fix_Jmatrix(self, value):
        self._fix_Jmatrix = validate_type("fix_Jmatrix", value, bool)

    @property
    def surface_faces(self):
        """Array defining which faces to interpret as surfaces of Neumann boundary

        DC problems will always enforce a Neumann boundary on surface interfaces.
        The default (available on semi-structured grids) assumes the top interface
        is the surface.

        Returns
        -------
        None or numpy.ndarray of bool
        """
        return self._surface_faces

    @surface_faces.setter
    def surface_faces(self, value):
        if value is not None:
            n_bf = self.mesh.boundary_faces.shape[0]
            value = validate_active_indices("surface_faces", value, n_bf)
        self._surface_faces = value

    def fields(self, m=None):
        if self.verbose:
            print(">> Compute fields")
        if m is not None:
            self.model = m
        if self.Ainv[0] is not None:
            for i in range(self.nky):
                self.Ainv[i].clean()
        f = self.fieldsPair(self)
        kys = self._quad_points
        f._quad_weights = self._quad_weights
        for iky, ky in enumerate(kys):
            A = self.getA(ky)
            if self.Ainv[iky] is not None:
                self.Ainv[iky].clean()
            self.Ainv[iky] = self.solver(A, **self.solver_opts)
            RHS = self.getRHS(ky)
            u = self.Ainv[iky] * RHS
            f[:, self._solutionType, iky] = u
        return f

    def fields_to_space(self, f, y=0.0):
        f_fwd = self.fieldsPair_fwd(self)
        phi = f[:, self._solutionType, :].dot(self._quad_weights)
        f_fwd[:, self._solutionType] = phi
        return f_fwd

    def dpred(self, m=None, f=None):
        """
        Project fields to receiver locations
        :param Fields u: fields object
        :rtype: numpy.ndarray
        :return: data
        """
        if f is None:
            if m is None:
                m = self.model
            f = self.fields(m)

        weights = self._quad_weights
        if self._mini_survey is not None:
            survey = self._mini_survey
        else:
            survey = self.survey

        temp = np.empty(survey.nD)
        count = 0
        for src in survey.source_list:
            for rx in src.receiver_list:
                d = rx.eval(src, self.mesh, f).dot(weights)
                temp[count : count + len(d)] = d
                count += len(d)

        return self._mini_survey_data(temp)

    def getJ(self, m, f=None):
        """
        Generate Full sensitivity matrix
        """
        if getattr(self, "_Jmatrix", None) is None:
            if self.verbose:
                print("Calculating J and storing")
            self.model = m
            if f is None:
                f = self.fields(m)
            self._Jmatrix = (self._Jtvec(m, v=None, f=f)).T
        return self._Jmatrix

    def Jvec(self, m, v, f=None):
        """
        Compute sensitivity matrix (J) and vector (v) product.
        """
        if self.storeJ:
            J = self.getJ(m, f=f)
            Jv = mkvc(np.dot(J, v))
            return Jv

        self.model = m

        if f is None:
            f = self.fields(m)

        if self._mini_survey is not None:
            survey = self._mini_survey
        else:
            survey = self.survey

        kys = self._quad_points
        weights = self._quad_weights

        Jv = np.zeros(survey.nD)
        # Assume y=0.
        # This needs some thoughts to implement in general when src is dipole

        # TODO: this loop is pretty slow .. (Parellize)
        for iky, ky in enumerate(kys):
            u_ky = f[:, self._solutionType, iky]
            count = 0
            for i_src, src in enumerate(survey.source_list):
                u_src = u_ky[:, i_src]
                dA_dm_v = self.getADeriv(ky, u_src, v, adjoint=False)
                # dRHS_dm_v = self.getRHSDeriv(ky, src, v) = 0
                du_dm_v = self.Ainv[iky] * (-dA_dm_v)  # + dRHS_dm_v)
                for rx in src.receiver_list:
                    df_dmFun = getattr(f, "_{0!s}Deriv".format(rx.projField), None)
                    df_dm_v = df_dmFun(iky, src, du_dm_v, v, adjoint=False)
                    Jv1_temp = rx.evalDeriv(src, self.mesh, f, df_dm_v)
                    # Trapezoidal intergration
                    Jv[count : count + len(Jv1_temp)] += weights[iky] * Jv1_temp
                    count += len(Jv1_temp)

        return self._mini_survey_data(Jv)

    def Jtvec(self, m, v, f=None):
        """
        Compute adjoint sensitivity matrix (J^T) and vector (v) product.
        """
        if self.storeJ:
            J = self.getJ(m, f=f)
            Jtv = mkvc(np.dot(J.T, v))
            return Jtv

        self.model = m

        if f is None:
            f = self.fields(m)

        return self._Jtvec(m, v=v, f=f)

    def _Jtvec(self, m, v=None, f=None):
        """
        Compute adjoint sensitivity matrix (J^T) and vector (v) product.
        Full J matrix can be computed by inputing v=None
        """
        kys = self._quad_points
        weights = self._quad_weights
        if self._mini_survey is not None:
            survey = self._mini_survey
        else:
            survey = self.survey

        if v is not None:
            # Ensure v is a data object.
            if isinstance(v, Data):
                v = v.dobs
            v = self._mini_survey_dataT(v)
            Jtv = np.zeros(m.size, dtype=float)

            for iky, ky in enumerate(kys):
                u_ky = f[:, self._solutionType, iky]
                count = 0
                for i_src, src in enumerate(survey.source_list):
                    u_src = u_ky[:, i_src]
                    df_duT_sum = 0
                    df_dmT_sum = 0
                    for rx in src.receiver_list:
                        my_v = v[count : count + rx.nD]
                        count += rx.nD
                        # wrt f, need possibility wrt m
                        PTv = rx.evalDeriv(src, self.mesh, f, my_v, adjoint=True)
                        df_duTFun = getattr(f, "_{0!s}Deriv".format(rx.projField), None)
                        df_duT, df_dmT = df_duTFun(iky, src, None, PTv, adjoint=True)
                        df_duT_sum += df_duT
                        df_dmT_sum += df_dmT

                    ATinvdf_duT = self.Ainv[iky] * df_duT_sum

                    dA_dmT = self.getADeriv(ky, u_src, ATinvdf_duT, adjoint=True)
                    # dRHS_dmT = self.getRHSDeriv(ky, src, ATinvdf_duT,
                    #                            adjoint=True)
                    du_dmT = -dA_dmT  # + dRHS_dmT=0
                    Jtv += weights[iky] * (df_dmT + du_dmT).astype(float)
            return mkvc(Jtv)

        else:
            # This is for forming full sensitivity matrix
            Jt = np.zeros((self.model.size, survey.nD), order="F")
            for iky, ky in enumerate(kys):
                u_ky = f[:, self._solutionType, iky]
                istrt = 0
                for i_src, src in enumerate(survey.source_list):
                    u_src = u_ky[:, i_src]
                    for rx in src.receiver_list:
                        # wrt f, need possibility wrt m
                        PT = rx.evalDeriv(src, self.mesh, f).toarray().T
                        ATinvdf_duT = self.Ainv[iky] * PT

                        dA_dmT = self.getADeriv(ky, u_src, ATinvdf_duT, adjoint=True)
                        Jtv = -weights[iky] * dA_dmT  # RHS=0
                        iend = istrt + rx.nD
                        if rx.nD == 1:
                            Jt[:, istrt] += Jtv
                        else:
                            Jt[:, istrt:iend] += Jtv
                        istrt += rx.nD
            return (self._mini_survey_data(Jt.T)).T

    def getSourceTerm(self, ky):
        """
        takes concept of source and turns it into a matrix
        """
        """
        Evaluates the sources, and puts them in matrix form
        :rtype: (numpy.ndarray, numpy.ndarray)
        :return: q (nC or nN, nSrc)
        """

        if self._mini_survey is not None:
            Srcs = self._mini_survey.source_list
        else:
            Srcs = self.survey.source_list

        if self._formulation == "EB":
            n = self.mesh.nN
            # return NotImplementedError

        elif self._formulation == "HJ":
            n = self.mesh.nC

        q = np.zeros((n, len(Srcs)), order="F")

        for i, src in enumerate(Srcs):
            q[:, i] = src.eval(self)
        return q

    @property
    def deleteTheseOnModelUpdate(self):
        toDelete = super().deleteTheseOnModelUpdate
        if self.fix_Jmatrix:
            return toDelete
        return toDelete + ["_Jmatrix"]

    def _mini_survey_data(self, d_mini):
        if self._mini_survey is not None:
            out = d_mini[self._invs[0]]  # AM
            out[self._dipoles[0]] -= d_mini[self._invs[1]]  # AN
            out[self._dipoles[1]] -= d_mini[self._invs[2]]  # BM
            out[self._dipoles[0] & self._dipoles[1]] += d_mini[self._invs[3]]  # BN
        else:
            out = d_mini
        return out

    def _mini_survey_dataT(self, v):
        if self._mini_survey is not None:
            out = np.zeros(self._mini_survey.nD)
            # Need to use ufunc.at because there could be repeated indices
            # That need to be properly handled.
            np.add.at(out, self._invs[0], v)  # AM
            np.subtract.at(out, self._invs[1], v[self._dipoles[0]])  # AN
            np.subtract.at(out, self._invs[2], v[self._dipoles[1]])  # BM
            np.add.at(out, self._invs[3], v[self._dipoles[0] & self._dipoles[1]])  # BN
            return out
        else:
            out = v
        return out


class Simulation2DCellCentered(BaseDCSimulation2D):
    """
    2.5D cell centered DC problem
    """

    _solutionType = "phiSolution"
    _formulation = "HJ"  # CC potentials means J is on faces
    fieldsPair = Fields2DCellCentered
    fieldsPair_fwd = Fields3DCellCentered

    def __init__(self, mesh, survey=None, bc_type="Robin", **kwargs):
        super().__init__(mesh, survey=survey, **kwargs)
        V = sdiag(self.mesh.cell_volumes)
        self.Div = V @ self.mesh.face_divergence
        self.Grad = self.Div.T
        self.bc_type = bc_type

    @property
    def bc_type(self):
        """Type of boundary condition to use for simulation.

        Returns
        -------
        {"Dirichlet", "Neumann", "Robin"}
        """
        return self._bc_type

    @bc_type.setter
    def bc_type(self, value):
        self._bc_type = validate_string(
            "bc_type", value, ["Dirichlet", "Neumann", ("Robin", "Mixed")]
        )

    def getA(self, ky):
        """
        Make the A matrix for the cell centered DC resistivity problem
        A = D MfRhoI G
        """
        # To handle Mixed boundary condition
        self.setBC(ky=ky)
        D = self.Div
        G = self.Grad
        if self.bc_type != "Dirichlet":
            G = G - self._MBC[ky]
        MfRhoI = self.MfRhoI
        # Get resistivity rho
        A = D * MfRhoI * G + ky**2 * self.MccSigma
        if self.bc_type == "Neumann":
            A[0, 0] = A[0, 0] + 1.0
        return A

    def getADeriv(self, ky, u, v, adjoint=False):
        D = self.Div
        G = self.Grad
        if self.bc_type != "Dirichlet":
            G = G - self._MBC[ky]
        if adjoint:
            return self.MfRhoIDeriv(
                G * u, D.T * v, adjoint=adjoint
            ) + ky**2 * self.MccSigmaDeriv(u, v, adjoint=adjoint)
        else:
            return D * self.MfRhoIDeriv(
                G * u, v, adjoint=adjoint
            ) + ky**2 * self.MccSigmaDeriv(u, v, adjoint=adjoint)

    def getRHS(self, ky):
        """
        RHS for the DC problem
        q
        """

        RHS = self.getSourceTerm(ky)
        return RHS

    def getRHSDeriv(self, ky, src, v, adjoint=False):
        """
        Derivative of the right hand side with respect to the model
        """
        # TODO: add qDeriv for RHS depending on m
        # qDeriv = src.evalDeriv(self, ky, adjoint=adjoint)
        # return qDeriv
        return Zero()

    def setBC(self, ky=None):
        if self.bc_type == "Dirichlet":
            return
        if getattr(self, "_MBC", None) is None:
            self._MBC = {}
        if ky in self._MBC:
            # I have already created the BC matrix for this wavenumber
            return
        if self.bc_type == "Neumann":
            alpha, beta, gamma = 0, 1, 0
        else:
            mesh = self.mesh
            boundary_faces = mesh.boundary_faces
            boundary_normals = mesh.boundary_face_outward_normals
            n_bf = len(boundary_faces)

            # Top gets 0 Neumann
            alpha = np.zeros(n_bf)
            beta = np.ones(n_bf)
            gamma = 0

            # assume a source point at the middle of the top of the mesh
            middle = np.median(mesh.nodes, axis=0)
            top_v = np.max(mesh.nodes[:, -1])
            source_point = np.r_[middle[:-1], top_v]

            r_vec = boundary_faces - source_point
            r = np.linalg.norm(r_vec, axis=-1)
            r_hat = r_vec / r[:, None]
            r_dot_n = np.einsum("ij,ij->i", r_hat, boundary_normals)

            if self.surface_faces is None:
                # determine faces that are on the sides and bottom of the mesh...
                if mesh._meshType.lower() == "tree":
                    not_top = boundary_faces[:, -1] != top_v
                elif mesh._meshType.lower() in ["tensor", "curv"]:
                    # mesh faces are ordered, faces_x, faces_y, faces_z so...
                    is_b = make_boundary_bool(mesh.shape_faces_y)
                    is_t = np.zeros(mesh.shape_faces_y, dtype=bool, order="F")
                    is_t[:, -1] = True
                    is_t = is_t.reshape(-1, order="F")[is_b]
                    not_top = np.ones(boundary_faces.shape[0], dtype=bool)
                    not_top[-len(is_t) :] = ~is_t
                else:
                    raise NotImplementedError(
                        f"Unable to infer surface boundaries for {type(mesh)}, please "
                        f"set the `surface_faces` property."
                    )
            else:
                not_top = ~self.surface_faces

            # use the exponentialy scaled modified bessel function of second kind,
            # (the division will cancel out the scaling)
            # This is more stable for large values of ky * r
            # actual ratio is k1/k0...
            alpha[not_top] = (ky * k1e(ky * r) / k0e(ky * r) * r_dot_n)[not_top]

        B, bc = self.mesh.cell_gradient_weak_form_robin(alpha, beta, gamma)
        # bc should always be 0 because gamma was always 0 above
        self._MBC[ky] = B


class Simulation2DNodal(BaseDCSimulation2D):
    """
    2.5D nodal DC problem
    """

    _solutionType = "phiSolution"
    _formulation = "EB"  # CC potentials means J is on faces
    fieldsPair = Fields2DNodal
    fieldsPair_fwd = Fields3DNodal
    _gradT = None

    def __init__(self, mesh, survey=None, bc_type="Robin", **kwargs):
        super().__init__(mesh=mesh, survey=survey, **kwargs)
        self.solver_opts["is_symmetric"] = True
        self.solver_opts["is_positive_definite"] = True
        self.bc_type = bc_type

    @property
    def bc_type(self):
        """Type of boundary condition to use for simulation.

        Returns
        -------
        {"Neumann", "Robin"}

        Notes
        -----
        Robin and Mixed are equivalent.
        """
        return self._bc_type

    @bc_type.setter
    def bc_type(self, value):
        self._bc_type = validate_string(
            "bc_type", value, ["Neumann", ("Robin", "Mixed")]
        )

    def getA(self, ky):
        """
        Make the A matrix for the cell centered DC resistivity problem
        A = D MfRhoI G
        """
        # To handle Mixed boundary condition
        self.setBC(ky=ky)

        MeSigma = self.MeSigma
        MnSigma = self.MnSigma
        Grad = self.mesh.nodal_gradient
        if self._gradT is None:
            self._gradT = Grad.T.tocsr()  # cache the .tocsr()
        GradT = self._gradT
        A = GradT * MeSigma * Grad + ky**2 * MnSigma

        if self.bc_type != "Neumann":
            try:
                A = A + sdiag(self._AvgBC[ky] @ self.sigma)
            except ValueError as err:
                if len(self.sigma) != len(self.mesh):
                    raise NotImplementedError(
                        "Anisotropic conductivity is not supported for Robin boundary "
                        "conditions, please use 'Neumann'."
                    )
                else:
                    raise err
        return A

    def getADeriv(self, ky, u, v, adjoint=False):
        Grad = self.mesh.nodal_gradient

        if adjoint:
            out = self.MeSigmaDeriv(
                Grad * u, Grad * v, adjoint=adjoint
            ) + ky**2 * self.MnSigmaDeriv(u, v, adjoint=adjoint)
        else:
            out = Grad.T * self.MeSigmaDeriv(
                Grad * u, v, adjoint=adjoint
            ) + ky**2 * self.MnSigmaDeriv(u, v, adjoint=adjoint)
        if self.bc_type != "Neumann" and self.sigmaMap is not None:
            if getattr(self, "_MBC_sigma", None) is None:
                self._MBC_sigma = {}
            if ky not in self._MBC_sigma:
                self._MBC_sigma[ky] = self._AvgBC[ky] @ self.sigmaDeriv
            if not isinstance(u, Zero):
                u = u.flatten()
                if v.ndim > 1:
                    u = u[:, None]
                if not adjoint:
                    out += u * (self._MBC_sigma[ky] @ v)
                else:
                    out += self._MBC_sigma[ky].T @ (u * v)
        return out

    def getRHS(self, ky):
        """
        RHS for the DC problem
        q
        """

        RHS = self.getSourceTerm(ky)
        return RHS

    def getRHSDeriv(self, ky, src, v, adjoint=False):
        """
        Derivative of the right hand side with respect to the model
        """
        # TODO: add qDeriv for RHS depending on m
        # qDeriv = src.evalDeriv(self, ky, adjoint=adjoint)
        # return qDeriv
        return Zero()

    def setBC(self, ky=None):
        if self.bc_type == "Dirichlet":
            # do nothing
            raise ValueError(
                "Dirichlet conditions are not supported in the Nodal formulation"
            )
        elif self.bc_type == "Neumann":
            if self.verbose:
                print(
                    "Homogeneous Neumann is the natural BC for this Nodal discretization."
                )
            return
        else:
            if getattr(self, "_AvgBC", None) is None:
                self._AvgBC = {}
            if ky in self._AvgBC:
                return
            mesh = self.mesh
            # calculate alpha, beta, gamma at the boundary faces
            boundary_faces = mesh.boundary_faces
            boundary_normals = mesh.boundary_face_outward_normals
            n_bf = len(boundary_faces)

            alpha = np.zeros(n_bf)

            # assume a source point at the middle of the top of the mesh
            middle = np.median(mesh.nodes, axis=0)
            top_v = np.max(mesh.nodes[:, -1])
            source_point = np.r_[middle[:-1], top_v]

            r_vec = boundary_faces - source_point
            r = np.linalg.norm(r_vec, axis=-1)
            r_hat = r_vec / r[:, None]
            r_dot_n = np.einsum("ij,ij->i", r_hat, boundary_normals)

            if self.surface_faces is None:
                # determine faces that are on the sides and bottom of the mesh...
                if mesh._meshType.lower() == "tree":
                    not_top = boundary_faces[:, -1] != top_v
                elif mesh._meshType.lower() in ["tensor", "curv"]:
                    # mesh faces are ordered, faces_x, faces_y, faces_z so...
                    is_b = make_boundary_bool(mesh.shape_faces_y)
                    is_t = np.zeros(mesh.shape_faces_y, dtype=bool, order="F")
                    is_t[:, -1] = True
                    is_t = is_t.reshape(-1, order="F")[is_b]
                    not_top = np.ones(boundary_faces.shape[0], dtype=bool)
                    not_top[-len(is_t) :] = ~is_t
                else:
                    raise NotImplementedError(
                        f"Unable to infer surface boundaries for {type(mesh)}, please "
                        f"set the `surface_faces` property."
                    )
            else:
                not_top = ~self.surface_faces

            # use the exponentiall scaled modified bessel function of second kind,
            # (the division will cancel out the scaling)
            # This is more stable for large values of ky * r
            # actual ratio is k1/k0...
            alpha[not_top] = (ky * k1e(ky * r) / k0e(ky * r) * r_dot_n)[not_top]

            P_bf = self.mesh.project_face_to_boundary_face

            AvgN2Fb = P_bf @ self.mesh.average_node_to_face
            AvgCC2Fb = P_bf @ self.mesh.average_cell_to_face

            AvgCC2Fb = sdiag(alpha * (P_bf @ self.mesh.face_areas)) @ AvgCC2Fb
            self._AvgBC[ky] = AvgN2Fb.T @ AvgCC2Fb


Simulation2DCellCentred = Simulation2DCellCentered  # UK and US
