import numpy as np
from scipy.special import k0, k1
from scipy.optimize import minimize
from numpy.polynomial.legendre import leggauss
import warnings
import properties
from ....utils.code_utils import deprecate_class

from ....utils import mkvc, sdiag, Zero
from ...base import BaseEMSimulation
from ....data import Data

from .survey import Survey
from .fields_2d import Fields2D, Fields2DCellCentered, Fields2DNodal
from .fields import FieldsDC, Fields3DCellCentered, Fields3DNodal
from .boundary_utils import getxBCyBC_CC
from .utils import _mini_pole_pole


class BaseDCSimulation2D(BaseEMSimulation):
    """
    Base 2.5D DC problem
    """

    survey = properties.Instance("a DC survey object", Survey, required=True)

    storeJ = properties.Bool("store the sensitivity matrix?", default=False)

    nky = properties.Integer(
        "Number of kys to use in wavenumber space", required=False, default=11
    )

    fieldsPair = Fields2D  # SimPEG.EM.Static.Fields_2D
    fieldsPair_fwd = FieldsDC
    # there's actually nT+1 fields, so we don't need to store the last one
    _Jmatrix = None
    fix_Jmatrix = False
    _mini_survey = None

    def __init__(self, *args, **kwargs):
        miniaturize = kwargs.pop("miniaturize", False)
        super().__init__(*args, **kwargs)

        # try to find an optimal set of quadrature points and weights
        def get_phi(r):
            e = np.ones_like(r)

            def phi(k):
                # use log10 transform to enforce positivity
                k = 10 ** k
                A = r[:, None] * k0(r[:, None] * k)
                v_i = A @ np.linalg.solve(A.T @ A, A.T @ e)
                dv = (e - v_i) / len(r)
                return np.linalg.norm(dv)

            def g(k):
                A = r[:, None] * k0(r[:, None] * k)
                return np.linalg.solve(A.T @ A, A.T @ e)

            return phi, g

        # find the minimum cell spacing, and the maximum side of the mesh
        min_r = min(*[np.min(h) for h in self.mesh.h])
        max_r = max(*[np.sum(h) for h in self.mesh.h])
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

        do_trap = False
        if not out["success"]:
            warnings.warn(
                "Falling back to trapezoidal for integration. "
                "You may need to change nky."
            )
            do_trap = True
        bc_type = getattr(self, "bc_type", "Neumann")
        if bc_type == "Mixed":
            # default for mixed
            do_trap = True
            nky = kwargs.get("nkys", None)
            if nky is None:
                self.nky = 15
        if do_trap:
            if self.verbose:
                print("doing trap")
            y = 0.0
            # gaussian quadrature
            # points, weights = leggauss(self.nky)
            # a, b = -4, 1  # log space of points
            # points = ((b - a)*points + (b+a))/2
            # weights = weights*(b-a)/2
            # weights *= np.log(10)*(10**points)
            # points = 10**points
            #
            # weights *= np.cos(points * y)/np.pi

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
        if miniaturize:
            self._dipoles, self._invs, self._mini_survey = _mini_pole_pole(self.survey)

    def set_geometric_factor(self, geometric_factor):
        index = 0
        for src in self.survey.source_list:
            for rx in src.receiver_list:
                rx._geometric_factor = geometric_factor[index]
                index += 1

    def fields(self, m):
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
        if self._Jmatrix is not None:
            return self._Jmatrix
        else:
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

            # TODO: this loop is pretty slow .. (Parellize)
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
                        P = rx.getP(self.mesh, rx.projGLoc(f)).toarray()

                        ATinvdf_duT = self.Ainv[iky] * (P.T)

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
        toDelete = super(BaseDCSimulation2D, self).deleteTheseOnModelUpdate
        if self.sigmaMap is not None:
            toDelete += ["_MnSigma", "_MnSigmaDerivMat", "_MccRhoi", "_MccRhoiDerivMat"]

        if self.fix_Jmatrix:
            return toDelete

        if self._Jmatrix is not None:
            toDelete += ["_Jmatrix"]
        return toDelete

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

    ####################################################
    # Mass Matrices
    ####################################################

    @property
    def MnSigma(self):
        """
            Node inner product matrix for \\(\\sigma\\). Used in the E-B
            formulation
        """
        # TODO: only works isotropic sigma
        if getattr(self, "_MnSigma", None) is None:
            sigma = self.sigma
            vol = self.mesh.vol
            self._MnSigma = sdiag(self.mesh.aveN2CC.T * (vol * sigma))
        return self._MnSigma

    @property
    def MnSigmaDerivMat(self):
        """
            Derivative of MnSigma with respect to the model
        """
        if getattr(self, "_MnSigmaDerivMat", None) is None:
            vol = self.mesh.vol
            self._MnSigmaDerivMat = self.mesh.aveN2CC.T * sdiag(vol) * self.sigmaDeriv
        return self._MnSigmaDerivMat

    def MnSigmaDeriv(self, u, v, adjoint=False):
        """
            Derivative of MnSigma with respect to the model times a vector (u)
        """
        if v.ndim > 1:
            u = u[:, None]
        if self.storeInnerProduct:
            if adjoint:
                return self.MnSigmaDerivMat.T * (u * v)
            else:
                return u * (self.MnSigmaDerivMat * v)
        else:
            vol = self.mesh.vol
            if v.ndim > 1:
                vol = vol[:, None]
            if adjoint:
                return self.sigmaDeriv.T * (vol * (self.mesh.aveN2CC * (u * v)))
            else:
                dsig_dm_v = self.sigmaDeriv * v
                return u * (self.mesh.aveN2CC.T * (vol * dsig_dm_v))

    @property
    def MccRhoi(self):
        """
            Cell inner product matrix for \\(\\rho^{-1}\\). Used in the H-J
            formulation
        """
        # TODO: only works isotropic rho
        if getattr(self, "_MccRhoi", None) is None:
            self._MccRhoi = sdiag(self.mesh.vol / self.rho)
        return self._MccRhoi

    @property
    def MccRhoiDerivMat(self):
        """
            Derivative of MccRho with respect to the model
        """
        if getattr(self, "_MccRhoiDerivMat", None) is None:
            rho = self.rho
            vol = self.mesh.vol
            self._MccRhoiDerivMat = sdiag(vol * (-1.0 / rho ** 2)) * self.rhoDeriv
        return self._MccRhoiDerivMat

    def MccRhoiDeriv(self, u, v, adjoint=False):
        """
            Derivative of :code:`MccRhoi` with respect to the model.
        """
        if self.rhoMap is None:
            return Zero()

        if len(self.rho.shape) > 1:
            if self.rho.shape[1] > self.mesh.dim:
                raise NotImplementedError(
                    "Full anisotropy is not implemented for MccRhoiDeriv."
                )
        if self.storeInnerProduct:
            if adjoint:
                return self.MccRhoiDerivMat.T * (sdiag(u) * v)
            else:
                return sdiag(u) * (self.MccRhoiDerivMat * v)
        else:
            vol = self.mesh.vol
            rho = self.rho
            if adjoint:
                return self.rhoDeriv.T * (sdiag(u * vol * (-1.0 / rho ** 2)) * v)
            else:
                return (sdiag(u * vol * (-1.0 / rho ** 2))) * (self.rhoDeriv * v)


class Simulation2DCellCentered(BaseDCSimulation2D):
    """
    2.5D cell centered DC problem
    """

    _solutionType = "phiSolution"
    _formulation = "HJ"  # CC potentials means J is on faces
    fieldsPair = Fields2DCellCentered
    fieldsPair_fwd = Fields3DCellCentered
    bc_type = "Mixed"

    def __init__(self, mesh, **kwargs):
        BaseDCSimulation2D.__init__(self, mesh, **kwargs)

    def getA(self, ky):
        """
        Make the A matrix for the cell centered DC resistivity problem
        A = D MfRhoI G
        """
        # To handle Mixed boundary condition
        self.setBC(ky=ky)
        D = self.Div
        G = self.Grad
        vol = self.mesh.vol
        MfRhoI = self.MfRhoI
        # Get resistivity rho
        rho = self.rho
        A = D * MfRhoI * G + ky ** 2 * self.MccRhoi
        if self.bc_type == "Neumann":
            A[0, 0] = A[0, 0] + 1.0
        return A

    def getADeriv(self, ky, u, v, adjoint=False):
        # To handle Mixed boundary condition
        # self.setBC(ky=ky)

        D = self.Div
        G = self.Grad
        if adjoint:
            return self.MfRhoIDeriv(
                G * u.flatten(), D.T * v, adjoint=adjoint
            ) + ky ** 2 * self.MccRhoiDeriv(u.flatten(), v, adjoint=adjoint)
        else:
            return D * self.MfRhoIDeriv(
                G * u.flatten(), v, adjoint=adjoint
            ) + ky ** 2 * self.MccRhoiDeriv(u.flatten(), v, adjoint=adjoint)

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
        fxm, fxp, fym, fyp = self.mesh.faceBoundaryInd
        gBFxm = self.mesh.gridFx[fxm, :]
        gBFxp = self.mesh.gridFx[fxp, :]
        gBFym = self.mesh.gridFy[fym, :]
        gBFyp = self.mesh.gridFy[fyp, :]

        # Setup Mixed B.C (alpha, beta, gamma)
        temp_xm = np.ones_like(gBFxm[:, 0])
        temp_xp = np.ones_like(gBFxp[:, 0])
        temp_ym = np.ones_like(gBFym[:, 1])
        temp_yp = np.ones_like(gBFyp[:, 1])

        if self.bc_type == "Neumann":
            alpha_xm, alpha_xp = temp_xm * 0.0, temp_xp * 0.0
            alpha_ym, alpha_yp = temp_ym * 0.0, temp_yp * 0.0

            beta_xm, beta_xp = temp_xm, temp_xp
            beta_ym, beta_yp = temp_ym, temp_yp

            gamma_xm, gamma_xp = temp_xm * 0.0, temp_xp * 0.0
            gamma_ym, gamma_yp = temp_ym * 0.0, temp_yp * 0.0

        elif self.bc_type == "Dirichlet":
            alpha_xm, alpha_xp = temp_xm, temp_xp
            alpha_ym, alpha_yp = temp_ym, temp_yp

            beta_xm, beta_xp = temp_xm * 0.0, temp_xp * 0.0
            beta_ym, beta_yp = temp_ym * 0.0, temp_yp * 0.0

            gamma_xm, gamma_xp = temp_xm * 0.0, temp_xp * 0.0
            gamma_ym, gamma_yp = temp_ym * 0.0, temp_yp * 0.0

        elif self.bc_type == "Mixed":
            xs = np.median(self.mesh.vectorCCx)
            ys = np.median(self.mesh.vectorCCy[-1])

            def r_boundary(x, y):
                return 1.0 / np.sqrt((x - xs) ** 2 + (y - ys) ** 2)

            rxm = r_boundary(gBFxm[:, 0], gBFxm[:, 1])
            rxp = r_boundary(gBFxp[:, 0], gBFxp[:, 1])
            rym = r_boundary(gBFym[:, 0], gBFym[:, 1])

            alpha_xm = ky * (k1(ky * rxm) / k0(ky * rxm) * (gBFxm[:, 0] - xs))
            alpha_xp = ky * (k1(ky * rxp) / k0(ky * rxp) * (gBFxp[:, 0] - xs))
            alpha_ym = ky * (k1(ky * rym) / k0(ky * rym) * (gBFym[:, 0] - ys))
            alpha_yp = temp_yp * 0.0
            beta_xm, beta_xp = temp_xm, temp_xp
            beta_ym, beta_yp = temp_ym, temp_yp

            gamma_xm, gamma_xp = temp_xm * 0.0, temp_xp * 0.0
            gamma_ym, gamma_yp = temp_ym * 0.0, temp_yp * 0.0

        alpha = [alpha_xm, alpha_xp, alpha_ym, alpha_yp]
        beta = [beta_xm, beta_xp, beta_ym, beta_yp]
        gamma = [gamma_xm, gamma_xp, gamma_ym, gamma_yp]

        x_BC, y_BC = getxBCyBC_CC(self.mesh, alpha, beta, gamma)
        V = self.Vol
        self.Div = V * self.mesh.faceDiv
        P_BC, B = self.mesh.getBCProjWF_simple()
        M = B * self.mesh.aveCC2F
        self.Grad = self.Div.T - P_BC * sdiag(y_BC) * M


class Simulation2DNodal(BaseDCSimulation2D):
    """
    2.5D nodal DC problem
    """

    _solutionType = "phiSolution"
    _formulation = "EB"  # CC potentials means J is on faces
    fieldsPair = Fields2DNodal
    fieldsPair_fwd = Fields3DNodal
    _gradT = None

    def __init__(self, mesh, **kwargs):
        BaseDCSimulation2D.__init__(self, mesh, **kwargs)
        # self.setBC()
        self.solver_opts["is_symmetric"] = True
        self.solver_opts["is_positive_definite"] = True

    def getA(self, ky):
        """
        Make the A matrix for the cell centered DC resistivity problem
        A = D MfRhoI G
        """
        MeSigma = self.MeSigma
        MnSigma = self.MnSigma
        Grad = self.mesh.nodalGrad
        if self._gradT is None:
            self._gradT = Grad.T.tocsr()  # cache the .tocsr()
        GradT = self._gradT
        A = GradT * MeSigma * Grad + ky ** 2 * MnSigma
        return A

    def getADeriv(self, ky, u, v, adjoint=False):

        Grad = self.mesh.nodalGrad

        if adjoint:
            return self.MeSigmaDeriv(
                Grad * u.flatten(), Grad * v, adjoint=adjoint
            ) + ky ** 2 * self.MnSigmaDeriv(u.flatten(), v, adjoint=adjoint)
        else:
            return Grad.T * self.MeSigmaDeriv(
                Grad * u.flatten(), v, adjoint=adjoint
            ) + ky ** 2 * self.MnSigmaDeriv(u.flatten(), v, adjoint=adjoint)
        # return (Grad.T*(self.MeSigmaDeriv(Grad*u.flatten(), v, adjoint)) +
        #         ky**2*self.MnSigmaDeriv(u.flatten())*v)

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


Simulation2DCellCentred = Simulation2DCellCentered  # UK and US


############
# Deprecated
############


@deprecate_class(removal_version="0.15.0")
class Problem2D_N(Simulation2DNodal):
    pass


@deprecate_class(removal_version="0.15.0")
class Problem2D_CC(Simulation2DCellCentered):
    pass
