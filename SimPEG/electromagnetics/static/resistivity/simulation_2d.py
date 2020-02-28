import numpy as np
from scipy.special import kn
import properties

from ....utils import mkvc, sdiag, Zero
from ...base import BaseEMSimulation
from ....data import Data

from .survey import Survey
from .fields_2d import Fields_ky, Fields_ky_CC, Fields_ky_N
from .fields import FieldsDC, Fields_CC, Fields_N
from .boundary_utils import getxBCyBC_CC


class BaseDCSimulation_2D(BaseEMSimulation):
    """
    Base 2.5D DC problem
    """

    survey = properties.Instance(
        "a DC survey object", Survey, required=True
    )

    storeJ = properties.Bool(
        "store the sensitivity", default=False
    )

    fieldsPair = Fields_ky  # SimPEG.EM.Static.Fields_2D
    fieldsPair_fwd = FieldsDC
    nky = 15
    kys = np.logspace(-4, 1, nky)
    Ainv = [None for i in range(nky)]
    nT = nky  # Only for using TimeFields
    _Jmatrix = None
    fix_Jmatrix = False

    def set_geometric_factor(self, geometric_factor):
        index = 0
        for src in self.survey.source_list:
            for rx in src.receiver_list:
                rx._geometric_factor = geometric_factor[index]
                index += 1

    def fields(self, m):
        if self.verbose:
            print (">> Compute fields")
        if m is not None:
            self.model = m
        if self.Ainv[0] is not None:
            for i in range(self.nky):
                self.Ainv[i].clean()
        f = self.fieldsPair(self)
        #Srcs = self.survey.source_list
        for iky, ky in enumerate(self.kys):
            A = self.getA(ky)
            self.Ainv[iky] = self.Solver(A, **self.solver_opts)
            RHS = self.getRHS(ky)
            u = self.Ainv[iky] * RHS
            f[:, self._solutionType, iky] = u
        return f

    def fields_to_space(self, f, y=0.):
        f_fwd = self.fieldsPair_fwd(self.mesh, self.survey)
        # Evaluating Integration using Trapezoidal rules
        dky = np.diff(self.kys)/2
        trap_weights = np.r_[dky, 0]+np.r_[0, dky]
        trap_weights *= np.cos(self.kys*y)
        # assume constant value at 0 frequency?
        trap_weights[0] += self.kys[0]/2 * (1.0 + np.cos(self.kys[0]*y))
        trap_weights /= np.pi
        phi = f[:, self._solutionType, :-1].dot(trap_weights)
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

        data = Data(self.survey)
        kys = self.kys
        cnt = 0
        for src in self.survey.source_list:
            for rx in src.receiver_list:
                data[src, rx] = rx.eval(kys, src, self.mesh, f)

        return mkvc(data)

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

        # TODO: This is not a good idea !! should change that as a list
        Jv = Data(self.survey)  # same size as the data

        # Assume y=0.
        # This needs some thoughts to implement in general when src is dipole
        y = 0.
        dky = np.diff(self.kys)/2
        trap_weights = np.r_[dky, 0]+np.r_[0, dky]
        trap_weights *= np.cos(self.kys*y)  # *(1.0/np.pi)
        # assume constant value at 0 frequency?
        trap_weights[0] += self.kys[0]/2 * (1.0 + np.cos(self.kys[0]*y))
        trap_weights /= np.pi

        Jv.dobs[:] = 0.0  # initialize Jv object with zeros

        # TODO: this loop is pretty slow .. (Parellize)
        for iky, ky in enumerate(self.kys):
            u_ky = f[:, self._solutionType, iky]
            for i_src, src in enumerate(self.survey.source_list):
                u_src = u_ky[:, i_src]
                dA_dm_v = self.getADeriv(ky, u_src, v, adjoint=False)
                # dRHS_dm_v = self.getRHSDeriv(ky, src, v) = 0
                du_dm_v = self.Ainv[iky] * (-dA_dm_v)  # + dRHS_dm_v)
                for rx in src.receiver_list:
                    df_dmFun = getattr(f, '_{0!s}Deriv'.format(rx.projField),
                                       None)
                    df_dm_v = df_dmFun(iky, src, du_dm_v, v, adjoint=False)
                    Jv1_temp = rx.evalDeriv(ky, src, self.mesh, f, df_dm_v)
                    # Trapezoidal intergration
                    Jv[src, rx] += trap_weights[iky]*Jv1_temp
        return mkvc(Jv)

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

        # Assume y=0.
        # This needs some thoughts to implement in general when src is dipole
        y = 0.
        dky = np.diff(self.kys)/2
        trap_weights = np.r_[dky, 0]+np.r_[0, dky]
        trap_weights *= np.cos(self.kys*y)  # *(1.0/np.pi)
        # assume constant value at 0 frequency?
        trap_weights[0] += self.kys[0]/2 * (1.0 + np.cos(self.kys[0]*y))
        trap_weights /= np.pi

        if v is not None:
            # Ensure v is a data object.
            if not isinstance(v, Data):
                v = Data(self.survey, v)
            Jtv = np.zeros(m.size, dtype=float)

            # TODO: this loop is pretty slow .. (Parellize)
            for iky, ky in enumerate(self.kys):
                u_ky = f[:, self._solutionType, iky]
                for i_src, src in enumerate(self.survey.source_list):
                    u_src = u_ky[:, i_src]
                    df_duT_sum = 0
                    df_dmT_sum = 0
                    for rx in src.receiver_list:
                        # wrt f, need possibility wrt m
                        PTv = rx.evalDeriv(ky, src, self.mesh, f, v[src, rx],
                                           adjoint=True)
                        df_duTFun = getattr(
                            f, '_{0!s}Deriv'.format(rx.projField), None
                        )
                        df_duT, df_dmT = df_duTFun(iky, src, None, PTv,
                                                   adjoint=True)
                        df_duT_sum += df_duT
                        df_dmT_sum += df_dmT

                    ATinvdf_duT = self.Ainv[iky] * df_duT_sum

                    dA_dmT = self.getADeriv(ky, u_src, ATinvdf_duT,
                                            adjoint=True)
                    # dRHS_dmT = self.getRHSDeriv(ky, src, ATinvdf_duT,
                    #                            adjoint=True)
                    du_dmT = -dA_dmT  # + dRHS_dmT=0
                    Jtv += trap_weights[iky]*(df_dmT + du_dmT).astype(float)
            return mkvc(Jtv)

        else:
            # This is for forming full sensitivity matrix
            Jt = np.zeros((self.model.size, self.survey.nD), order='F')
            for iky, ky in enumerate(self.kys):
                u_ky = f[:, self._solutionType, iky]
                istrt = 0
                for i_src, src in enumerate(self.survey.source_list):
                    u_src = u_ky[:, i_src]
                    for rx in src.receiver_list:
                        # wrt f, need possibility wrt m
                        P = rx.getP(self.mesh, rx.projGLoc(f)).toarray()

                        ATinvdf_duT = self.Ainv[iky] * (P.T)

                        dA_dmT = self.getADeriv(ky, u_src, ATinvdf_duT,
                                                adjoint=True)
                        Jtv = -trap_weights[iky]*dA_dmT #RHS=0
                        iend = istrt + rx.nD
                        if rx.nD == 1:
                            Jt[:, istrt] += Jtv
                        else:
                            Jt[:, istrt:iend] += Jtv
                        istrt += rx.nD
            return Jt

    def getSourceTerm(self, ky):
        """
        takes concept of source and turns it into a matrix
        """
        """
        Evaluates the sources, and puts them in matrix form
        :rtype: (numpy.ndarray, numpy.ndarray)
        :return: q (nC or nN, nSrc)
        """

        Srcs = self.survey.source_list

        if self._formulation == 'EB':
            n = self.mesh.nN
            # return NotImplementedError

        elif self._formulation == 'HJ':
            n = self.mesh.nC

        q = np.zeros((n, len(Srcs)), order='F')

        for i, src in enumerate(Srcs):
            q[:, i] = src.eval(self)
        return q

    @property
    def deleteTheseOnModelUpdate(self):
        toDelete = super(BaseDCSimulation_2D, self).deleteTheseOnModelUpdate
        if self.sigmaMap is not None:
            toDelete += [
                '_MnSigma', '_MnSigmaDerivMat',
                '_MccRhoi', '_MccRhoiDerivMat'
            ]

        if self.fix_Jmatrix:
            return toDelete

        if self._Jmatrix is not None:
            toDelete += ['_Jmatrix']
        return toDelete

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
        if getattr(self, '_MnSigma', None) is None:
            sigma = self.sigma
            vol = self.mesh.vol
            self._MnSigma = sdiag(
                self.mesh.aveN2CC.T*(vol*sigma)
            )
        return self._MnSigma

    @property
    def MnSigmaDerivMat(self):
        """
            Derivative of MnSigma with respect to the model
        """
        if getattr(self, '_MnSigmaDerivMat', None) is None:
            vol = self.mesh.vol
            self._MnSigmaDerivMat = (
                self.mesh.aveN2CC.T * sdiag(vol) * self.sigmaDeriv
                )
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
                return u*(self.MnSigmaDerivMat * v)
        else:
            vol = self.mesh.vol
            if v.ndim > 1:
                vol = vol[:, None]
            if adjoint:
                return self.sigmaDeriv.T * (
                    vol * (self.mesh.aveN2CC * (u * v))
                )
            else:
                dsig_dm_v = self.sigmaDeriv * v
                return (
                    u * (self.mesh.aveN2CC.T * (vol * dsig_dm_v))
                )

    @property
    def MccRhoi(self):
        """
            Cell inner product matrix for \\(\\rho^{-1}\\). Used in the H-J
            formulation
        """
        # TODO: only works isotropic rho
        if getattr(self, '_MccRhoi', None) is None:
            self._MccRhoi = sdiag(
                self.mesh.vol/self.rho
            )
        return self._MccRhoi

    @property
    def MccRhoiDerivMat(self):
        """
            Derivative of MccRho with respect to the model
        """
        if getattr(self, '_MccRhoiDerivMat', None) is None:
            rho = self.rho
            vol = self.mesh.vol
            self._MccRhoiDerivMat = (
                sdiag(vol*(-1./rho**2))*self.rhoDeriv
            )
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
                return self.rhoDeriv.T * (sdiag(u*vol*(-1./rho**2)) * v)
            else:
                return (sdiag(u*vol*(-1./rho**2)))*(self.rhoDeriv * v)


class Problem2D_CC(BaseDCSimulation_2D):
    """
    2.5D cell centered DC problem
    """

    _solutionType = 'phiSolution'
    _formulation = 'HJ'  # CC potentials means J is on faces
    fieldsPair = Fields_ky_CC
    fieldsPair_fwd = Fields_CC
    bc_type = 'Mixed'

    def __init__(self, mesh, **kwargs):
        BaseDCSimulation_2D.__init__(self, mesh, **kwargs)

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
        A = D * MfRhoI * G + ky**2 * self.MccRhoi
        if self.bc_type == "Neumann":
            A[0, 0] = A[0, 0] + 1.
        return A

    def getADeriv(self, ky, u, v, adjoint=False):
        # To handle Mixed boundary condition
        # self.setBC(ky=ky)

        D = self.Div
        G = self.Grad
        if adjoint:
            return (
                self.MfRhoIDeriv(G*u.flatten(), D.T*v, adjoint=adjoint) +
                ky**2 * self.MccRhoiDeriv(u.flatten(), v, adjoint=adjoint)
            )
        else:
            return (
                D * self.MfRhoIDeriv(G*u.flatten(), v, adjoint=adjoint) +
                ky**2 * self.MccRhoiDeriv(u.flatten(), v, adjoint=adjoint)
            )

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
            alpha_xm, alpha_xp = temp_xm*0., temp_xp*0.
            alpha_ym, alpha_yp = temp_ym*0., temp_yp*0.

            beta_xm, beta_xp = temp_xm, temp_xp
            beta_ym, beta_yp = temp_ym, temp_yp

            gamma_xm, gamma_xp = temp_xm*0., temp_xp*0.
            gamma_ym, gamma_yp = temp_ym*0., temp_yp*0.

        elif self.bc_type == "Dirichlet":
            alpha_xm, alpha_xp = temp_xm, temp_xp
            alpha_ym, alpha_yp = temp_ym, temp_yp

            beta_xm, beta_xp = temp_xm*0., temp_xp*0.
            beta_ym, beta_yp = temp_ym*0., temp_yp*0.

            gamma_xm, gamma_xp = temp_xm*0., temp_xp*0.
            gamma_ym, gamma_yp = temp_ym*0., temp_yp*0.

        elif self.bc_type == "Mixed":
            xs = np.median(self.mesh.vectorCCx)
            ys = np.median(self.mesh.vectorCCy[-1])

            def r_boundary(x, y):
                return 1./np.sqrt(
                    (x - xs)**2 + (y - ys)**2
                    )

            rxm = r_boundary(gBFxm[:, 0], gBFxm[:, 1])
            rxp = r_boundary(gBFxp[:, 0], gBFxp[:, 1])
            rym = r_boundary(gBFym[:, 0], gBFym[:, 1])

            alpha_xm = ky*(
                kn(1, ky*rxm) / kn(0, ky*rxm) * (gBFxm[:, 0]-xs)
                )
            alpha_xp = ky*(
                kn(1, ky*rxp) / kn(0, ky*rxp) * (gBFxp[:, 0]-xs)
                )
            alpha_ym = ky*(
                kn(1, ky*rym) / kn(0, ky*rym) * (gBFym[:, 0]-ys)
                )
            alpha_yp = temp_yp*0.
            beta_xm, beta_xp = temp_xm, temp_xp
            beta_ym, beta_yp = temp_ym, temp_yp

            gamma_xm, gamma_xp = temp_xm*0., temp_xp*0.
            gamma_ym, gamma_yp = temp_ym*0., temp_yp*0.

        alpha = [alpha_xm, alpha_xp, alpha_ym, alpha_yp]
        beta = [beta_xm, beta_xp, beta_ym, beta_yp]
        gamma = [gamma_xm, gamma_xp, gamma_ym, gamma_yp]

        x_BC, y_BC = getxBCyBC_CC(self.mesh, alpha, beta, gamma)
        V = self.Vol
        self.Div = V * self.mesh.faceDiv
        P_BC, B = self.mesh.getBCProjWF_simple()
        M = B*self.mesh.aveCC2F
        self.Grad = self.Div.T - P_BC*sdiag(y_BC)*M


class Problem2D_N(BaseDCSimulation_2D):
    """
    2.5D nodal DC problem
    """

    _solutionType = 'phiSolution'
    _formulation = 'EB'  # CC potentials means J is on faces
    fieldsPair = Fields_ky_N
    fieldsPair_fwd = Fields_N

    def __init__(self, mesh, **kwargs):
        BaseDCSimulation_2D.__init__(self, mesh, **kwargs)
        # self.setBC()
        self._grad = self.mesh.nodalGrad
        self._gradT = self._grad.T.tocsr()
        self.solver_opts['is_symmetric'] = True
        self.solver_opts['is_positive_definite'] = True

    def getA(self, ky):
        """
        Make the A matrix for the cell centered DC resistivity problem
        A = D MfRhoI G
        """
        MeSigma = self.MeSigma
        MnSigma = self.MnSigma
        A = self._gradT * MeSigma * self._grad + ky**2*MnSigma
        return A

    def getADeriv(self, ky, u, v, adjoint=False):

        Grad = self.mesh.nodalGrad

        if adjoint:
            return (
                self.MeSigmaDeriv(Grad*u.flatten(), Grad*v, adjoint=adjoint) +
                ky**2*self.MnSigmaDeriv(u.flatten(), v, adjoint=adjoint)
            )
        else:
            return (
                Grad.T*self.MeSigmaDeriv(Grad*u.flatten(), v, adjoint=adjoint) +
                ky**2*self.MnSigmaDeriv(u.flatten(), v, adjoint=adjoint)
            )
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
