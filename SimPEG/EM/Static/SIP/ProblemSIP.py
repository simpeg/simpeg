from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import sys

from SimPEG import Utils
from SimPEG import Props
from SimPEG import Maps

from SimPEG.EM.Base import BaseEMProblem
from SimPEG.EM.Static.DC.FieldsDC import FieldsDC, Fields_CC, Fields_N
from SimPEG.EM.Static.DC import getxBCyBC_CC
from .SurveySIP import Survey, Data
import gc


class BaseSIPProblem(BaseEMProblem):

    sigma = Props.PhysicalProperty(
        "Electrical conductivity (S/m)"
    )

    rho = Props.PhysicalProperty(
        "Electrical resistivity (Ohm m)"
    )

    Props.Reciprocal(sigma, rho)

    eta, etaMap, etaDeriv = Props.Invertible(
        "Electrical Chargeability (V/V)"
    )

    tau, tauMap, tauDeriv = Props.Invertible(
        "Time constant (s)",
        default=0.1
    )

    taui, tauiMap, tauiDeriv = Props.Invertible(
        "Inverse of time constant (1/s)"
    )

    c, cMap, cDeriv = Props.Invertible(
        "Frequency dependency",
        default=0.5
    )

    Props.Reciprocal(tau, taui)

    surveyPair = Survey
    fieldsPair = FieldsDC
    dataPair = Data
    Ainv = None
    _f = None
    actinds = None
    storeJ = False
    _Jmatrix = None
    actMap = None

    def getPeta(self, t):
        peta = self.eta*np.exp(-(self.taui*t)**self.c)
        return peta

    def PetaEtaDeriv(self, t, v, adjoint=False):
        v = np.array(v, dtype=float)
        taui_t_c = (self.taui*t)**self.c
        dpetadeta = np.exp(-taui_t_c)
        if adjoint:
            return self.etaDeriv.T * (dpetadeta * v)
        else:
            return dpetadeta * (self.etaDeriv*v)

    def PetaTauiDeriv(self, t, v, adjoint=False):
        v = np.array(v, dtype=float)
        taui_t_c = (self.taui*t)**self.c
        dpetadtaui = (
            - self.c * self.eta / self.taui * taui_t_c * np.exp(-taui_t_c)
            )
        if adjoint:
            return self.tauiDeriv.T * (dpetadtaui*v)
        else:
            return dpetadtaui * (self.tauiDeriv*v)

    def PetaCDeriv(self, t, v, adjoint=False):
        v = np.array(v, dtype=float)
        taui_t_c = (self.taui*t)**self.c
        dpetadc = (
            -self.eta * (taui_t_c)*np.exp(-taui_t_c) * np.log(self.taui*t)
            )
        if adjoint:
            return self.cDeriv.T * (dpetadc*v)
        else:
            return dpetadc * (self.cDeriv*v)

    def fields(self, m):
        if self.verbose:
            print (">> Compute DC fields")
        if self._f is None:
            self._f = self.fieldsPair(self.mesh, self.survey)
            if self.Ainv is None:
                A = self.getA()
                self.Ainv = self.Solver(A, **self.solverOpts)
            RHS = self.getRHS()
            u = self.Ainv * RHS
            Srcs = self.survey.srcList
            self._f[Srcs, self._solutionType] = u
        return self._f

    def getJ(self, m, f=None):
        """
            Generate Full sensitivity matrix
        """

        if self.verbose:
            print("Calculating J and storing")

        if self._Jmatrix is not None:
            return self._Jmatrix
        else:

            if f is None:
                f = self.fields(m)

            Jt = []

            for isrc, src in enumerate(self.survey.srcList):
                if self.verbose:
                    sys.stdout.write(("\r %d / %d") % (isrc+1, self.survey.nSrc))
                    sys.stdout.flush()
                u_src = f[src, self._solutionType]
                for rx in src.rxList:
                    P = rx.getP(self.mesh, rx.projGLoc(f)).toarray()
                    ATinvdf_duT = self.Ainv * (P.T)
                    dA_dmT = self.getADeriv(u_src, ATinvdf_duT, adjoint=True)
                    if rx.nD == 1:
                        # Consider when rx has one location
                        Jt.append(-dA_dmT.reshape([-1, 1]))
                    else:
                        Jt.append(-dA_dmT)
            self._Jmatrix = np.hstack(Jt).T
            if self.verbose:
                collected = gc.collect()
                print (
                    "Garbage collector: collected %d objects." % (collected)
                )

            # Not sure why below has raise memory issue
            # only for problem_cc, test_dataObj
            # if self._f is not None:
            #     del self._f
            # clean all factorization
            if self.Ainv is not None:
                self.Ainv.clean()

            return self._Jmatrix

    def forward(self, m, f=None):

        self.model = m
        Jv = []

        # When sensitivity matrix is stored
        if self.storeJ:
            J = self.getJ(m, f=f)

            ntime = len(self.survey.times)

            self.model = m
            for tind in range(ntime):
                Jv.append(
                    J.dot(
                        self.actMap.P.T*self.getPeta(self.survey.times[tind]))
                    )
            return self.sign * np.hstack(Jv)

        # Do not store sensitivity matrix (memory-wise efficient)
        else:

            if f is None:
                f = self.fields(m)

            # A = self.getA()
            for tind in range(len(self.survey.times)):
                # Pseudo-chareability
                t = self.survey.times[tind]
                v = self.getPeta(t)
                for src in self.survey.srcList:
                    u_src = f[src, self._solutionType]  # solution vector
                    dA_dm_v = self.getADeriv(u_src, v)
                    dRHS_dm_v = self.getRHSDeriv(src, v)
                    du_dm_v = self.Ainv * (- dA_dm_v + dRHS_dm_v)
                    for rx in src.rxList:
                        timeindex = rx.getTimeP(self.survey.times)
                        if timeindex[tind]:
                            df_dmFun = getattr(
                                f, '_{0!s}Deriv'.format(rx.projField),
                                None
                                )
                            df_dm_v = df_dmFun(src, du_dm_v, v, adjoint=False)
                            Jv.append(
                                rx.evalDeriv(src, self.mesh, f, df_dm_v)
                                )

            return self.sign*np.hstack(Jv)

    def Jvec(self, m, v, f=None):

        self.model = m

        Jv = []

        # When sensitivity matrix is stored
        if self.storeJ:
            J = self.getJ(m, f=f)
            ntime = len(self.survey.times)

            for tind in range(ntime):

                t = self.survey.times[tind]
                v0 = self.PetaEtaDeriv(t, v)
                v1 = self.PetaTauiDeriv(t, v)
                v2 = self.PetaCDeriv(t, v)
                PTv = self.actMap.P.T*(v0+v1+v2)
                Jv.append(J.dot(PTv))

            return self.sign * np.hstack(Jv)

        # Do not store sensitivity matrix (memory-wise efficient)
        else:

            if f is None:
                f = self.fields(m)

            for tind in range(len(self.survey.times)):

                t = self.survey.times[tind]
                v0 = self.PetaEtaDeriv(t, v)
                v1 = self.PetaTauiDeriv(t, v)
                v2 = self.PetaCDeriv(t, v)

                for src in self.survey.srcList:
                    u_src = f[src, self._solutionType]  # solution vector
                    dA_dm_v = self.getADeriv(u_src, v0+v1+v2)
                    dRHS_dm_v = self.getRHSDeriv(src, v0+v1+v2)
                    du_dm_v = self.Ainv * (
                        - dA_dm_v + dRHS_dm_v
                        )

                    for rx in src.rxList:
                        timeindex = rx.getTimeP(self.survey.times)
                        if timeindex[tind]:
                            df_dmFun = getattr(
                                f, '_{0!s}Deriv'.format(rx.projField),
                                None
                                )
                            # df_dm_v = df_dmFun(
                            #     src, du_dm_v, v0+v1+v2, adjoint=False
                            #     )

                            Jv_temp = (
                                rx.evalDeriv(src, self.mesh, f, du_dm_v)
                                )
                            if rx.nD == 1:
                                Jv_temp = Jv_temp.reshape([-1, 1])

                            Jv.append(Jv_temp)

            return self.sign*np.hstack(Jv)

    def Jtvec(self, m, v, f=None):

        self.model = m

        # When sensitivity matrix is stored
        if self.storeJ:
            J = self.getJ(m, f=f)
            ntime = len(self.survey.times)
            Jtvec = np.zeros(m.size)
            v = v.reshape((int(self.survey.nD/ntime), ntime), order="F")

            for tind in range(ntime):
                t = self.survey.times[tind]
                Jtv = self.actMap.P*J.T.dot(v[:, tind])
                Jtvec += (
                    self.PetaEtaDeriv(t, Jtv, adjoint=True) +
                    self.PetaTauiDeriv(t, Jtv, adjoint=True) +
                    self.PetaCDeriv(t, Jtv, adjoint=True)
                    )

            return self.sign * Jtvec

        # Do not store sensitivity matrix (memory-wise efficient)
        else:

            if f is None:
                f = self.fields(m)

            # Ensure v is a data object.
            if not isinstance(v, self.dataPair):
                v = self.dataPair(self.survey, v)

            Jtv = np.zeros(m.size)

            for tind in range(len(self.survey.times)):
                t = self.survey.times[tind]
                for src in self.survey.srcList:
                    u_src = f[src, self._solutionType]
                    for rx in src.rxList:
                        timeindex = rx.getTimeP(self.survey.times)
                        if timeindex[tind]:
                            # wrt f, need possibility wrt m
                            PTv = rx.evalDeriv(
                                src, self.mesh, f, v[src, rx, t], adjoint=True
                            )
                            df_duTFun = getattr(
                                f, '_{0!s}Deriv'.format(rx.projField), None
                                )
                            df_duT, df_dmT = df_duTFun(
                                src, None, PTv, adjoint=True
                                )
                            ATinvdf_duT = self.Ainv * df_duT
                            dA_dmT = self.getADeriv(
                                u_src, ATinvdf_duT, adjoint=True
                                )
                            dRHS_dmT = self.getRHSDeriv(
                                src, ATinvdf_duT, adjoint=True
                                )
                            du_dmT = -dA_dmT + dRHS_dmT
                            Jtv += (
                                self.PetaEtaDeriv(
                                    self.survey.times[tind], du_dmT, adjoint=True
                                    ) +
                                self.PetaTauiDeriv(
                                    self.survey.times[tind], du_dmT, adjoint=True
                                    ) +
                                self.PetaCDeriv(
                                    self.survey.times[tind], du_dmT, adjoint=True
                                    )
                                )

            return self.sign*Jtv

    def getSourceTerm(self):
        """
        takes concept of source and turns it into a matrix
        """
        """
        Evaluates the sources, and puts them in matrix form

        :rtype: (numpy.ndarray, numpy.ndarray)
        :return: q (nC or nN, nSrc)
        """

        Srcs = self.survey.srcList

        if self._formulation == 'EB':
            n = self.mesh.nN
            # return NotImplementedError

        elif self._formulation == 'HJ':
            n = self.mesh.nC

        q = np.zeros((n, len(Srcs)))

        for i, src in enumerate(Srcs):
            q[:, i] = src.eval(self)
        return q

    @property
    def deleteTheseOnModelUpdate(self):
        toDelete = []
        return toDelete

    @property
    def MfRhoDerivMat(self):
        """
        Derivative of MfRho with respect to the model
        """
        if getattr(self, '_MfRhoDerivMat', None) is None:
            if self.storeJ:
                drho_dlogrho = Utils.sdiag(self.rho)*self.actMap.P
            else:
                drho_dlogrho = Utils.sdiag(self.rho)
            self._MfRhoDerivMat = self.mesh.getFaceInnerProductDeriv(
                np.ones(self.mesh.nC)
            )(np.ones(self.mesh.nF)) * drho_dlogrho
        return self._MfRhoDerivMat

    def MfRhoIDeriv(self, u, v, adjoint=False):
        """
            Derivative of :code:`MfRhoI` with respect to the model.
        """
        dMfRhoI_dI = -self.MfRhoI**2

        if self.storeJ:
            drho_dlogrho = Utils.sdiag(self.rho)*self.actMap.P
        else:
            drho_dlogrho = Utils.sdiag(self.rho)

        if self.storeInnerProduct:
            if adjoint:
                return self.MfRhoDerivMat.T * (Utils.sdiag(u) * (dMfRhoI_dI.T * v))
            else:
                return dMfRhoI_dI * (Utils.sdiag(u) * (self.MfRhoDerivMat*v))
        else:
            dMf_drho = self.mesh.getFaceInnerProductDeriv(self.rho)(u)
            drho_dlogrho = Utils.sdiag(self.rho)*self.etaDeriv
            if adjoint:
                return drho_dlogrho.T * (dMf_drho.T * (dMfRhoI_dI.T*v))
            else:
                return dMfRhoI_dI * (dMf_drho * (drho_dlogrho*v))

    @property
    def MeSigmaDerivMat(self):
        """
        Derivative of MeSigma with respect to the model
        """
        if getattr(self, '_MeSigmaDerivMat', None) is None:
            if self.storeJ:
                dsigma_dlogsigma = Utils.sdiag(self.sigma)*self.actMap.P
            else:
                dsigma_dlogsigma = Utils.sdiag(self.sigma)
            self._MeSigmaDerivMat = self.mesh.getEdgeInnerProductDeriv(
                np.ones(self.mesh.nC)
            )(np.ones(self.mesh.nE)) * dsigma_dlogsigma
        return self._MeSigmaDerivMat

    # TODO: This should take a vector
    def MeSigmaDeriv(self, u, v, adjoint=False):
        """
        Derivative of MeSigma with respect to the model times a vector (u)
        """
        if self.storeJ:
            dsigma_dlogsigma = Utils.sdiag(self.sigma)*self.actMap.P
        else:
            dsigma_dlogsigma = Utils.sdiag(self.sigma)

        if self.storeInnerProduct:
            if adjoint:
                return self.MeSigmaDerivMat.T * (Utils.sdiag(u)*v)
            else:
                return Utils.sdiag(u)*(self.MeSigmaDerivMat * v)
        else:
            dsigma_dlogsigma = Utils.sdiag(self.sigma)*self.etaDeriv
            if adjoint:
                return (
                    dsigma_dlogsigma.T * (
                        self.mesh.getEdgeInnerProductDeriv(self.sigma)(u).T * v
                    )
                )
            else:
                return (
                    self.mesh.getEdgeInnerProductDeriv(self.sigma)(u) *
                    (dsigma_dlogsigma * v)
                )


class Problem3D_CC(BaseSIPProblem):

    _solutionType = 'phiSolution'
    _formulation = 'HJ'  # CC potentials means J is on faces
    fieldsPair = Fields_CC
    sign = 1.
    bc_type = 'Neumann'

    def __init__(self, mesh, **kwargs):
        BaseSIPProblem.__init__(self, mesh, **kwargs)
        self.setBC()

        if self.storeJ:
            if self.actinds is None:
                print ("You did not put Active indices")
                print ("So, set actMap = IdentityMap(mesh)")
                self.actinds = np.ones(mesh.nC, dtype=bool)

            self.actMap = Maps.InjectActiveCells(mesh, self.actinds, 0.)

    def getA(self):
        """

        Make the A matrix for the cell centered DC resistivity problem

        A = D MfRhoI G

        """

        D = self.Div
        G = self.Grad
        # TODO: this won't work for full anisotropy
        MfRhoI = self.MfRhoI
        A = D * MfRhoI * G

        # I think we should deprecate this for DC problem.
        # if self._makeASymmetric is True:
        #     return V.T * A
        return A

    def getADeriv(self, u, v, adjoint=False):

        D = self.Div
        G = self.Grad
        MfRhoIDeriv = self.MfRhoIDeriv

        if adjoint:
            return self.MfRhoIDeriv(G * u, D.T * v, adjoint)

        return D * self.MfRhoIDeriv(G * u, v, adjoint)

    def getRHS(self):
        """
        RHS for the DC problem

        q
        """

        RHS = self.getSourceTerm()

        # I think we should deprecate this for DC problem.
        # if self._makeASymmetric is True:
        #     return self.Vol.T * RHS

        return RHS

    def getRHSDeriv(self, src, v, adjoint=False):
        """
        Derivative of the right hand side with respect to the model
        """
        # TODO: add qDeriv for RHS depending on m
        # qDeriv = src.evalDeriv(self, adjoint=adjoint)
        # return qDeriv
        return Utils.Zero()

    def setBC(self):
        if self.mesh._meshType == "TREE":
            if(self.bc_type == 'Neumann'):
                raise NotImplementedError()
            elif(self.bc_type == 'Dirchlet'):
                print('Homogeneous Dirchlet is the natural BC for this CC discretization.')
                self.Div = Utils.sdiag(self.mesh.vol) * self.mesh.faceDiv
                self.Grad = self.Div.T

        else:

            if self.mesh.dim == 3:
                fxm, fxp, fym, fyp, fzm, fzp = self.mesh.faceBoundaryInd
                gBFxm = self.mesh.gridFx[fxm, :]
                gBFxp = self.mesh.gridFx[fxp, :]
                gBFym = self.mesh.gridFy[fym, :]
                gBFyp = self.mesh.gridFy[fyp, :]
                gBFzm = self.mesh.gridFz[fzm, :]
                gBFzp = self.mesh.gridFz[fzp, :]

                # Setup Mixed B.C (alpha, beta, gamma)
                temp_xm = np.ones_like(gBFxm[:, 0])
                temp_xp = np.ones_like(gBFxp[:, 0])
                temp_ym = np.ones_like(gBFym[:, 1])
                temp_yp = np.ones_like(gBFyp[:, 1])
                temp_zm = np.ones_like(gBFzm[:, 2])
                temp_zp = np.ones_like(gBFzp[:, 2])

                if(self.bc_type == 'Neumann'):
                    if self.verbose:
                        print('Setting BC to Neumann.')
                    alpha_xm, alpha_xp = temp_xm*0., temp_xp*0.
                    alpha_ym, alpha_yp = temp_ym*0., temp_yp*0.
                    alpha_zm, alpha_zp = temp_zm*0., temp_zp*0.

                    beta_xm, beta_xp = temp_xm, temp_xp
                    beta_ym, beta_yp = temp_ym, temp_yp
                    beta_zm, beta_zp = temp_zm, temp_zp

                    gamma_xm, gamma_xp = temp_xm*0., temp_xp*0.
                    gamma_ym, gamma_yp = temp_ym*0., temp_yp*0.
                    gamma_zm, gamma_zp = temp_zm*0., temp_zp*0.

                elif(self.bc_type == 'Dirchlet'):
                    if self.verbose:
                        print('Setting BC to Dirchlet.')
                    alpha_xm, alpha_xp = temp_xm, temp_xp
                    alpha_ym, alpha_yp = temp_ym, temp_yp
                    alpha_zm, alpha_zp = temp_zm, temp_zp

                    beta_xm, beta_xp = temp_xm*1, temp_xp*1
                    beta_ym, beta_yp = temp_ym*1, temp_yp*1
                    beta_zm, beta_zp = temp_zm*1, temp_zp*1

                    gamma_xm, gamma_xp = temp_xm*0., temp_xp*0.
                    gamma_ym, gamma_yp = temp_ym*0., temp_yp*0.
                    gamma_zm, gamma_zp = temp_zm*0., temp_zp*0.

                elif(self.bc_type == 'Mixed'):
                    # Ztop: Neumann
                    # Others: Mixed: alpha * phi + d phi dn = 0
                    # where alpha = 1 / r  * dr/dn
                    # (Dey and Morrison, 1979)

                    # This assumes that the source is located at
                    # (x_center, y_center_y, ztop)
                    # TODO: Implement Zhang et al. (1995)

                    xs = np.median(self.mesh.vectorCCx)
                    ys = np.median(self.mesh.vectorCCy)
                    zs = self.mesh.vectorCCz[-1]

                    def r_boundary(x, y, z):
                        return 1./np.sqrt(
                            (x - xs)**2 + (y - ys)**2 + (z - zs)**2
                            )
                    rxm = r_boundary(gBFxm[:, 0], gBFxm[:, 1], gBFxm[:, 2])
                    rxp = r_boundary(gBFxp[:, 0], gBFxp[:, 1], gBFxp[:, 2])
                    rym = r_boundary(gBFym[:, 0], gBFym[:, 1], gBFym[:, 2])
                    ryp = r_boundary(gBFyp[:, 0], gBFyp[:, 1], gBFyp[:, 2])
                    rzm = r_boundary(gBFzm[:, 0], gBFzm[:, 1], gBFzm[:, 2])

                    alpha_xm = (gBFxm[:, 0]-xs)/rxm**2
                    alpha_xp = (gBFxp[:, 0]-xs)/rxp**2
                    alpha_ym = (gBFym[:, 1]-ys)/rym**2
                    alpha_yp = (gBFyp[:, 1]-ys)/ryp**2
                    alpha_zm = (gBFzm[:, 2]-zs)/rzm**2
                    alpha_zp = temp_zp.copy() * 0.

                    beta_xm, beta_xp = temp_xm*1, temp_xp*1
                    beta_ym, beta_yp = temp_ym*1, temp_yp*1
                    beta_zm, beta_zp = temp_zm*1, temp_zp*1

                    gamma_xm, gamma_xp = temp_xm*0., temp_xp*0.
                    gamma_ym, gamma_yp = temp_ym*0., temp_yp*0.
                    gamma_zm, gamma_zp = temp_zm*0., temp_zp*0.

                alpha = [
                    alpha_xm, alpha_xp,
                    alpha_ym, alpha_yp,
                    alpha_zm, alpha_zp
                ]
                beta = [beta_xm, beta_xp, beta_ym, beta_yp, beta_zm, beta_zp]
                gamma = [gamma_xm, gamma_xp, gamma_ym, gamma_yp, gamma_zm,
                         gamma_zp]

            elif self.mesh.dim == 2:

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

                alpha_xm, alpha_xp = temp_xm*0., temp_xp*0.
                alpha_ym, alpha_yp = temp_ym*0., temp_yp*0.

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
            self.Grad = self.Div.T - P_BC*Utils.sdiag(y_BC)*M


class Problem3D_N(BaseSIPProblem):

    _solutionType = 'phiSolution'
    _formulation = 'EB'  # N potentials means B is on faces
    fieldsPair = Fields_N
    sign = -1.

    def __init__(self, mesh, **kwargs):
        BaseSIPProblem.__init__(self, mesh, **kwargs)

        if self.storeJ:
            if self.actinds is None:
                print ("You did not put Active indices")
                print ("So, set actMap = IdentityMap(mesh)")
                self.actinds = np.ones(mesh.nC, dtype=bool)

            self.actMap = Maps.InjectActiveCells(mesh, self.actinds, 0.)

    def getA(self):
        """
            Make the A matrix for the cell centered DC resistivity problem

            A = G.T MeSigma G
        """

        # TODO: this won't work for full anisotropy
        MeSigma = self.MeSigma
        Grad = self.mesh.nodalGrad
        A = Grad.T * MeSigma * Grad

        # Handling Null space of A
        A[0, 0] = A[0, 0] + 1.

        return A

    def getADeriv(self, u, v, adjoint=False):
        """
            Product of the derivative of our system matrix with respect to
            the model and a vector
        """
        Grad = self.mesh.nodalGrad
        if not adjoint:
            return Grad.T*self.MeSigmaDeriv(Grad*u, v, adjoint)
        elif adjoint:
            return self.MeSigmaDeriv(Grad*u, Grad*v, adjoint)

    def getRHS(self):
        """
            RHS for the DC problem
        """

        RHS = self.getSourceTerm()
        return RHS

    def getRHSDeriv(self, src, v, adjoint=False):
        """
            Derivative of the right hand side with respect to the model
        """
        # TODO: add qDeriv for RHS depending on m
        # qDeriv = src.evalDeriv(self, adjoint=adjoint)
        # return qDeriv
        return Utils.Zero()
