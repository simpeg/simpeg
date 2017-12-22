from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from SimPEG import Utils
from SimPEG.EM.Base import BaseEMProblem
from SimPEG.EM.Static.DC.FieldsDC import FieldsDC, Fields_CC, Fields_N
import numpy as np
from SimPEG.Utils import Zero
from SimPEG.EM.Static.DC import getxBCyBC_CC
from .SurveyIP import Survey
from SimPEG import Props
import sys


class BaseIPProblem(BaseEMProblem):

    sigma = Props.PhysicalProperty(
        "Electrical conductivity (S/m)"
    )

    rho = Props.PhysicalProperty(
        "Electrical resistivity (Ohm m)"
    )

    Props.Reciprocal(sigma, rho)

    eta, etaMap, etaDeriv = Props.Invertible(
        "Electrical Chargeability"
    )

    surveyPair = Survey
    fieldsPair = FieldsDC
    Ainv = None
    _f = None
    storeJ = False
    _Jmatrix = None
    sign = None

    def fields(self, m):
        if self.verbose is True:
            print (">> Compute fields")

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
            self._Jmatrix = (self._Jtvec(m, v=None, f=f)).T

            # delete fields after computing sensitivity
            del f
            if self._f is not None:
                del self._f
            # clean all factorization
            if self.Ainv is not None:
                self.Ainv.clean()

        return self._Jmatrix

    def Jvec(self, m, v, f=None):

        self.model = m

        # When sensitivity matrix J is stored
        if self.storeJ:
            J = self.getJ(m, f=f)
            Jv = Utils.mkvc(np.dot(J, v))
            return self.sign * Jv

        else:

            if f is None:
                f = self.fields(m)

            Jv = []

            for src in self.survey.srcList:
                u_src = f[src, self._solutionType] # solution vector
                dA_dm_v = self.getADeriv(u_src, v)
                dRHS_dm_v = self.getRHSDeriv(src, v)
                du_dm_v = self.Ainv * ( - dA_dm_v + dRHS_dm_v )

                for rx in src.rxList:
                    df_dmFun = getattr(f, '_{0!s}Deriv'.format(rx.projField), None)
                    df_dm_v = df_dmFun(src, du_dm_v, v, adjoint=False)
                    Jv.append(rx.evalDeriv(src, self.mesh, f, df_dm_v))

            # Conductivity (d u / d log sigma) - EB form
            # Resistivity (d u / d log rho) - HJ form
            return self.sign*np.hstack(Jv)

    def Jtvec(self, m, v, f=None):
        """
            Compute adjoint sensitivity matrix (J^T) and vector (v) product.

        """

        # When sensitivity matrix J is stored
        if self.storeJ:
            J = self.getJ(m, f=f)
            Jtv = Utils.mkvc(np.dot(J.T, v))
            return self.sign * Jtv

        else:
            self.model = m

            if f is None:
                f = self.fields(m)
            return self._Jtvec(m, v=v, f=f)

    def _Jtvec(self, m, v=None, f=None):
        """
            Compute adjoint sensitivity matrix (J^T) and vector (v) product.
            Full J matrix can be computed by inputing v=None
        """

        if v is not None:
            # Ensure v is a data object.
            if not isinstance(v, self.dataPair):
                v = self.dataPair(self.survey, v)
            Jtv = np.zeros(m.size)
        else:
            # This is for forming full sensitivity matrix
            Jtv = []

        for isrc, src in enumerate(self.survey.srcList):
            u_src = f[src, self._solutionType]
            if self.storeJ:
                # TODO: use logging package
                sys.stdout.write(("\r %d / %d") % (isrc+1, self.survey.nSrc))
                sys.stdout.flush()

            for rx in src.rxList:
                if v is not None:
                    PTv = rx.evalDeriv(
                        src, self.mesh, f, v[src, rx], adjoint=True
                    )  # wrt f, need possibility wrt m
                    df_duTFun = getattr(
                        f, '_{0!s}Deriv'.format(rx.projField), None
                    )
                    df_duT, df_dmT = df_duTFun(src, None, PTv, adjoint=True)
                    ATinvdf_duT = self.Ainv * df_duT
                    dA_dmT = self.getADeriv(u_src, ATinvdf_duT, adjoint=True)
                    dRHS_dmT = self.getRHSDeriv(src, ATinvdf_duT, adjoint=True)
                    du_dmT = -dA_dmT + dRHS_dmT
                    Jtv += (df_dmT + du_dmT).astype(float)
                else:
                    P = rx.getP(self.mesh, rx.projGLoc(f)).toarray()
                    ATinvdf_duT = self.Ainv * (P.T)
                    dA_dmT = self.getADeriv(u_src, ATinvdf_duT, adjoint=True)
                    if rx.nD == 1:
                        # Consider when rx has one location
                        Jtv.append(-dA_dmT.reshape([-1, 1]))
                    else:
                        Jtv.append(-dA_dmT)

        # Conductivity ((d u / d log sigma).T) - EB form
        # Resistivity ((d u / d log rho).T) - HJ form

        if v is not None:
            return self.sign*Utils.mkvc(Jtv)
        else:
            return self.sign*np.hstack(Jtv)
        return

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

    def delete_these_for_sensitivity(self):
        del self._Jmatrix, self._MfRhoI, self._MeSigma

    @property
    def deleteTheseOnModelUpdate(self):
        toDelete = []
        return toDelete

    # assume log rho or log cond
    @property
    def MeSigma(self):
        """
            Edge inner product matrix for \\(\\sigma\\).
            Used in the E-B formulation
        """
        if getattr(self, '_MeSigma', None) is None:
            self._MeSigma = self.mesh.getEdgeInnerProduct(self.sigma)
        return self._MeSigma

    @property
    def MfRhoI(self):
        """
            Inverse of :code:`MfRho`
        """
        if getattr(self, '_MfRhoI', None) is None:
            self._MfRhoI = self.mesh.getFaceInnerProduct(self.rho, invMat=True)
        return self._MfRhoI

    def MfRhoIDeriv(self, u):
        """
            Derivative of :code:`MfRhoI` with respect to the model.
        """

        dMfRhoI_dI = -self.MfRhoI**2
        dMf_drho = self.mesh.getFaceInnerProductDeriv(self.rho)(u)
        drho_dlogrho = Utils.sdiag(self.rho)*self.etaDeriv
        return dMfRhoI_dI * (dMf_drho * drho_dlogrho)

    # TODO: This should take a vector
    def MeSigmaDeriv(self, u):
        """
            Derivative of MeSigma with respect to the model
        """
        dsigma_dlogsigma = Utils.sdiag(self.sigma)*self.etaDeriv
        MeSigmaDeriv = (
            self.mesh.getEdgeInnerProductDeriv(self.sigma)(u) *
            dsigma_dlogsigma
        )
        return MeSigmaDeriv


class Problem3D_CC(BaseIPProblem):

    _solutionType = 'phiSolution'
    _formulation = 'HJ'  # CC potentials means J is on faces
    fieldsPair = Fields_CC
    sign = 1.
    bc_type = 'Neumann'

    def __init__(self, mesh, **kwargs):
        BaseIPProblem.__init__(self, mesh, **kwargs)
        self.setBC()

    def getA(self):
        """

        Make the A matrix for the cell centered DC resistivity problem

        A = D MfRhoI G

        """

        D = self.Div
        G = self.Grad
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
            # if self._makeASymmetric is True:
            #     v = V * v
            return (MfRhoIDeriv(G * u).T) * (D.T * v)

        # I think we should deprecate this for DC problem.
        # if self._makeASymmetric is True:
        #     return V.T * ( D * ( MfRhoIDeriv( D.T * ( V * u ) ) * v ) )
        return D * (MfRhoIDeriv(G * u) * v)

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
        return Zero()

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


class Problem3D_N(BaseIPProblem):

    _solutionType = 'phiSolution'
    _formulation = 'EB'  # N potentials means B is on faces
    fieldsPair = Fields_N
    sign = -1.

    def __init__(self, mesh, **kwargs):
        BaseIPProblem.__init__(self, mesh, **kwargs)

    def getA(self):
        """

        Make the A matrix for the cell centered DC resistivity problem

        A = G.T MeSigma G

        """

        MeSigma = self.MeSigma
        Grad = self.mesh.nodalGrad
        A = Grad.T * MeSigma * Grad

        # Handling Null space of A
        A[0, 0] = A[0, 0] + 1.

        return A

    def getADeriv(self, u, v, adjoint=False):
        """
            Product of the derivative of our system matrix with
            respect to the model and a vector
        """
        Grad = self.mesh.nodalGrad
        if not adjoint:
            return Grad.T*(self.MeSigmaDeriv(Grad*u)*v)
        elif adjoint:
            return self.MeSigmaDeriv(Grad*u).T * (Grad*v)

    def getRHS(self):
        """
        RHS for the DC problem

        q
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
        return Zero()


