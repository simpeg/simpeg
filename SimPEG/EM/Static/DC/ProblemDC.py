from SimPEG import Problem, Utils
from SimPEG.EM.Base import BaseEMProblem
from SurveyDC import Survey
from FieldsDC import Fields, Fields_CC, Fields_N
from SimPEG.Utils import sdiag
import numpy as np
from SimPEG.Utils import Zero

class BaseDCProblem(BaseEMProblem):

    surveyPair = Survey
    fieldsPair = Fields

    def fields(self, m):
        self.curModel = m
        f = self.fieldsPair(self.mesh, self.survey)
        A = self.getA()
        self.Ainv = self.Solver(A, **self.solverOpts)
        RHS = self.getRHS()
        u = self.Ainv * RHS
        Srcs = self.survey.srcList
        f[Srcs, self._solutionType] = u
        return f

    def Jvec(self, m, v, f=None):

        if f is None:
            f = self.fields(m)

        self.curModel = m

        Jv = self.dataPair(self.survey) #same size as the data

        A = self.getA()
        Ainv = self.Solver(A, **self.solverOpts)

        for src in self.survey.srcList:
            u_src = f[src, self._solutionType] # solution vector
            dA_dm_v = self.getADeriv(u_src, v)
            dRHS_dm_v = self.getRHSDeriv(src, v)
            du_dm_v = Ainv * ( - dA_dm_v + dRHS_dm_v )

            for rx in src.rxList:
                df_dmFun = getattr(f, '_%sDeriv'%rx.projField, None)
                df_dm_v = df_dmFun(src, du_dm_v, v, adjoint=False)
                Jv[src, rx] = rx.evalDeriv(src, self.mesh, f, df_dm_v)
        return Utils.mkvc(Jv)

    def Jtvec(self, m, v, f=None):
        if f is None:
            f = self.fields(m)

        self.curModel = m

        # Ensure v is a data object.
        if not isinstance(v, self.dataPair):
            v = self.dataPair(self.survey, v)

        Jtv = np.zeros(m.size)
        AT = self.getA().T
        ATinv = self.Solver(AT, **self.solverOpts)

        for src in self.survey.srcList:
            u_src = f[src, self._solutionType]
            for rx in src.rxList:
                PTv = rx.evalDeriv(src, self.mesh, f, v[src, rx], adjoint=True) # wrt f, need possibility wrt m
                df_duTFun = getattr(f, '_%sDeriv'%rx.projField, None)
                df_duT, df_dmT = df_duTFun(src, None, PTv, adjoint=True)

                ATinvdf_duT = ATinv * df_duT

                dA_dmT = self.getADeriv(u_src, ATinvdf_duT, adjoint=True)
                dRHS_dmT = self.getRHSDeriv(src, ATinvdf_duT, adjoint=True)
                du_dmT = -dA_dmT + dRHS_dmT
                Jtv += df_dmT + du_dmT

        return Utils.mkvc(Jtv)

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

        if self._formulation is 'EB':
            n = self.mesh.nN
            # return NotImplementedError

        elif self._formulation is 'HJ':
            n = self.mesh.nC

        q = np.zeros((n, len(Srcs)))

        for i, src in enumerate(Srcs):
            q[:,i] = src.eval(self)
        return q

class Problem3D_N(BaseDCProblem):

    _solutionType = 'phiSolution'
    _formulation  = 'EB' # N potentials means B is on faces
    fieldsPair    = Fields_N

    def __init__(self, mesh, **kwargs):
        BaseDCProblem.__init__(self, mesh, **kwargs)

    def getA(self):
        """

        Make the A matrix for the cell centered DC resistivity problem

        A = D MfRhoI D^\\top V

        """

        # TODO: this won't work for full anisotropy
        MeSigma = self.MeSigma
        Grad = self.mesh.nodalGrad
        A = Grad.T * MeSigma * Grad
        # if self._makeASymmetric is True:
        #     return V.T * A
        return A

    def getADeriv(self, u, v, adoint=False):
        """

        Product of the derivative of our system matrix with respect to the model and a vector

        """
        return Div*self.MfRhoIDeriv(Div.T*u)


    def getRHS(self):
        """
        RHS for the DC problem

        q
        """

        RHS = self.getSourceTerm()
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

class Problem3D_CC(BaseDCProblem):

    _solutionType = 'phiSolution'
    _formulation  = 'HJ' # CC potentials means J is on faces
    fieldsPair    = Fields_CC

    def __init__(self, mesh, **kwargs):
        BaseDCProblem.__init__(self, mesh, **kwargs)

    def getA(self):
        """

        Make the A matrix for the cell centered DC resistivity problem

        A = D MfRhoI D^\\top V

        """

        V = self.Vol
        D = V * self.mesh.faceDiv
        # TODO: this won't work for full anisotropy
        MfRhoI = self.MfRhoI
        A = D * MfRhoI * D.T

        # I think we should deprecate this for DC problem.
        # if self._makeASymmetric is True:
        #     return V.T * A
        return A

    def getADeriv(self, u, v, adjoint= False):

        V = self.Vol
        D = V * self.mesh.faceDiv
        MfRhoIDeriv = self.MfRhoIDeriv

        if adjoint:
            # if self._makeASymmetric is True:
            #     v = V * v
            return( MfRhoIDeriv( D.T * u ).T) * ( D.T * v)

        # I think we should deprecate this for DC problem.
        # if self._makeASymmetric is True:
        #     return V.T * ( D * ( MfRhoIDeriv( D.T * ( V * u ) ) * v ) )
        return D * (MfRhoIDeriv( D.T * u ) * v)

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



