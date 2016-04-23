from SimPEG import Problem
from SimPEG.EM.Base import BaseEMProblem
from SurveyDC import Survey
from FieldsDC import Fields, Fields_CC
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
            print type(dA_dm_v + dRHS_dm_v), (dA_dm_v + dRHS_dm_v).shape
            du_dm_v = Ainv * ( - dA_dm_v + dRHS_dm_v )

            for rx in src.rxList:
                df_dmFun = getattr(f, '_%sDeriv'%rx.projField, None)
                df_dm_v = df_dmFun(src, du_dm_v, v, adjoint=False)
                Jv[src, rx] = rx.evalDeriv(src, self.mesh, f, df_dm_v)
            Ainv.clean()
        return Utils.mkvc(Jv)

    def Jtvec(self, m, v, f=None):
        raise NotImplementedError

    def getSourceTerm(self):
        """
        takes concept of source and turns it into a matrix
        """
        """
        Evaluates the sources for a given frequency and puts them in matrix form

        :param float freq: Frequency
        :rtype: (numpy.ndarray, numpy.ndarray)
        :return: s_m, s_e (nE or nF, nSrc)
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

        # TODO: this won't work for full anisotropy

        D = self.mesh.faceDiv
        MfRhoI = self.MfRhoI
        V = self.Vol
        A = D * ( MfRhoI * ( D.T * V ) )

        if self._makeASymmetric is True:
            return V.T * A
        return A

    def getADeriv(self, u, v, adjoint= False):

        D = self.mesh.faceDiv
        MfRhoIDeriv = self.MfRhoIDeriv
        V = self.Vol

        if adjoint:
            if self._makeASymmetric is True:
                v = V * v
            return V.T * ( D * ( MfRhoIDeriv(D * v) ) )

        if self._makeASymmetric is True:
            return V.T * ( D * ( MfRhoIDeriv( * D.T * ( V * u ) ) * v ) )
        return D * ( MfRhoIDeriv( D.T * ( V * v ) ) )

    def getRHS(self):
        """
        RHS for the DC problem

        q
        """

        RHS = self.getSourceTerm()
        if self._makeASymmetric is True:
            return self.Vol.T * RHS
        return RHS

    def getRHSDeriv(self, src, v, adjoint=False):
        return Zero()



