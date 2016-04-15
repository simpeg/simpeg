from SimPEG import Problem
from SimPEG.EM.Base import BaseEMProblem
from SurveyDC import Survey
from FieldsDC import Fields, Fields_CC

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
        raise NotImplementedError

    def Jtvec(self, m, v, f=None):
        raise NotImplementedError

    def getSourceTerm(self):
        """
        takes concept of source and turns it into a matrix
        """
        raise NotImplementedError



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


    def getRHS(self):
        """
        RHS for the DC problem

        q
        """

        RHS = self.getSourceTerm()
        if self._makeASymmetric is True:
            return self.Vol.T * RHS
        return RHS


