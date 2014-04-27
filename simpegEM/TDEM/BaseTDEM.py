from SimPEG import Solver
from SimPEG.Problem import BaseTimeProblem
from simpegEM.Utils import Sources
from SurveyTDEM import FieldsTDEM, SurveyTDEM
from scipy.constants import mu_0
from SimPEG.Utils import sdiag, mkvc
from SimPEG import Utils, Mesh
from simpegEM.Base import BaseEMProblem
import numpy as np


class BaseTDEMProblem(BaseTimeProblem, BaseEMProblem):
    """docstring for ProblemTDEM1D"""
    def __init__(self, mesh, mapping=None, **kwargs):
        BaseTimeProblem.__init__(self, mesh, mapping=mapping, **kwargs)

    surveyPair = SurveyTDEM

    def calcFields(self, sol, solType, tInd):

        if solType == 'b':
            b = sol
            e = self.MeSigmaI*self.mesh.edgeCurl.T*self.MfMui*b
            # Todo: implement non-zero js
        else:
            errStr = 'solType: ' + solType
            raise NotImplementedError(errStr)

        return {'b':b, 'e':e}

    def fields(self, m):
        self.curModel = m
        # Create a fields storage object
        F = FieldsTDEM(self.mesh, self.survey)
        for tx in self.survey.txList:
            # Set the initial conditions
            F[tx,:,0] = tx.getInitialFields(self.mesh)
        return self.forward(m, self.getRHS, self.calcFields, F=F)


    def forward(self, m, RHS, CalcFields, F=None):
        F = F or FieldsTDEM(self.mesh, self.survey)

        dtFact = None
        for tInd, dt in enumerate(self.timeSteps):
            if dt != dtFact:
                dtFact = dt
                A = self.getA(tInd)
                # print 'Factoring...   (dt = ' + str(dt) + ')'
                Asolve = self.Solver(A, **self.solverOpts)
                # print 'Done'
            rhs = RHS(tInd, F)
            sol = Asolve.solve(rhs)
            if sol.ndim == 1:
                sol.shape = (sol.size,1)
            F[:,:,tInd+1] = CalcFields(sol, self.solType, tInd)
        return F

    def adjoint(self, m, RHS, CalcFields, F=None):
        if F is None:
            F = FieldsTDEM(self.mesh, self.survey.nTx, self.nT, store=self.storeTheseFields)

        dtFact = None
        for tInd, dt in reversed(list(enumerate(self.timeSteps))):
            if dt != dtFact:
                dtFact = dt
                A = self.getA(tInd)
                # print 'Factoring...   (dt = ' + str(dt) + ')'
                Asolve = Solver(A, options=self.solverOpts)
                # print 'Done'
            rhs = RHS(tInd, F)
            sol = Asolve.solve(rhs)
            if sol.ndim == 1:
                sol.shape = (sol.size,1)
            newFields = CalcFields(sol, self.solType, tInd)
            F.update(newFields, tInd)
        return F

