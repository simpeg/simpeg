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

    def fields(self, m):
        self.curModel = m
        # Create a fields storage object
        F = FieldsTDEM(self.mesh, self.survey)
        for tx in self.survey.txList:
            # Set the initial conditions
            F[tx,:,0] = tx.getInitialFields(self.mesh)
        return self.forward(m, self.getRHS, self.calcFields, F=F)


    def forward(self, m, RHS, CalcFields, F=None):
        self.curModel = m
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
            F[:,:,tInd+1] = CalcFields(sol, tInd)
        return F

    def adjoint(self, m, RHS, CalcFields, F=None):
        self.curModel = m
        F = F or FieldsTDEM(self.mesh, self.survey)

        dtFact = None
        for tInd, dt in reversed(list(enumerate(self.timeSteps))):
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
            F[:,:,tInd+1] = CalcFields(sol, tInd)
        return F

    def Jvec(self, m, v, u=None):
        """
            :param numpy.array m: Conductivity model
            :param numpy.ndarray v: vector (model object)
            :param simpegEM.TDEM.FieldsTDEM u: Fields resulting from m
            :rtype: numpy.ndarray
            :return: w (data object)

            Multiplying \\\(\\\mathbf{J}\\\) onto a vector can be broken into three steps

            * Compute \\\(\\\\vec{p} = \\\mathbf{G}v\\\)
            * Solve \\\(\\\hat{\\\mathbf{A}} \\\\vec{y} = \\\\vec{p}\\\)
            * Compute \\\(\\\\vec{w} = -\\\mathbf{Q} \\\\vec{y}\\\)

        """
        self.curModel = m
        u = u or self.fields(m)
        p = self.Gvec(m, v, u)
        y = self.solveAh(m, p)
        Jv = self.survey.projectFieldsDeriv(u, v=y)
        return - mkvc(Jv)

    def Jtvec(self, m, v, u=None):
        """
            :param numpy.array m: Conductivity model
            :param numpy.ndarray,SimPEG.Survey.Data v: vector (data object)
            :param simpegEM.TDEM.FieldsTDEM u: Fields resulting from m
            :rtype: numpy.ndarray
            :return: w (model object)

            Multiplying \\\(\\\mathbf{J}^\\\\top\\\) onto a vector can be broken into three steps

            * Compute \\\(\\\\vec{p} = \\\mathbf{Q}^\\\\top \\\\vec{v}\\\)
            * Solve \\\(\\\hat{\\\mathbf{A}}^\\\\top \\\\vec{y} = \\\\vec{p}\\\)
            * Compute \\\(\\\\vec{w} = -\\\mathbf{G}^\\\\top y\\\)

        """
        self.curModel = m
        u = u or self.fields(m)

        if not isinstance(v, self.dataPair):
            v = self.dataPair(self.survey, v)

        p = self.survey.projectFieldsDeriv(u, v=v, adjoint=True)
        y = self.solveAht(m, p)
        w = self.Gtvec(m, y, u)
        return - mkvc(w)

