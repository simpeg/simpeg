from SimPEG import Solver, Problem
from SimPEG.Problem import BaseTimeProblem
from SimPEG.EM.Utils import *
from scipy.constants import mu_0
from SimPEG.Utils import sdiag, mkvc
from SimPEG import Utils, Mesh
from SimPEG.EM.Base import BaseEMProblem
import numpy as np


class FieldsTDEM(Problem.TimeFields):
    """Fancy Field Storage for a TDEM survey."""
    knownFields = {'b': 'F', 'e': 'E'}

    def tovec(self):
        nSrc, nF, nE = self.survey.nSrc, self.mesh.nF, self.mesh.nE
        u = np.empty((0,nSrc)) #((0,1) if nSrc == 1 else (0, nSrc))

        for i in range(self.survey.prob.nT):
            if 'b' in self:
                b = self[:,'b',i+1]
            else:
                b = np.zeros((nF,nSrc)) # if nSrc == 1 else (nF, nSrc))

            if 'e' in self:
                e = self[:,'e',i+1]
            else:
                e = np.zeros((nE,nSrc)) # if nSrc == 1 else (nE, nSrc))
            u = np.concatenate((u, b, e))

        return Utils.mkvc(u,nSrc)


class BaseTDEMProblem(BaseTimeProblem, BaseEMProblem):
    """docstring for BaseTDEMProblem"""
    def __init__(self, mesh, mapping=None, **kwargs):
        BaseTimeProblem.__init__(self, mesh, mapping=mapping, **kwargs)

    _FieldsForward_pair = FieldsTDEM  #: used for the forward calculation only

    waveformType = "STEPOFF"
    current = None

    def currentwaveform(self, wave):
        self._timeSteps = np.diff(wave[:,0])
        self.current = wave[:,1]
        self.waveformType = "GENERAL"

    def fields(self, m):
        if self.verbose: print '{0!s}\nCalculating fields(m)\n{1!s}'.format('*'*50, '*'*50)
        self.curModel = m
        # Create a fields storage object
        F = self._FieldsForward_pair(self.mesh, self.survey)
        for src in self.survey.srcList:
            # Set the initial conditions
            F[src,:,0] = src.getInitialFields(self.mesh)
        F = self.forward(m, self.getRHS, F=F)
        if self.verbose: print '{0!s}\nDone calculating fields(m)\n{1!s}'.format('*'*50, '*'*50)
        return F

    def forward(self, m, RHS, F=None):
        self.curModel = m
        F = F or FieldsTDEM(self.mesh, self.survey)

        dtFact = None
        Ainv   = None
        for tInd, dt in enumerate(self.timeSteps):
            if dt != dtFact:
                dtFact = dt
                if Ainv is not None:
                    Ainv.clean()
                A = self.getA(tInd)
                if self.verbose: print 'Factoring...   (dt = {0:e})'.format(dt)
                Ainv = self.Solver(A, **self.solverOpts)
                if self.verbose: print 'Done'
            rhs = RHS(tInd, F)
            if self.verbose: print '    Solving...   (tInd = {0:d})'.format(tInd)
            sol = Ainv * rhs
            if self.verbose: print '    Done...'
            if sol.ndim == 1:
                sol.shape = (sol.size,1)
            F[:,self.solType,tInd+1] = sol
        Ainv.clean()
        return F

    def adjoint(self, m, RHS, F=None):
        self.curModel = m
        F = F or FieldsTDEM(self.mesh, self.survey)

        dtFact = None
        Ainv   = None
        for tInd, dt in reversed(list(enumerate(self.timeSteps))):
            if dt != dtFact:
                dtFact = dt
                if Ainv is not None:
                    Ainv.clean()
                A = self.getA(tInd)
                if self.verbose: print 'Factoring (Adjoint)...   (dt = {0:e})'.format(dt)
                Ainv = self.Solver(A, **self.solverOpts)
                if self.verbose: print 'Done'
            rhs = RHS(tInd, F)
            if self.verbose: print '    Solving (Adjoint)...   (tInd = {0:d})'.format(tInd)
            sol = Ainv * rhs
            if self.verbose: print '    Done...'
            if sol.ndim == 1:
                sol.shape = (sol.size,1)
            F[:,self.solType,tInd+1] = sol
        Ainv.clean()
        return F

    def Jvec(self, m, v, f=None):
        """
            :param numpy.array m: Conductivity model
            :param numpy.ndarray v: vector (model object)
            :param FieldsTDEM f: Fields resulting from m
            :rtype: numpy.ndarray
            :return: w (data object)

            Multiplying \\\(\\\mathbf{J}\\\) onto a vector can be broken into three steps

            * Compute \\\(\\\\vec{p} = \\\mathbf{G}v\\\)
            * Solve \\\(\\\hat{\\\mathbf{A}} \\\\vec{y} = \\\\vec{p}\\\)
            * Compute \\\(\\\\vec{w} = -\\\mathbf{Q} \\\\vec{y}\\\)

        """
        if self.verbose: print '{0!s}\nCalculating J(v)\n{1!s}'.format('*'*50, '*'*50)
        self.curModel = m
        if f is None:
            f = self.fields(m)
        p = self.Gvec(m, v, f)
        y = self.solveAh(m, p)
        Jv = self.survey.evalDeriv(f, v=y)
        if self.verbose: print '{0!s}\nDone calculating J(v)\n{1!s}'.format('*'*50, '*'*50)
        return - mkvc(Jv)

    def Jtvec(self, m, v, f=None):
        """
            :param numpy.array m: Conductivity model
            :param numpy.ndarray v: vector (or a :class:`SimPEG.Survey.Data` object)
            :param FieldsTDEM u: Fields resulting from m
            :rtype: numpy.ndarray
            :return: w (model object)

            Multiplying \\\(\\\mathbf{J}^\\\\top\\\) onto a vector can be broken into three steps

            * Compute \\\(\\\\vec{p} = \\\mathbf{Q}^\\\\top \\\\vec{v}\\\)
            * Solve \\\(\\\hat{\\\mathbf{A}}^\\\\top \\\\vec{y} = \\\\vec{p}\\\)
            * Compute \\\(\\\\vec{w} = -\\\mathbf{G}^\\\\top y\\\)

        """
        if self.verbose: print '{0!s}\nCalculating J^T(v)\n{1!s}'.format('*'*50, '*'*50)
        self.curModel = m
        if f is None:
            f = self.fields(m)

        if not isinstance(v, self.dataPair):
            v = self.dataPair(self.survey, v)

        p = self.survey.evalDeriv(f, v=v, adjoint=True)
        y = self.solveAht(m, p)
        w = self.Gtvec(m, y, f)
        if self.verbose: print '{0!s}\nDone calculating J^T(v)\n{1!s}'.format('*'*50, '*'*50)
        return - mkvc(w)

