from SimPEG import Problem
from SimPEG import Utils
from SimPEG import Mesh
from SimPEG import Survey
from SimPEG import Regularization
from SimPEG import DataMisfit
from SimPEG import Optimization
from SimPEG import InvProblem
from SimPEG import Directives
from SimPEG import Inversion

import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt


def lengthInCell(O, D, x, y, plotIt=False):

    maxD = np.sqrt(np.sum(D**2))
    D = D/maxD

    def dist(a):
        return O + a*D

    if plotIt:
        plt.plot(x[[0, 0, 1, 1, 0]], y[[0, 1, 1, 0, 0]], 'b')
        plt.plot(O[0], O[1], 'rs')
        d = np.r_[0, maxD]
        plt.plot(O[0] + d*D[0], O[1] + d*D[1], 'k-')

    alp = np.array([-np.inf, np.inf, -np.inf, np.inf])

    if np.abs(D[0]) > 0:
        alp[0] = (x[0] - O[0])/D[0]
        alp[1] = (x[1] - O[0])/D[0]
        if plotIt:
            plt.plot(dist(alp[0])[0], dist(alp[0])[1], 'mo')
            plt.plot(dist(alp[1])[0], dist(alp[1])[1], 'mo')

    if np.abs(D[1]) > 0:
        alp[2] = (y[0] - O[1])/D[1]
        alp[3] = (y[1] - O[1])/D[1]
        if plotIt:
            plt.plot(dist(alp[2])[0], dist(alp[2])[1], 'go')
            plt.plot(dist(alp[3])[0], dist(alp[3])[1], 'go')

    midAlp = np.array(sorted(alp)[1:3])
    midAlp[midAlp < 0] = 0
    midAlp[midAlp > maxD] = maxD
    midPoint = dist(np.mean(midAlp))

#     print alp, midAlp, midPoint

    if midPoint[0] >= x[0] and midPoint[0] <= x[1] and midPoint[1] >= y[0] and midPoint[1] <= y[1]:
        vec = dist(midAlp[0]) - dist(midAlp[1])
        if plotIt:
            c = np.c_[dist(midAlp[0]), dist(midAlp[1])]
            plt.plot(c[0, :], c[1, :], 'r', lw=2)
        return np.sqrt(np.sum(vec**2))

    return None


def lineintegral(M, Tx, Rx):
    O, D = Tx, Rx - Tx
    I, J, V = [], [], []
    for i in range(M.nCx):
        for j in range(M.nCy):
            x = M.vectorNx[[i, i+1]]
            y = M.vectorNy[[j, j+1]]
            v = lengthInCell(O, D, x, y)
            if v is not None:
                I += [i]
                J += [j]
                V += [v]
    inds = Utils.sub2ind(M.vnC, np.array([I, J]).T)
    return inds, V


class StraightRaySurvey(Survey.LinearSurvey):
    def __init__(self, txList):
        self.txList = txList

    @property
    def nD(self):
        n = 0
        for tx in self.txList:
            n += np.sum([rx.nD for rx in tx.rxList])
        return n

    def projectFields(self, u):
        return u

    def plot(self, ax=None):
        if ax is None:
            ax = plt.subplot(111)
        for tx in self.txList:
            ax.plot(tx.loc[0], tx.loc[1], 'ws', ms=8)

            for rx in tx.rxList:
                for loc_i in range(rx.locs.shape[0]):
                    ax.plot(rx.locs[loc_i, 0],rx.locs[loc_i, 1], 'w^', ms=8)
                    ax.plot(np.r_[tx.loc[0], rx.locs[loc_i, 0]], np.r_[tx.loc[1], rx.locs[loc_i, 1]], 'w-', lw=0.5, alpha=0.8)



class StraightRayProblem(Problem.LinearProblem):

    @property
    def A(self):
        if getattr(self, '_A', None) is not None:
            return self._A

        self._A = sp.lil_matrix((self.survey.nD, self.mesh.nC))
        row = 0
        for tx in self.survey.txList:
            for rx in tx.rxList:
                for loc_i in range(rx.locs.shape[0]):
                    inds, V = lineintegral(self.mesh, tx.loc, rx.locs[loc_i, :])
                    self._A[inds*0+row, inds] = V
                    row += 1

        return self._A

    def fields(self, m):
        self.model = m
        # logsig = self.model.transform
        # return self.A * logsig
        return self.A * self.model

    def Jvec(self, m, v, f=None):
        self.model = m
        # mt = self.model.transformDeriv
        # return self.A * ( mt * v )
        return self.A * v

    def Jtvec(self, m, v, f=None):
        self.model = m
        # mt = self.model.transformDeriv
        # return mt.T * ( self.A.T * v )
        return self.A.T * v


if __name__ == '__main__':
    O = np.r_[-1.2, -1.]
    D = np.r_[10., 10.]
    x = np.r_[0., 1.]
    y = np.r_[0., 1.]
    print 'length:', lengthInCell(O, D, x, y, plotIt=True)
    O = np.r_[0, -1.]
    D = np.r_[1., 1.]*1.5
    print 'length:', lengthInCell(O, D, x*2, y*2, plotIt=True)

    nC = 20
    M = Mesh.TensorMesh([nC, nC])
    y = np.linspace(0., 1., nC/2)
    rlocs = np.c_[y*0+M.vectorCCx[-1], y]
    rx = Survey.BaseRx(rlocs, None)

    srcList = [
        Survey.BaseSrc(loc=np.r_[M.vectorCCx[0], yi], rxList=[rx])
        for yi in y
    ]

    survey = StraightRaySurvey(srcList)
    problem = StraightRayProblem(M)
    problem.pair(survey)

    s = Utils.mkvc(Utils.ModelBuilder.randomModel(M.vnC)) + 1.
    survey.dobs = survey.dpred(s)
    survey.std = 0.01
    plt.plot(survey.dobs)
    plt.plot(survey.dpred(s*0+1.5))
    ax = plt.subplot(111)
    M.plotImage(s, ax=ax)
    survey.plot(ax=ax)

    # Create an optimization program
    opt = Optimization.InexactGaussNewton(maxIter=10)
    opt.remember('xc')
    # Create a regularization program
    # regModel = Model.BaseModel(M)

    reg = Regularization.Tikhonov(M)
    dmis = DataMisfit.l2_DataMisfit(survey)
    opt = Optimization.InexactGaussNewton(maxIter=40)
    invProb = InvProblem.BaseInvProblem(dmis, reg, opt)
    beta = Directives.BetaSchedule()
    betaest = Directives.BetaEstimate_ByEig()
    inv = Inversion.BaseInversion(invProb, directiveList=[beta, betaest])

    # Start the inversion with a model of zeros, and run the inversion
    m0 = np.ones(M.nC)*1.5
    mopt = inv.run(m0)

    plt.colorbar(M.plotImage(mopt)[0])
    plt.colorbar(M.plotImage(s)[0])

    plt.show()
