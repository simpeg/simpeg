import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

from SimPEG import Problem
from SimPEG import Utils
from SimPEG import Mesh
from SimPEG import Props


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


class StraightRayProblem(Problem.LinearProblem):

    slowness, slownessMap, slownessDeriv = Props.Invertible(
        "Slowness model (1/v)"
    )

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
        return self.A * self.slowness

    def Jvec(self, m, v, f=None):
        self.model = m
        # mt = self.model.transformDeriv
        # return self.A * ( mt * v )
        return self.A * self.slownessDeriv * v

    def Jtvec(self, m, v, f=None):
        self.model = m
        # mt = self.model.transformDeriv
        # return mt.T * ( self.A.T * v )
        return self.slownessDeriv.T * self.A.T * v
