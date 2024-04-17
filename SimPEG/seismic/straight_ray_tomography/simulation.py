import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

from ...simulation import LinearSimulation
from ...utils import sub2ind
from ... import props


def _lengthInCell(O, D, xyz, plotIt=False):
    maxD = np.sqrt(np.sum(D**2))
    D = D / maxD

    def dist(a):
        # a : the distance along the line
        # D : Direction vector of the line
        return O + a * D

    if len(xyz) == 2:  # The model is 2D
        x = xyz[0]
        y = xyz[1]
        if plotIt:
            plt.plot(x[[0, 0, 1, 1, 0]], y[[0, 1, 1, 0, 0]], "b")
            plt.plot(O[0], O[1], "rs")
            d = np.r_[0, maxD]
            plt.plot(O[0] + d * D[0], O[1] + d * D[1], "k-")

        alp = np.array([0, maxD], [-np.inf, np.inf], [-np.inf, np.inf])
        # alp contains all the possible intersection points along the ray with the cell
        if np.abs(D[0]) > 0:
            alp[1, 0] = (x[0] - O[0]) / D[0]
            alp[1, 1] = (x[1] - O[0]) / D[0]
            if plotIt:
                plt.plot(dist(alp[1, 0])[0], dist(alp[1, 0])[1], "mo")
                plt.plot(dist(alp[1, 1])[0], dist(alp[1, 1])[1], "mo")

        if np.abs(D[1]) > 0:
            alp[2, 0] = (y[0] - O[1]) / D[1]
            alp[2, 1] = (y[1] - O[1]) / D[1]
            if plotIt:
                plt.plot(dist(alp[2, 0])[0], dist(alp[2, 0])[1], "go")
                plt.plot(dist(alp[2, 1])[0], dist(alp[2, 1])[1], "go")

        midAlp = np.array([max(alp[:, 0]), min(alp[:, 1])])
        midPoint = dist(np.mean(midAlp))

        #     print alp, midAlp, midPoint

        if (
                x[0] <= midPoint[0] <= x[1]
                and y[0] <= midPoint[1] <= y[1]
        ):
            vec = dist(midAlp[0]) - dist(midAlp[1])
            if plotIt:
                c = np.c_[dist(midAlp[0]), dist(midAlp[1])]
                plt.plot(c[0, :], c[1, :], "r", lw=2)
            return np.sqrt(np.sum(vec ** 2))

    elif len(xyz) == 3: # The model is 3D
        x = xyz[0]
        y = xyz[1]
        z = xyz[2]
        if plotIt:
            plt.plot(x[[0, 0, 1, 1, 0]], y[[0, 1, 1, 0, 0]], z[[0, 0, 0, 0, 0]], "b")
            plt.plot(x[[0, 0, 1, 1, 0]], y[[0, 1, 1, 0, 0]], z[[1, 1, 1, 1, 1]], "b")
            plt.plot(x[[0, 0, 1, 1, 0]], y[[0, 0, 0, 0, 0]], z[[0, 1, 1, 0, 0]], "b")
            plt.plot(x[[0, 0, 1, 1, 0]], y[[1, 1, 1, 1, 1]], z[[0, 1, 1, 0, 0]], "b")
            plt.plot(O[0], O[1], O[2], "rs")
            d = np.r_[0, maxD]
            plt.plot(O[0] + d * D[0], O[1] + d * D[1], O[2] + d * D[2], "k-")

        alp = np.array([0, maxD], [-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf])

        if np.abs(D[0]) > 0:
            alp[1, 0] = (x[0] - O[0]) / D[0]
            alp[1, 1] = (x[1] - O[0]) / D[0]
            if plotIt:
                plt.plot(dist(alp[1, 0])[0], dist(alp[1, 0])[1], dist(alp[1, 0])[2], "mo")
                plt.plot(dist(alp[1, 1])[0], dist(alp[1, 1])[1], dist(alp[1, 1])[2], "mo")

        if np.abs(D[1]) > 0:
            alp[2, 0] = (y[0] - O[1]) / D[1]
            alp[2, 1] = (y[1] - O[1]) / D[1]
            if plotIt:
                plt.plot(dist(alp[2, 0])[0], dist(alp[2, 0])[1], dist(alp[2, 0])[2], "go")
                plt.plot(dist(alp[2, 1])[0], dist(alp[2, 1])[1], dist(alp[2, 1])[2], "go")

        if np.abs(D[2]) > 0:
            alp[3, 0] = (z[0] - O[0]) / D[2]
            alp[3, 1] = (z[1] - O[0]) / D[2]
            if plotIt:
                plt.plot(dist(alp[3, 0])[0], dist(alp[3, 0])[1], dist(alp[3, 0])[2], "mo")
                plt.plot(dist(alp[3, 1])[0], dist(alp[3, 1])[1], dist(alp[3, 1])[2], "mo")

        midAlp = np.array([max(alp[:, 0]), min(alp[:, 1])])
        midPoint = dist(np.mean(midAlp))
        #     print alp, midAlp, midPoint

        if (
                x[0] <= midPoint[0] <= x[1]
                and y[0] <= midPoint[1] <= y[1]
                and z[0] <= midPoint[2] <= z[1]
        ):
            vec = dist(midAlp[0]) - dist(midAlp[1])
            if plotIt:
                c = np.c_[dist(midAlp[0]), dist(midAlp[1])]
                plt.plot(c[0, :], c[1, :], c[2, :], "r--", lw=2)
            return np.sqrt(np.sum(vec ** 2))

    return None


def _lineintegral(M, Tx, Rx):
    """
    :parameter
    M: the mesh
    Tx: Source location which is the array of its location
    Rx: Receiver location which is the array of its location

    Symbols:
    -------------------------
    dim: the dimension of simulation
    O : stands for the origin which the location of source
    D : stands for the direction vector. The direction from the source to the receiver
    """
    dim = M.dim
    O, D = Tx, Rx - Tx
    if dim == 2:
        I, J, V = [], [], []
        for i in range(M.shape_cells[0]):
            for j in range(M.shape_cells[1]):
                x = M.nodes_x[[i, i + 1]]
                y = M.nodes_y[[j, j + 1]]
                v = _lengthInCell(O, D, [x, y])
                if v is not None:
                    I += [i]
                    J += [j]
                    V += [v]
        inds = sub2ind(M.vnC, np.array([I, J]).T)
        return inds, V
    elif dim == 3:
        I, J, K, V = [], [], [], []
        for i in range(M.shape_cells[0]):
            for j in range(M.shape_cells[1]):
                for k in range(M.shape_cells[2]):
                    x = M.nodes_x[[i, i + 1]]
                    y = M.nodes_y[[j, j + 1]]
                    z = M.nodes_z[[k, k + 1]]
                    v = _lengthInCell(O, D, [x, y, z])
                    if v is not None:
                        I += [i]
                        J += [j]
                        K += [k]
                        V += [v]
        inds = sub2ind(M.vnC, np.array([I, J, K]).T)
        return inds, V


class SimulationIntegral(LinearSimulation):
    slowness, slownessMap, slownessDeriv = props.Invertible("Slowness model (1/v)")

    def __init__(
        self, mesh=None, survey=None, slowness=None, slownessMap=None, **kwargs
    ):
        super().__init__(mesh=mesh, survey=survey, **kwargs)
        self.slowness = slowness
        self.slownessMap = slownessMap

    @property
    def A(self):
        if getattr(self, "_A", None) is not None:
            return self._A

        self._A = sp.lil_matrix((self.survey.nD, self.mesh.nC))
        row = 0
        for src in self.survey.source_list:
            for rx in src.receiver_list:
                for loc_i in range(rx.locations.shape[0]):
                    inds, V = _lineintegral(
                        self.mesh, src.location, rx.locations[loc_i, :]
                    )
                    self._A[inds * 0 + row, inds] = V
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