import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D


class TensorView(object):
    """
    Provides viewing functions for TensorMesh

    This class is inherited by TensorMesh
    """
    def __init__(self):
        pass

    def plotImage(self, I, imageType='CC', figNum=1,ax=None):

        assert type(I) == np.ndarray, "I must be a numpy array"
        assert imageType in ["CC", "N"], "imageType must be 'CC' or 'N'"

        if imageType == 'CC':
            assert I.size == self.nC, "Incorrect dimensions for CC."
        elif imageType == 'N':
            assert I.size == self.nN, "Incorrect dimensions for N."

        if ax is None:
            fig = plt.figure(figNum)
            fig.clf()
            ax = plt.subplot(111)
        else:
            assert isinstance(ax,matplotlib.axes.Axes), "ax must be an Axes!"
            fig = ax.figure
       
        if self.dim == 1:
            if imageType == 'CC':
                ph = ax.plot(self.vectorCCx, I, '-ro')
            elif imageType == 'N':
                ph = ax.plot(self.vectorNx, I, '-bs')
            ax.set_xticks(self.vectorNx)
            ax.set_xlabel("x")
            ax.axis('tight')
        elif self.dim == 2:
            if imageType == 'CC':
                C = I[:].reshape(self.n, order='F')
            elif imageType == 'N':
                C = I[:].reshape(self.n+1, order='F')
                C = 0.25*(C[:-1, :-1] + C[1:, :-1] + C[:-1, 1:] + C[1:, 1:])

            ph = ax.pcolormesh(self.vectorNx, self.vectorNy, C.T)
            ax.axis('tight')
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_xticks(self.vectorNx)
            ax.set_yticks(self.vectorNy)
            

        fig.show()
        return ph


    def plotGrid(self):
        """Plot the nodal, cell-centered and staggered grids for 1,2 and 3 dimensions."""
        if self.dim == 1:
            fig = plt.figure(1)
            fig.clf()
            ax = plt.subplot(111)
            xn = self.gridN
            xc = self.gridCC
            ax.hold(True)
            ax.plot(xn, np.ones(np.shape(xn)), 'bs')
            ax.plot(xc, np.ones(np.shape(xc)), 'ro')
            ax.plot(xn, np.ones(np.shape(xn)), 'k--')
            ax.grid(True)
            ax.hold(False)
            ax.set_xlabel('x1')
            fig.show()
        elif self.dim == 2:
            fig = plt.figure(2)
            fig.clf()
            ax = plt.subplot(111)
            xn = self.gridN
            xc = self.gridCC
            xs1 = self.gridFx
            xs2 = self.gridFy

            ax.hold(True)
            ax.plot(xn[:, 0], xn[:, 1], 'bs')
            ax.plot(xc[:, 0], xc[:, 1], 'ro')
            ax.plot(xs1[:, 0], xs1[:, 1], 'g>')
            ax.plot(xs2[:, 0], xs2[:, 1], 'g^')
            ax.grid(True)
            ax.hold(False)
            ax.set_xlabel('x1')
            ax.set_ylabel('x2')
            fig.show()
        elif self.dim == 3:
            fig = plt.figure(3)
            fig.clf()
            ax = fig.add_subplot(111, projection='3d')
            xn = self.gridN
            xc = self.gridCC
            xfs1 = self.gridFx
            xfs2 = self.gridFy
            xfs3 = self.gridFz

            xes1 = self.gridEx
            xes2 = self.gridEy
            xes3 = self.gridEz

            ax.hold(True)
            ax.plot(xn[:, 0], xn[:, 1], 'bs', zs=xn[:, 2])
            ax.plot(xc[:, 0], xc[:, 1], 'ro', zs=xc[:, 2])
            ax.plot(xfs1[:, 0], xfs1[:, 1], 'g>', zs=xfs1[:, 2])
            ax.plot(xfs2[:, 0], xfs2[:, 1], 'g<', zs=xfs2[:, 2])
            ax.plot(xfs3[:, 0], xfs3[:, 1], 'g^', zs=xfs3[:, 2])
            ax.plot(xes1[:, 0], xes1[:, 1], 'k>', zs=xes1[:, 2])
            ax.plot(xes2[:, 0], xes2[:, 1], 'k<', zs=xes2[:, 2])
            ax.plot(xes3[:, 0], xes3[:, 1], 'k^', zs=xes3[:, 2])
            ax.grid(True)
            ax.hold(False)
            ax.set_xlabel('x1')
            ax.set_ylabel('x2')
            ax.set_zlabel('x3')
            fig.show()
