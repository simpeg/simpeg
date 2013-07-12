import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class TensorView(object):
    """
    Provides viewing functions for TensorMesh

    This class is inherited by TensorMesh
    """
    def __init__(self):
        pass

    def plotImage(self, I):

        fig = plt.figure(1)
        fig.clf()
        ax = plt.subplot(111)
        if self.dim == 1:
            if np.size(I) == self.n[0]:
                print 'cell-centered image'
                xx = self.gridCC
                ax.plot(xx, I, 'ro')
            elif np.size(I) == self.n[0]+1:
                print 'nodal image'
                xx = self.gridN
                ax.plot(xx, I, 'bs')
        elif self.dim == 2:
            print "assume cell-centered image"
            x  = self.vectorNx
            y  = self.vectorNy
            fh = ax.pcolormesh(x,y,I.reshape(self.n,order='F').T)
            lx = plt.xlabel("x")
            ly = plt.xlabel("y")
            fig.colorbar(fh)
        fig.show()


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
