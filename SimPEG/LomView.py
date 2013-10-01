import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from utils import mkvc


class LomView(object):
    """
    Provides viewing functions for LogicallyOrthogonalMesh

    This class is inherited by LogicallyOrthogonalMesh

    """
    def __init__(self):
        pass

    def plotGrid(self, length=0.05):
        """Plot the nodal, cell-centered and staggered grids for 1,2 and 3 dimensions.

        .. plot:: examples/mesh/plot_LogicallyOrthogonalMesh.py
        """
        NN = self.r(self.gridN, 'N', 'N', 'M')
        if self.dim == 2:
            fig = plt.figure(2)
            fig.clf()
            ax = plt.subplot(111)
            X1 = np.c_[mkvc(NN[0][:-1, :]), mkvc(NN[0][1:, :]), mkvc(NN[0][:-1, :])*np.nan].flatten()
            Y1 = np.c_[mkvc(NN[1][:-1, :]), mkvc(NN[1][1:, :]), mkvc(NN[1][:-1, :])*np.nan].flatten()

            X2 = np.c_[mkvc(NN[0][:, :-1]), mkvc(NN[0][:, 1:]), mkvc(NN[0][:, :-1])*np.nan].flatten()
            Y2 = np.c_[mkvc(NN[1][:, :-1]), mkvc(NN[1][:, 1:]), mkvc(NN[1][:, :-1])*np.nan].flatten()

            X = np.r_[X1, X2]
            Y = np.r_[Y1, Y2]

            plt.plot(X, Y)

            plt.hold(True)
            Nx = self.r(self.normals, 'F', 'Fx', 'V')
            Ny = self.r(self.normals, 'F', 'Fy', 'V')
            Tx = self.r(self.tangents, 'E', 'Ex', 'V')
            Ty = self.r(self.tangents, 'E', 'Ey', 'V')

            plt.plot(self.gridN[:, 0], self.gridN[:, 1], 'bo')

            nX = np.c_[self.gridFx[:, 0], self.gridFx[:, 0] + Nx[0]*length, self.gridFx[:, 0]*np.nan].flatten()
            nY = np.c_[self.gridFx[:, 1], self.gridFx[:, 1] + Nx[1]*length, self.gridFx[:, 1]*np.nan].flatten()
            plt.plot(self.gridFx[:, 0], self.gridFx[:, 1], 'rs')
            plt.plot(nX, nY, 'r-')

            nX = np.c_[self.gridFy[:, 0], self.gridFy[:, 0] + Ny[0]*length, self.gridFy[:, 0]*np.nan].flatten()
            nY = np.c_[self.gridFy[:, 1], self.gridFy[:, 1] + Ny[1]*length, self.gridFy[:, 1]*np.nan].flatten()
            #plt.plot(self.gridFy[:, 0], self.gridFy[:, 1], 'gs')
            plt.plot(nX, nY, 'g-')

            tX = np.c_[self.gridEx[:, 0], self.gridEx[:, 0] + Tx[0]*length, self.gridEx[:, 0]*np.nan].flatten()
            tY = np.c_[self.gridEx[:, 1], self.gridEx[:, 1] + Tx[1]*length, self.gridEx[:, 1]*np.nan].flatten()
            plt.plot(self.gridEx[:, 0], self.gridEx[:, 1], 'r^')
            plt.plot(tX, tY, 'r-')

            nX = np.c_[self.gridEy[:, 0], self.gridEy[:, 0] + Ty[0]*length, self.gridEy[:, 0]*np.nan].flatten()
            nY = np.c_[self.gridEy[:, 1], self.gridEy[:, 1] + Ty[1]*length, self.gridEy[:, 1]*np.nan].flatten()
            #plt.plot(self.gridEy[:, 0], self.gridEy[:, 1], 'g^')
            plt.plot(nX, nY, 'g-')
            plt.axis('equal')

        elif self.dim == 3:
            fig = plt.figure(3)
            fig.clf()
            ax = fig.add_subplot(111, projection='3d')
            X1 = np.c_[mkvc(NN[0][:-1, :, :]), mkvc(NN[0][1:, :, :]), mkvc(NN[0][:-1, :, :])*np.nan].flatten()
            Y1 = np.c_[mkvc(NN[1][:-1, :, :]), mkvc(NN[1][1:, :, :]), mkvc(NN[1][:-1, :, :])*np.nan].flatten()
            Z1 = np.c_[mkvc(NN[2][:-1, :, :]), mkvc(NN[2][1:, :, :]), mkvc(NN[2][:-1, :, :])*np.nan].flatten()

            X2 = np.c_[mkvc(NN[0][:, :-1, :]), mkvc(NN[0][:, 1:, :]), mkvc(NN[0][:, :-1, :])*np.nan].flatten()
            Y2 = np.c_[mkvc(NN[1][:, :-1, :]), mkvc(NN[1][:, 1:, :]), mkvc(NN[1][:, :-1, :])*np.nan].flatten()
            Z2 = np.c_[mkvc(NN[2][:, :-1, :]), mkvc(NN[2][:, 1:, :]), mkvc(NN[2][:, :-1, :])*np.nan].flatten()

            X3 = np.c_[mkvc(NN[0][:, :, :-1]), mkvc(NN[0][:, :, 1:]), mkvc(NN[0][:, :, :-1])*np.nan].flatten()
            Y3 = np.c_[mkvc(NN[1][:, :, :-1]), mkvc(NN[1][:, :, 1:]), mkvc(NN[1][:, :, :-1])*np.nan].flatten()
            Z3 = np.c_[mkvc(NN[2][:, :, :-1]), mkvc(NN[2][:, :, 1:]), mkvc(NN[2][:, :, :-1])*np.nan].flatten()

            X = np.r_[X1, X2, X3]
            Y = np.r_[Y1, Y2, Y3]
            Z = np.r_[Z1, Z2, Z3]

            plt.plot(X, Y, 'b', zs=Z)
            ax.set_zlabel('x3')

        ax.grid(True)
        ax.hold(False)
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        fig.show()
