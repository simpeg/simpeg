"""
Utils: surface2ind_topo
=======================

Here we show how to use :code:`Utils.surface2ind_topo` to identify
cells below a topographic surface and compare the different options
"""
import numpy as np
from SimPEG import Mesh
from SimPEG import Utils
from SimPEG.Utils import surface2ind_topo, mkvc
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


def run(plotIt=True, nx=5, ny=5):

    # 2D mesh
    mesh = Mesh.TensorMesh([nx, ny], x0='CC')
    xtopo = mesh.vectorNx

    # define a topographic surface
    topo = 0.4*np.sin(xtopo*5)

    # make it an array
    Topo = np.hstack([Utils.mkvc(xtopo, 2), Utils.mkvc(topo, 2)])

    # Compare the different options
    indtopoCC_near = surface2ind_topo(mesh, Topo, gridLoc='CC', method='nearest')
    indtopoN_near = surface2ind_topo(mesh, Topo, gridLoc='N', method='nearest')

    indtopoCC_linear = surface2ind_topo(mesh, Topo, gridLoc='CC', method='linear')
    indtopoN_linear = surface2ind_topo(mesh, Topo, gridLoc='N', method='linear')

    indtopoCC_cubic = surface2ind_topo(mesh, Topo, gridLoc='CC', method='cubic')
    indtopoN_cubic = surface2ind_topo(mesh, Topo, gridLoc='N', method='cubic')

    if plotIt:
        fig, ax = plt.subplots(2, 3, figsize=(9, 6))
        ax = mkvc(ax)
        xinterpolate = np.linspace(mesh.gridN[:, 0].min(), mesh.gridN[:, 0].max(),100)
        listindex = [indtopoCC_near,indtopoN_near,indtopoCC_linear,indtopoN_linear,indtopoCC_cubic,indtopoN_cubic]
        listmethod = ['nearest','nearest', 'linear', 'linear', 'cubic', 'cubic']
        for i in range(6):
            mesh.plotGrid(ax=ax[i], nodes=True, centers=True)
            mesh.plotImage(listindex[i], ax=ax[i], pcolorOpts = {"alpha":0.5, "cmap":plt.cm.gray})
            ax[i].scatter(Topo[:,0], Topo[:,1], color = 'black', marker = 'o',s = 50)
            ax[i].plot(
                xinterpolate,
                interp1d(Topo[:, 0], Topo[:, 1], kind=listmethod[i])(xinterpolate),
                '--k',
                linewidth=3
                )
            ax[i].xaxis.set_ticklabels([])
            ax[i].yaxis.set_ticklabels([])
            ax[i].set_aspect('equal')
            ax[i].set_xlabel('')
            ax[i].set_ylabel('')

        ax[0].set_xlabel('Nearest Interpolation', fontsize=16)
        ax[2].set_xlabel('Linear Interpolation', fontsize=16)
        ax[4].set_xlabel('Cubic Interpolation', fontsize=16)

        ax[0].set_ylabel('Cells Center \n based selection', fontsize=16)
        ax[1].set_ylabel('Nodes \n based selection', fontsize=16)

        plt.tight_layout()


if __name__ == '__main__':
    run()
    plt.show()
