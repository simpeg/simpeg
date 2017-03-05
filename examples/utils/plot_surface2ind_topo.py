import numpy as np
from SimPEG import Mesh
from SimPEG import Utils
from SimPEG.Utils import surface2ind_topo
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


def run(plotIt=True, nx=5, ny=5):
    """

        Utils: surface2ind_topo
        =======================

        Here we show how to use :code:`Utils.surface2ind_topo` to identify
        cells below a topographic surface.

    """

    # 2D mesh
    mesh = Mesh.TensorMesh([nx, ny], x0='CC')
    xtopo = np.linspace(mesh.gridN[:, 0].min(), mesh.gridN[:, 0].max())

    # define a topographic surface
    topo = 0.4*np.sin(xtopo*5)

    # make it an array
    Topo = np.hstack([Utils.mkvc(xtopo, 2), Utils.mkvc(topo, 2)])

    indcc = surface2ind_topo(mesh, Topo, 'CC')

    if plotIt:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        mesh.plotGrid(ax=ax, nodes=True, centers=True)
        ax.plot(xtopo, topo, 'k', linewidth=1)
        ax.plot(
            mesh.vectorCCx,
            interp1d(xtopo, topo)(mesh.vectorCCx),
            '--k',
            linewidth=3
        )

        aveN2CC = Utils.sdiag(mesh.aveN2CC.T.sum(1))*mesh.aveN2CC.T
        a = aveN2CC * indcc
        a[a > 0] = 1.
        a[a < 0.25] = np.nan
        a = a.reshape(mesh.vnN, order='F')
        masked_array = np.ma.array(a, mask=np.isnan(a))
        ax.pcolor(
            mesh.vectorNx,
            mesh.vectorNy,
            masked_array.T,
            cmap=plt.cm.gray,
            alpha=0.2
        )


if __name__ == '__main__':
    run()
    plt.show()
