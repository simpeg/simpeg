from SimPEG import *
from SimPEG.Utils import surface2ind_topo


def run(plotIt=True, nx=5, ny=5):
    """

        Utils: surface2ind_topo
        =======================

        Here we show how to use :code:`Utils.surface2ind_topo` to identify cells below
        a topographic surface.

    """

    mesh = Mesh.TensorMesh([nx,ny], x0='CC') # 2D mesh
    xtopo = np.linspace(mesh.gridN[:,0].min(), mesh.gridN[:,0].max())
    topo = 0.4*np.sin(xtopo*5) # define a topographic surface

    Topo = np.hstack([Utils.mkvc(xtopo,2), Utils.mkvc(topo,2)]) #make it an array

    indcc = surface2ind_topo(mesh, Topo, 'CC')

    if plotIt:
        from matplotlib.pylab import plt
        from scipy.interpolate import interp1d
        fig, ax = plt.subplots(1,1, figsize=(6,6))
        mesh.plotGrid(ax=ax, nodes=True, centers=True)
        ax.plot(xtopo,topo,'k',linewidth=1)
        ax.plot(mesh.vectorCCx, interp1d(xtopo,topo)(mesh.vectorCCx),'--k',linewidth=3)

        aveN2CC = Utils.sdiag(mesh.aveN2CC.T.sum(1))*mesh.aveN2CC.T
        a = aveN2CC * indcc
        a[a > 0] = 1.
        a[a < 0.25] = np.nan
        a = a.reshape(mesh.vnN, order='F')
        masked_array = np.ma.array(a, mask=np.isnan(a))
        ax.pcolor(mesh.vectorNx,mesh.vectorNy,masked_array.T, cmap=plt.cm.gray, alpha=0.2)
        plt.show()


if __name__ == '__main__':
    run(plotIt=True)
