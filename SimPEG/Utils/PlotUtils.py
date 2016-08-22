import numpy as np
from scipy.interpolate import LinearNDInterpolator
import matplotlib.pyplot as plt

def plot2Ddata(xyz, data, vec=None, nx=100, ny=100,
               ax=None, mask=None, level=None, figname=None,
               ncontour=10, dataloc=False, contourOpts={}, clim=None):
    """

        Take unstructured xy points, interpolate, then plot in 2D

        :param numpy.array gridCC: mesh.gridCC is the cell centered grid
        :param numpy.array modelCC: cell centered model
        :param numpy.array p0: bottom, southwest corner of block
        :param numpy.array p1: top, northeast corner of block
        :blockProp float blockProp: property to assign to the model

        :return numpy.array, modelBlock: model with block

    """
    if ax is None:
        fig = plt.figure()
        ax = plt.subplot(111)

    xmin, xmax = xyz[:,0].min(), xyz[:,0].max()
    ymin, ymax = xyz[:,1].min(), xyz[:,1].max()
    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(x, y)
    xy = np.c_[X.flatten(), Y.flatten()]
    if vec is None:
        F = LinearNDInterpolator(xyz[:,:2], data)
        DATA = F(xy)
        DATA = DATA.reshape(X.shape)
        cont = ax.contourf(X, Y, DATA, ncontour, **contourOpts)
        if level is not None:
            CS = ax.contour(X, Y, DATA, level, colors="k", linewidths=2)

    else:
        # Assume size of data is (N,2)
        datax = data[:,0]
        datay = data[:,1]
        Fx = LinearNDInterpolator(xyz[:,:2], datax)
        Fy = LinearNDInterpolator(xyz[:,:2], datay)
        DATAx = Fx(xy)
        DATAy = Fy(xy)
        DATA = np.sqrt(DATAx**2+DATAy**2).reshape(X.shape)
        DATAx = DATAx.reshape(X.shape)
        DATAy = DATAy.reshape(X.shape)
        cont = ax.contourf(X, Y, DATA, ncontour, **contourOpts)
        ax.streamplot(X, Y, DATAx, DATAy, color="w")
        if level is not None:
            CS = ax.contour(X, Y, DATA, level, colors="k", linewidths=2)

#         plt.clabel(CS, inline=1, fmt="%.1e")
    if dataloc:
        ax.plot(xyz[:,0], xyz[:,1], 'k.', ms=2)
    plt.gca().set_aspect('equal', adjustable='box')
    if figname:
        plt.axis("off")
        fig.savefig(figname, dpi=200)
    if level is  None:
        return cont, ax
    else:
        return cont, ax, CS
