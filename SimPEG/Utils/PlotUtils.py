import numpy as np
from scipy.interpolate import LinearNDInterpolator
import matplotlib.pyplot as plt

def plot2Ddata(xyz, data, vec=False, nx=100, ny=100,
               ax=None, mask=None, level=None, figname=None,
               ncontour=10, dataloc=False, contourOpts={}, clim=None):
    """

        Take unstructured xy points, interpolate, then plot in 2D

        :param numpy.array xyz: data locations
        :param numpy.array data: data values
        :param bool vec: plot streamplot?
        :param float nx: number of x grid locations
        :param float ny: number of y grid locations
        :param matplotlib.axes ax: axes
        :param numpy.array mask: mask for the array
        :param float level: level at which to draw a contour
        :param string figname: figure name
        :param float ncontour: number of :meth:`matplotlib.pyplot.contourf`
                               contours
        :param bool dataloc: plot the data locations
        :param dict controuOpts: :meth:`matplotlib.pyplot.contourf` options
        :param numpy.array clim: colorbar limits

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
    if vec is False:
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
