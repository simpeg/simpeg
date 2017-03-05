import numpy as np
from scipy.interpolate import LinearNDInterpolator
import matplotlib.pyplot as plt


def plot2Ddata(xyz, data, vec=False, nx=100, ny=100,
               ax=None, mask=None, level=None, figname=None,
               ncontour=10, dataloc=False, contourOpts={},
               scale="linear", clim=None):
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

    xmin, xmax = xyz[:, 0].min(), xyz[:, 0].max()
    ymin, ymax = xyz[:, 1].min(), xyz[:, 1].max()
    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(x, y)
    xy = np.c_[X.flatten(), Y.flatten()]
    if vec is False:
        F = LinearNDInterpolator(xyz[:, :2], data)
        DATA = F(xy)
        DATA = DATA.reshape(X.shape)
        if scale == "log":
            DATA = np.log10(abs(DATA))
        cont = ax.contourf(X, Y, DATA, ncontour, **contourOpts)
        if level is not None:
            if scale == "log":
                level = np.log10(level)
            CS = ax.contour(X, Y, DATA, level, colors="k", linewidths=2)

    else:
        # Assume size of data is (N,2)
        datax = data[:, 0]
        datay = data[:, 1]
        Fx = LinearNDInterpolator(xyz[:, :2], datax)
        Fy = LinearNDInterpolator(xyz[:, :2], datay)
        DATAx = Fx(xy)
        DATAy = Fy(xy)
        DATA = np.sqrt(DATAx**2+DATAy**2).reshape(X.shape)
        DATAx = DATAx.reshape(X.shape)
        DATAy = DATAy.reshape(X.shape)
        if scale == "log":
            DATA = np.log10(abs(DATA))

        cont = ax.contourf(X, Y, DATA, ncontour, **contourOpts)
        ax.streamplot(X, Y, DATAx, DATAy, color="w")
        if level is not None:
            CS = ax.contour(X, Y, DATA, level, colors="k", linewidths=2)

    if dataloc:
        ax.plot(xyz[:, 0], xyz[:, 1], 'k.', ms=2)
    plt.gca().set_aspect('equal', adjustable='box')
    if figname:
        plt.axis("off")
        fig.savefig(figname, dpi=200)
    if level is None:
        return cont, ax
    else:
        return cont, ax, CS


def plotLayer(sig, LocSigZ, xscale='log', ax=None,
              showlayers=False, xlim=None, **kwargs):
    """Plot a layered earth model"""
    sigma = np.repeat(sig, 2, axis=0)
    z = np.repeat(LocSigZ[1:], 2, axis=0)
    z = np.r_[LocSigZ[0], z, LocSigZ[-1]]

    if xlim is None:
        sig_min = sig.min()*0.5
        sig_max = sig.max()*2
    else:
        sig_min, sig_max = xlim

    if xscale == 'linear' and sig.min() == 0.:
        if xlim is None:
            sig_min = -sig.max()*0.5
            sig_max = sig.max()*2

    if ax is None:
        plt.xscale(xscale)
        plt.xlim(sig_min, sig_max)
        plt.ylim(z.min(), z.max())
        plt.xlabel('Conductivity (S/m)', fontsize=14)
        plt.ylabel('Depth (m)', fontsize=14)
        plt.ylabel('Depth (m)', fontsize=14)
        if showlayers is True:
            for locz in LocSigZ:
                plt.plot(
                    np.linspace(sig_min, sig_max, 100),
                    np.ones(100)*locz, 'b--', lw=0.5
                )
        return plt.plot(sigma, z, 'k-', **kwargs)

    else:
        ax.set_xscale(xscale)
        ax.set_xlim(sig_min, sig_max)
        ax.set_ylim(z.min(), z.max())
        ax.set_xlabel('Conductivity (S/m)', fontsize=14)
        ax.set_ylabel('Depth (m)', fontsize=14)
        if showlayers is True:
            for locz in LocSigZ:
                ax.plot(
                    np.linspace(sig_min, sig_max, 100),
                    np.ones(100)*locz, 'b--', lw=0.5
                )
        return ax.plot(sigma, z, 'k-', **kwargs)
