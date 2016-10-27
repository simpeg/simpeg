from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from scipy.interpolate import LinearNDInterpolator
import matplotlib.pyplot as plt

def plot2Ddata(xyz, data, vec=None, nx=100, ny=100, ax=None, mask=None, level=None, figname=None, ncontour=10, dataloc=False, contourOpts={}, clim=None):
    """
        TODOs:

            Write explanation for parameters

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
        if clim is not None:
            DATA[DATA<clim[0]] = clim[0]
            DATA[DATA>clim[1]] = clim[1]
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
        if clim is not None:
            DATA[DATA<clim[0]] = clim[0]
            DATA[DATA>clim[1]] = clim[1]
        DATAx = DATAx.reshape(X.shape)
        DATAy = DATAy.reshape(X.shape)
        cont = ax.contourf(X, Y, DATA, ncontour, **contourOpts)
        ax.streamplot(X, Y, DATAx, DATAy, color="w")
        if level is not None:
            CS = ax.contour(X, Y, DATA, level, colors="k", linewidths=2)

#         plt.clabel(CS, inline=1, fmt="%.1e")
    if dataloc:
        ax.plot(xyz[:,0], xyz[:,1], 'k.', ms=1)
    plt.gca().set_aspect('equal', adjustable='box')
    if figname:
        plt.axis("off")
        fig.savefig(figname, dpi=200)
    if level is  None:
        return True, cont, ax
    else:
        return True, cont, ax, CS
