import numpy as np
from scipy.interpolate import LinearNDInterpolator

def plot2Ddata(xyz, data, nx=100, ny=100, ax=None, mask=None, level=None, figname=None, contourOpts={}):
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
    F = LinearNDInterpolator(xyz[:,:2], data)
    xy = np.c_[X.flatten(), Y.flatten()]
    DATA = F(xy)
    DATA = DATA.reshape(X.shape)
    cont = ax.contourf(X, Y, DATA, **contourOpts)
    if level is not None:
        CS = ax.contour(X, Y, DATA, level, colors="k", linewidths=2)
#         plt.clabel(CS, inline=1, fmt="%.1e")
    ax.plot(xyz[:,0], xyz[:,1], 'k.', ms=1)
    plt.gca().set_aspect('equal', adjustable='box')
    if figname:
        plt.axis("off")
        fig.savefig(figname, dpi=200)
    if level is  None:
        return F, cont, ax
    else:
        return F, cont, ax, CS
