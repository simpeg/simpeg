import numpy as np
from scipy.interpolate import LinearNDInterpolator
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.spatial import cKDTree
from scipy.interpolate.interpnd import _ndim_coords_from_arrays
from matplotlib.colors import LightSource, Normalize
import matplotlib.gridspec as gridspec
from SimPEG.Utils import mkvc

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


def plotDataHillside(x, y, z, axs=None, fill=True, contour=0,
                     vmin=None, vmax=None,
                     clabel=True, cmap='RdBu_r', ve=1., alpha=1., alphaHS=1.,
                     distMax=1000, midpoint=None, azdeg=315, altdeg=45):

    ls = LightSource(azdeg=azdeg, altdeg=altdeg)

    if x.ndim == 1:
        # Create grid of points
        vectorX = np.linspace(x.min(), x.max(), 1000)
        vectorY = np.linspace(y.min(), y.max(), 1000)

        X, Y = np.meshgrid(vectorX, vectorY)

        # Interpolate
        d_grid = griddata(np.c_[x, y], z, (X, Y), method='cubic')

        # Remove points beyond treshold
        tree = cKDTree(np.c_[x, y])
        xi = _ndim_coords_from_arrays((X, Y), ndim=2)
        dists, indexes = tree.query(xi)

        # Copy original result but mask missing values with NaNs
        d_grid[dists > distMax] = np.nan

    else:

        X, Y, d_grid = x, y, z

    class MidPointNorm(Normalize):
        def __init__(self, midpoint=None, vmin=None, vmax=None, clip=False):
            Normalize.__init__(self, vmin, vmax, clip)
            self.midpoint = midpoint

        def __call__(self, value, clip=None):
            if clip is None:
                clip = self.clip

            result, is_scalar = self.process_value(value)

            self.autoscale_None(result)

            if self.midpoint is None:
                self.midpoint = np.mean(value)
            vmin, vmax, midpoint = self.vmin, self.vmax, self.midpoint

            if not (vmin < midpoint < vmax):
                raise ValueError("midpoint must be between maxvalue and minvalue.")
            elif vmin == vmax:
                result.fill(0) # Or should it be all masked? Or 0.5?
            elif vmin > vmax:
                raise ValueError("maxvalue must be bigger than minvalue")
            else:
                vmin = float(vmin)
                vmax = float(vmax)
                if clip:
                    mask = np.ma.getmask(result)
                    result = np.ma.array(np.clip(result.filled(vmax), vmin, vmax),
                                      mask=mask)

                # ma division is very slow; we can take a shortcut
                resdat = result.data

                # First scale to -1 to 1 range, than to from 0 to 1.
                resdat -= midpoint
                resdat[resdat > 0] /= abs(vmax - midpoint)
                resdat[resdat < 0] /= abs(vmin - midpoint)

                resdat /= 2.
                resdat += 0.5
                result = np.ma.array(resdat, mask=result.mask, copy=False)

            if is_scalar:
                result = result[0]
            return result

        def inverse(self, value):
            if not self.scaled():
                raise ValueError("Not invertible until scaled")
            vmin, vmax, midpoint = self.vmin, self.vmax, self.midpoint

            if cbook.iterable(value):
                val = ma.asarray(value)
                val = 2 * (val-0.5)
                val[val > 0] *= abs(vmax - midpoint)
                val[val < 0] *= abs(vmin - midpoint)
                val += midpoint
                return val
            else:
                val = 2 * (val - 0.5)
                if val < 0:
                    return val*abs(vmin-midpoint) + midpoint
                else:
                    return val*abs(vmax-midpoint) + midpoint

    im, CS = [], []
    if axs is None:
        axs = plt.subplot()

    if fill:
        extent = x.min(), x.max(), y.min(), y.max()
        im = axs.contourf(
            X, Y, d_grid, 50, vmin=vmin, vmax=vmax,
            cmap=cmap, norm=MidPointNorm(midpoint=midpoint), alpha=alpha
        )

        axs.imshow(ls.hillshade(d_grid, vert_exag=ve, dx=1., dy=1.),
                   cmap='gray', alpha=alphaHS,
                   extent=extent, origin='lower')

    if contour > 0:
        CS = axs.contour(
            X, Y, d_grid, int(contour), colors='k', linewidths=0.5
        )

        if clabel:
            plt.clabel(CS, inline=1, fontsize=10, fmt='%i')
    return im, CS


def plotModelSections(mesh, m, normal='x', ind=0, vmin=None, vmax=None,
                      subFact=2, scale=1., xlim=None, ylim=None, vec='k',
                      title=None, axs=None, actv=None, contours=None, fill=True,
                      orientation='vertical', cmap='pink_r',
                      contourf=False, colorbar=False):

    """
    Plot section through a 3D tensor model
    """
    # plot recovered model
    nC = mesh.nC

    if vmin is None:
        vmin = m[np.isnan(m)!=True].min()

    if vmax is None:
        vmax = m[np.isnan(m)!=True].max()

    if len(m) == 3*nC:
        m_lpx = m[0:nC]
        m_lpy = m[nC:2*nC]
        m_lpz = m[2*nC:]

        if actv is not None:
            m_lpx[actv!=True] = np.nan
            m_lpy[actv!=True] = np.nan
            m_lpz[actv!=True] = np.nan

        amp = np.sqrt(m_lpx**2. + m_lpy**2. + m_lpz**2.)

        m_lpx = (m_lpx).reshape(mesh.vnC, order='F')
        m_lpy = (m_lpy).reshape(mesh.vnC, order='F')
        m_lpz = (m_lpz).reshape(mesh.vnC, order='F')
        amp = amp.reshape(mesh.vnC, order='F')
    else:

        if actv is not None:
            m[actv!=True] = np.nan

        amp = m.reshape(mesh.vnC, order='F')

    xx = mesh.gridCC[:, 0].reshape(mesh.vnC, order="F")
    zz = mesh.gridCC[:, 2].reshape(mesh.vnC, order="F")
    yy = mesh.gridCC[:, 1].reshape(mesh.vnC, order="F")

    if axs is None:
        fig, axs = plt.figure(), plt.subplot()

    if normal == 'x':
        xx = yy[ind, :, :].T
        yy = zz[ind, :, :].T
        model = amp[ind, :, :].T

        if len(m) == 3*nC:
            mx = m_lpy[ind, ::subFact, ::subFact].T
            my = m_lpz[ind, ::subFact, ::subFact].T

    elif normal == 'y':
        xx = xx[:, ind, :].T
        yy = zz[:, ind, :].T
        model = amp[:, ind, :].T

        if len(m) == 3*nC:
            mx = m_lpx[::subFact, ind, ::subFact].T
            my = m_lpz[::subFact, ind, ::subFact].T

    elif normal == 'z':

        if actv is not None:
            actIndFull = np.zeros(mesh.nC, dtype=bool)
            actIndFull[actv] = True
        else:
            actIndFull = np.ones(mesh.nC, dtype=bool)

        actIndFull = actIndFull.reshape(mesh.vnC, order='F')

        model = np.zeros((mesh.nCx, mesh.nCy))
        mx = np.zeros((mesh.nCx, mesh.nCy))
        my = np.zeros((mesh.nCx, mesh.nCy))
        for ii in range(mesh.nCx):
            for jj in range(mesh.nCy):

                zcol = actIndFull[ii, jj, :]
                model[ii, jj] = amp[ii, jj, np.where(zcol)[0][-ind]]

                if len(m) == 3*nC:
                    mx[ii, jj] = m_lpx[ii, jj, np.where(zcol)[0][-ind]]
                    my[ii, jj] = m_lpy[ii, jj, np.where(zcol)[0][-ind]]

        xx = xx[:, :, ind].T
        yy = yy[:, :, ind].T
        model = model.T

        if len(m) == 3*nC:
            mx = mx[::subFact, ::subFact].T
            my = my[::subFact, ::subFact].T

    im2, cbar = [], []
    if fill:
        if contourf:
            im2 = axs.contourf(xx, yy, amp,
                               10, vmin=vmin, vmax=vmax,
                               cmap=cmap)
        else:
            if mesh.dim == 3:
                im2 = mesh.plotSlice(mkvc(amp), ind=ind, normal=normal.upper(), ax=axs, clim=[vmin, vmax],
                                     pcolorOpts={'clim':[vmin, vmax] ,'cmap':cmap})[0]
            else:
                im2 = mesh.plotImage(mkvc(amp), ax=axs, clim=[vmin, vmax],
                                     pcolorOpts={'clim':[vmin, vmax] ,'cmap':cmap, 'alpha':alpha})[0]


        if colorbar:
            cbar = plt.colorbar(im2, orientation=orientation, ax=axs,
                     ticks=np.linspace(vmin, vmax, 4),
                     format="${%.3f}$", shrink=0.5)

    if contours is not None:
        axs.contour(xx, yy, model, contours, colors='k')

    if len(m) == 3*nC:

        axs.quiver(mkvc(xx[::subFact, ::subFact]),
                   mkvc(yy[::subFact, ::subFact]),
                   mkvc(mx),
                   mkvc(my),
                   pivot='mid',
                   scale_units="inches", scale=scale, linewidths=(1,),
                   edgecolors=(vec),
                   headaxislength=0.1, headwidth=10, headlength=30)

    axs.set_aspect('equal')

    if xlim is not None:
        axs.set_xlim(xlim[0], xlim[1])

    if ylim is not None:
        axs.set_ylim(ylim[0], ylim[1])

    if title is not None:
        axs.set_title(title)

    return axs, im2, cbar


# def vizCond(mesh, model, axs=None, normal = 'z', ind = 0, xlim=None, ylim=None, vmin=None, contours=None, fill=True, vmax=None,subFact=None, scale=1., savefig=False, cmap = 'jet_r', figname="Conductivity.png"):


#     axs, im, cbar = plotModelSections(mesh, model, normal=normal,
#                                ind=ind, axs=axs, cmap=cmap, subFact=subFact,
#                                xlim=xlim, scale = scale, vec ='w',
#                                ylim=ylim, contours=contours, fill=fill,
#                                vmin=vmin, vmax=vmax)





#     if normal=='x':
#         axs.set_title(str(int(mesh.vectorCCx[ind])) + ' E')
#         # Add lakes and hydro
# #         for file in pline[:11]:
# #             trace = np.loadtxt(file, skiprows=1, delimiter=',')
# #             ax2.plot(trace[:,1], trace[:,2], 'k', ms=1)
# #             ax2.text(trace[0,1], trace[0,2],file[28:-4])

#     elif normal=='y':
#         axs.set_title(str(int(mesh.vectorCCy[ind])) + ' N')
#         # Add lakes and hydro
# #         for file in pline[11:]:
# #             trace = np.loadtxt(file, skiprows=1, delimiter=',')
# #             ax2.plot(trace[:,0], trace[:,2], 'k', ms=1)
# #             ax2.text(trace[0,0], trace[0,2],file[28:-4])

#     else:
#         axs.set_title('Depth: -' + str(np.sum(mesh.hz[-ind:-1])+mesh.hz[-ind]/2) + ' m')

#     return axs, im, cbar


def plotProfile(xyzd, a, b, npts, data=None,
                fig=None, ax=None, plotStr='k',
                coordinate_system='local'):
    """
    Plot the data and line profile inside the spcified limits
    """
    def linefun(x1, x2, y1, y2, nx, tol=1e-3):
        dx = x2-x1
        dy = y2-y1

        if np.abs(dx) <= tol:
            y = np.linspace(y1, y2, nx)
            x = np.ones_like(y)*x1
        elif np.abs(dy) <= tol:
            x = np.linspace(x1, x2, nx)
            y = np.ones_like(x)*y1
        else:
            x = np.linspace(x1, x2, nx)
            slope = (y2-y1)/(x2-x1)
            y = slope*(x-x1)+y1
        return x, y

    if fig is None:
        fig = plt.figure(figsize=(6, 9))

        plt.rcParams.update({'font.size': 14})

    if ax is None:
        ax = plt.subplot()

    x, y = linefun(a[0], b[0], a[1], b[1], npts)
    distance = np.sqrt((x-a[0])**2.+(y-a[1])**2.)
    dline = griddata(xyzd[:, :2], xyzd[:, -1], (x, y), method='cubic')

    if coordinate_system == 'xProfile':
        distance += a[0]
    elif coordinate_system == 'yProfile':
        distance += a[1]

    ax.plot(distance, dline, plotStr)

    if data is not None:

        # if len(plotStr) == len(data):
        for ii, d in enumerate(data):

            dline = griddata(xyzd[:, :2], d, (x, y), method='cubic')

            if plotStr[ii]:
                ax.plot(distance, dline, plotStr[ii])
            else:
                ax.plot(distance, dline)

    ax.set_xlim(distance.min(), distance.max())

    # ax.set_xlabel("Distance (m)")
    # ax.set_ylabel("Magnetic field (nT)")

    #ax.text(distance.min(), dline.max()*0.8, 'A', fontsize = 16)
    # ax.text(distance.max()*0.97, out_linei.max()*0.8, 'B', fontsize = 16)
    # ax.legend(("Observed", "Simulated"), bbox_to_anchor=(0.5, -0.3))
    # ax.grid(True)

    return ax
