import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
import matplotlib.pyplot as plt
from matplotlib import colors


def plot2Ddata(
    xyz,
    data,
    vec=False,
    nx=100,
    ny=100,
    ax=None,
    mask=None,
    level=False,
    figname=None,
    ncontour=10,
    dataloc=False,
    contourOpts={},
    levelOpts={},
    scale="linear",
    clim=None,
    method="linear",
    shade=False,
    shade_ncontour=100,
    shade_azimuth=-45.0,
    shade_angle_altitude=45.0,
    shadeOpts={},
):
    """

        Take unstructured xy points, interpolate, then plot in 2D

        :param numpy.ndarray xyz: data locations
        :param numpy.ndarray data: data values
        :param bool vec: plot streamplot?
        :param float nx: number of x grid locations
        :param float ny: number of y grid locations
        :param matplotlib.axes ax: axes
        :param boolean numpy.ndarray mask: mask for the array
        :param boolean level: boolean to plot (or not)
                                :meth:`matplotlib.pyplot.contour`
        :param string figname: figure name
        :param float ncontour: number of :meth:`matplotlib.pyplot.contourf`
                                contours
        :param bool dataloc: plot the data locations
        :param dict controuOpts: :meth:`matplotlib.pyplot.contourf` options
        :param dict levelOpts: :meth:`matplotlib.pyplot.contour` options
        :param numpy.ndarray clim: colorbar limits
        :param str method: interpolation method, either 'linear' or 'nearest'
        :param bool shade: add shading to the plot
        :param float shade_ncontour: number of :meth:`matplotlib.pyplot.contourf`
                                contours for the shading
        :param float shade_azimuth: azimuth for the light source in shading
        :param float shade_angle_altitude: angle altitude for the light source
                                in shading
        :param dict shaeOpts: :meth:`matplotlib.pyplot.contourf` options

    """

    # Error checking and set vmin, vmax
    vlimits = [None, None]

    if clim is not None:
        vlimits = [np.min(clim), np.max(clim)]

    for i, key in enumerate(["vmin", "vmax"]):
        if key in contourOpts.keys():
            if vlimits[i] is None:
                vlimits[i] = contourOpts.pop(key)
            else:
                if not np.isclose(contourOpts[key], vlimits[i]):
                    raise Exception(
                        "The values provided in the colorbar limit, clim {} "
                        "does not match the value of {} provided in the "
                        "contourOpts: {}. Only one value should be provided or "
                        "the two values must be equal.".format(
                            vlimits[i], key, contourOpts[key]
                        )
                    )
                contourOpts.pop(key)
    vmin, vmax = vlimits[0], vlimits[1]

    # create a figure if it doesn't exist
    if ax is None:
        fig = plt.figure()
        ax = plt.subplot(111)

    # interpolate data to grid locations
    xmin, xmax = xyz[:, 0].min(), xyz[:, 0].max()
    ymin, ymax = xyz[:, 1].min(), xyz[:, 1].max()
    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(x, y)
    xy = np.c_[X.flatten(), Y.flatten()]

    if vec is False:
        if method == "nearest":
            F = NearestNDInterpolator(xyz[:, :2], data)
        else:
            F = LinearNDInterpolator(xyz[:, :2], data)
        DATA = F(xy)
        DATA = DATA.reshape(X.shape)

        # Levels definitions
        dataselection = np.logical_and(~np.isnan(DATA), np.abs(DATA) != np.inf)
        if scale == "log":
            DATA = np.abs(DATA)

        vmin = DATA[dataselection].min() if vmin is None else vmin
        vmax = DATA[dataselection].max() if vmax is None else vmax

        if scale == "log":
            levels = np.logspace(np.log10(vmin), np.log10(vmax), ncontour + 1)
            norm = colors.LogNorm(vmin=vmin, vmax=vmax)
        else:
            levels = np.linspace(vmin, vmax, ncontour + 1)
            norm = colors.Normalize(vmin=vmin, vmax=vmax)

        if mask is not None:
            Fmask = NearestNDInterpolator(xyz[:, :2], mask)
            MASK = Fmask(xy)
            MASK = MASK.reshape(X.shape)
            DATA = np.ma.masked_array(DATA, mask=MASK)

        cont = ax.contourf(X, Y, DATA, levels=levels, norm=norm, **contourOpts)
        if level:
            CS = ax.contour(X, Y, DATA, levels=levels, zorder=3, **levelOpts)

    else:
        # Assume size of data is (N,2)
        datax = data[:, 0]
        datay = data[:, 1]
        if method == "nearest":
            Fx = NearestNDInterpolator(xyz[:, :2], datax)
            Fy = NearestNDInterpolator(xyz[:, :2], datay)
        else:
            Fx = LinearNDInterpolator(xyz[:, :2], datax)
            Fy = LinearNDInterpolator(xyz[:, :2], datay)
        DATAx = Fx(xy)
        DATAy = Fy(xy)
        DATA = np.sqrt(DATAx ** 2 + DATAy ** 2).reshape(X.shape)
        DATAx = DATAx.reshape(X.shape)
        DATAy = DATAy.reshape(X.shape)
        if scale == "log":
            DATA = np.abs(DATA)

        # Levels definitions
        dataselection = np.logical_and(~np.isnan(DATA), np.abs(DATA) != np.inf)

        # set vmin, vmax
        vmin = DATA[dataselection].min() if vmin is None else vmin
        vmax = DATA[dataselection].max() if vmax is None else vmax

        if scale == "log":
            levels = np.logspace(np.log10(vmin), np.log10(vmax), ncontour + 1)
            norm = colors.LogNorm(vmin=vmin, vmax=vmax)
        else:
            levels = np.linspace(vmin, vmax, ncontour + 1)
            norm = colors.Normalize(vmin=vmin, vmax=vmax)

        if mask is not None:
            Fmask = NearestNDInterpolator(xyz[:, :2], mask)
            MASK = Fmask(xy)
            MASK = MASK.reshape(X.shape)
            DATA = np.ma.masked_array(DATA, mask=MASK)

        cont = ax.contourf(X, Y, DATA, levels=levels, norm=norm, **contourOpts)
        ax.streamplot(X, Y, DATAx, DATAy, zorder=4, color="w")
        if level:
            CS = ax.contour(X, Y, DATA, levels=levels, zorder=3, **levelOpts)

    if shade:

        def hillshade(array, azimuth, angle_altitude):
            """
            coded copied from https://www.neonscience.org/create-hillshade-py
            """
            azimuth = 360.0 - azimuth
            x, y = np.gradient(array)
            slope = np.pi / 2.0 - np.arctan(np.sqrt(x * x + y * y))
            aspect = np.arctan2(-x, y)
            azimuthrad = azimuth * np.pi / 180.0
            altituderad = angle_altitude * np.pi / 180.0
            shaded = np.sin(altituderad) * np.sin(slope) + np.cos(altituderad) * np.cos(
                slope
            ) * np.cos((azimuthrad - np.pi / 2.0) - aspect)
            return 255 * (shaded + 1) / 2

        defaultshadeOpts = {
            "cmap": "Greys",
            "alpha": 0.35,
            "antialiased": True,
            "zorder": 2,
        }
        for key in shadeOpts.keys():
            defaultshadeOpts[key] = shadeOpts[key]

        ax.contourf(
            X,
            Y,
            hillshade(DATA, shade_azimuth, shade_angle_altitude),
            shade_ncontour,
            **defaultshadeOpts
        )

    if dataloc:
        ax.plot(xyz[:, 0], xyz[:, 1], "k.", ms=2)
    ax.set_aspect("equal", adjustable="box")
    if figname:
        plt.axis("off")
        fig.savefig(figname, dpi=200)
    if level:
        return cont, ax, CS
    else:
        return cont, ax


def plotLayer(
    sig, LocSigZ, xscale="log", ax=None, showlayers=False, xlim=None, **kwargs
):
    """Plot a layered earth model"""
    sigma = np.repeat(sig, 2, axis=0)
    z = np.repeat(LocSigZ[1:], 2, axis=0)
    z = np.r_[LocSigZ[0], z, LocSigZ[-1]]

    if xlim is None:
        sig_min = sig.min() * 0.5
        sig_max = sig.max() * 2
    else:
        sig_min, sig_max = xlim

    if xscale == "linear" and sig.min() == 0.0:
        if xlim is None:
            sig_min = -sig.max() * 0.5
            sig_max = sig.max() * 2

    if ax is None:
        plt.xscale(xscale)
        plt.xlim(sig_min, sig_max)
        plt.ylim(z.min(), z.max())
        plt.xlabel("Conductivity (S/m)", fontsize=14)
        plt.ylabel("Depth (m)", fontsize=14)
        plt.ylabel("Depth (m)", fontsize=14)
        if showlayers is True:
            for locz in LocSigZ:
                plt.plot(
                    np.linspace(sig_min, sig_max, 100),
                    np.ones(100) * locz,
                    "b--",
                    lw=0.5,
                )
        return plt.plot(sigma, z, "k-", **kwargs)

    else:
        ax.set_xscale(xscale)
        ax.set_xlim(sig_min, sig_max)
        ax.set_ylim(z.min(), z.max())
        ax.set_xlabel("Conductivity (S/m)", fontsize=14)
        ax.set_ylabel("Depth (m)", fontsize=14)
        if showlayers is True:
            for locz in LocSigZ:
                ax.plot(
                    np.linspace(sig_min, sig_max, 100),
                    np.ones(100) * locz,
                    "b--",
                    lw=0.5,
                )
        return ax.plot(sigma, z, "k-", **kwargs)
