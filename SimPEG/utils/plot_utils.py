import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
import matplotlib.pyplot as plt
from matplotlib import colors
import warnings


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
    streamplotOpts={},
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

        # set vmin, vmax if they are not already set
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

        contourOpts = {"levels": levels, "norm": norm, "zorder": 1, **contourOpts}
        cont = ax.contourf(X, Y, DATA, **contourOpts)

        if level:
            levelOpts = {"levels": levels, "zorder": 3, **levelOpts}
            CS = ax.contour(X, Y, DATA, **levelOpts)

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
        DATA = np.sqrt(DATAx**2 + DATAy**2).reshape(X.shape)
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

        contourOpts = {"levels": levels, "norm": norm, "zorder": 1, **contourOpts}
        cont = ax.contourf(X, Y, DATA, **contourOpts)

        streamplotOpts = {"zorder": 4, "color": "w", **streamplotOpts}
        ax.streamplot(X, Y, DATAx, DATAy, **streamplotOpts)

        if level:
            levelOpts = {"levels": levels, "zorder": 3, **levelOpts}
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

        shadeOpts = {
            "cmap": "Greys",
            "alpha": 0.35,
            "antialiased": True,
            "zorder": 2,
            **shadeOpts,
        }

        ax.contourf(
            X,
            Y,
            hillshade(DATA, shade_azimuth, shade_angle_altitude),
            shade_ncontour,
            **shadeOpts
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


def plot_1d_layer_model(
    thicknesses,
    values,
    z0=0,
    scale="log",
    ax=None,
    plot_elevation=False,
    show_layers=False,
    **kwargs
):
    """
    Plot the vertical profile for a 1D layered Earth model.

    Input:
    thicknesses : List[Float]
        A list or numpy.array containing the layer thicknesses from the top layer down
    values : List[Float]
        A list or numpy.array containing the physical property values from the top layer down
    z0 : Float
        Elevation of the surface
    scale: str
        scale {'linear', 'log'}. Plot physical property values on a linear or log10 scale.
    ax: matplotlib.axes.Axes, optional
        An axis object for the plot
    plot_elevation : bool
        If False, the yaxis will be the depth. If True, the yaxis is the elevation.
    show_layers : bool
        Plot horizontal lines to denote layers.

    Output:
    matplotlib.axes.Axes
        The axis object that holds the plot

    """

    if len(thicknesses) < len(values):
        thicknesses = np.r_[thicknesses, thicknesses[-1]]
    z_grid = np.r_[0.0, np.cumsum(thicknesses)]
    resistivity = np.repeat(values, 2)
    v_min = 0.9 * np.min(values)
    v_max = 1.1 * np.max(values)

    z = []
    for i in range(0, len(thicknesses)):
        z.append(np.r_[z_grid[i], z_grid[i + 1]])
    z = np.hstack(z)
    z[-1] = 1e200  # A really large number

    if plot_elevation:
        y_label = "Elevation (m)"
        z = z0 - z
        z_grid = z0 - z_grid
        flip_axis = False
    else:
        y_label = "Depth (m)"
        flip_axis = True

    if ax is None:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_axes([0.15, 0.15, 0.75, 0.75])

    if show_layers:
        for locz in z_grid:
            plt.plot(
                np.linspace(v_min, v_max, 2),
                np.ones(2) * locz,
                "k--",
                lw=0.5,
                label="_nolegend_",
            )

    ax.plot(resistivity, z, **kwargs)
    ax.set_xscale(scale)
    ax.set_xlim(v_min, v_max)
    if flip_axis:
        ax.set_ylim(z_grid.max(), z_grid.min())
    else:
        ax.set_ylim(z_grid.min(), z_grid.max())
    ax.set_ylabel(y_label)

    return ax


def plotLayer(
    sig, LocSigZ, xscale="log", ax=None, showlayers=False, xlim=None, **kwargs
):
    warnings.warn(
        "plotLayer has been deprecated, please use plot_1d_layer_model",
        DeprecationWarning,
    )
    thicknesses = np.diff(LocSigZ)
    z0 = LocSigZ[0]
    ax = plot_1d_layer_model(
        thicknesses,
        sig,
        z0=z0,
        scale=xscale,
        ax=ax,
        show_layers=showlayers,
        plot_elevation=False,
    )
    if xlim is not None:
        ax.xlim(0.5 * sig.min(), 2 * sig.max())
    return ax
