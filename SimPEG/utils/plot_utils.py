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
    """Interpolate and plot unstructured 2D data.

    General plotting for scalar and vector quantities as a function of their
    *x* and *y* locations. **plot2Ddata** uses interpolates the unstructured
    data to a specified set of gridded locations before plotting with
    :meth:`matplotlib.pyplot.contourf`. For vectors, :meth:`matplotlib.pyplot.streamplot`
    is used to add a stream plot. As this function produces a plot for
    2D data, the vertical position and vertical vector component (in the case
    of a vector) is ignored.

    Parameters
    ----------
    xyz : numpy.ndarray
        Data locations [x,y(,z)]. If the data locations are defined in 3D, the z-column is ignored.
    data : numpy.ndarray
        Data values. For scalar quantities, the data are stored in a 1D ``numpy.ndarray``.
        For vector quantities, data are stored in a numpy array of shape (N, dim).
    vec : bool
        If ``True``, the data values represent a vector quantity and the function
        creates a stream plot illustrating the *x* and *y* components of the vector.
    nx : int
        Number of grid locations along x-direction
    ny : int
        Number of grid locations along y-direction
    ax : matplotlib.axes
        An axes object on which to plot. If ``None``, the function creates an axes object
    mask : numpy.ndarray of bool
        Locations in the unstructured grid whose data are masked.
    level : boolean
        If ``True``, adds contours according to :meth:`matplotlib.pyplot.contour`
    figname : str
        Figure name
    ncontour : int
        number of contours in the contour plot
    dataloc : bool
        If ``True``, plot the data locations
    contourOpts : dict
        Dictionary defining keyword arguments when :meth:`matplotlib.pyplot.contourf` is called
    levelOpts : dict
        Dictionary defining keyword arguments when :meth:`matplotlib.pyplot.contourf` is called.
        This is only necessary when *level* = ``True``.
    clim : (2) numpy.ndarray
        Colorbar limits
    method : str
        Interpolation method used to approximate at gridded locations. Must be 'linear' or 'nearest'
    shade : bool
        If ``True``, add shading to the plot
    shade_ncontour : int
        Number of :meth:`matplotlib.pyplot.contourf` contours for the shading
    shade_azimuth : float
        Azimuthal angle for the light source if shading
    shade_angle_altitude : float
        Altitude angle for the light source if shading

    Returns
    -------
    cont : matplotlib.contour.ContourSet
        The filled contour plot
    ax : matplotlib.axes
        The axes object for the plot.
    CS : matplotlib.contour.ContourSet
        If the input parameter *levels* is ``True``, the function outputs
        the level set for the contours

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

    Parameters
    ----------
    thicknesses : list or numpy.ndarray of float
        A ``list`` or ``numpy.ndarray`` containing the layer thicknesses from the top layer down
    values : list or numpy.ndarray of float
        A ``list`` or ``numpy.ndarray`` containing the physical property values from the top layer down
    z0 : float
        Elevation of the surface
    scale: {'linear', 'log'}
        Plot physical property values on a linear or log10 scale.
    ax: matplotlib.axes.Axes, optional
        An axis object for the plot
    plot_elevation : bool
        If ``False``, the yaxis will be the depth. If ``True``, the yaxis is the elevation.
    show_layers : bool
        Plot horizontal lines to denote layers.

    Returns
    -------
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
    """*plotLayer* has been deprecated, please use :func:`plot_1d_layer_model`"""
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
