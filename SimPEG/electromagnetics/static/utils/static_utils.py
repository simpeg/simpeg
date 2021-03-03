import numpy as np
from scipy.interpolate import LinearNDInterpolator, interp1d, griddata
from scipy.spatial import cKDTree
from numpy import matlib
import discretize
from discretize import TensorMesh
import matplotlib.pyplot as plt
import matplotlib as mpl
import warnings

from ....data import Data
from .. import resistivity as dc
from ....utils import (
    closestPoints,
    mkvc,
    surface2ind_topo,
    model_builder,
    define_plane_from_points,
)


DATA_TYPES = {
    "apparent resistivity": [
        "apparent resistivity",
        "appresistivity",
        "apparentresistivity",
        "apparent-resistivity",
        "apparent_resistivity",
        "appres",
    ],
    "apparent conductivity": [
        "apparent conductivity",
        "appconductivity",
        "apparentconductivity",
        "apparent-conductivity",
        "apparent_conductivity",
        "appcon",
    ],
    "apparent chargeability": [
        "apparent chargeability",
        "appchargeability",
        "apparentchargeability",
        "apparent-chargeability",
        "apparent_chargeability",
    ],
    "potential": ["potential", "potentials", "volt", "V", "voltages", "voltage"],
}

SPACE_TYPES = {
    "half space": ["half space", "half-space", "half_space", "halfspace", "half"],
    "whole space": ["whole space", "whole-space", "whole_space", "wholespace", "whole"],
}


#######################################################################
#                          SURVEY GEOMETRY
#######################################################################

def electrode_separations(survey_object, electrode_pair="all", **kwargs):
    """
    Calculate horizontal separation between specific or all electrodes.

    Input:
    survey_object : SimPEG.electromagnetics.static.survey.Survey
        A DC or IP survey object
    electrode_pair : str or list of str
        A string or list of strings from the following {'all', 'AB', 'MN', 'AM', 'AN', 'BM', 'BN}

    Output:
    list of np.ndarray
        For each electrode pair specified, the electrode distance is returned
        in a list.

    """
    if "survey_type" in kwargs:
        warnings.warn(
            "The survey_type is no longer necessary to calculate electrode separations. "
            "Feel free to remove it from the call. This option will be removed in SimPEG 0.15.0",
            DeprecationWarning,
        )

    if not isinstance(electrode_pair, list):
        if electrode_pair.lower() == "all":
            electrode_pair = ["AB", "MN", "AM", "AN", "BM", "BN"]
        elif isinstance(electrode_pair, str):
            electrode_pair = [electrode_pair.upper()]
        else:
            raise Exception(
                "electrode_pair must be either a string, list of strings, or an "
                "ndarray containing the electrode separation distances you would "
                "like to calculate not {}".format(type(electrode_pair))
            )

    elecSepDict = {}
    AB = []
    MN = []
    AM = []
    AN = []
    BM = []
    BN = []

    for src in survey_object.source_list:
        # pole or dipole source
        if isinstance(src.location, list):
            a_loc = src.location[0]
            b_loc = src.location[1]
        else:
            a_loc = src.location
            b_loc = np.inf * np.ones_like(src.location)

        for rx in src.receiver_list:
            # pole or dipole receiver
            if isinstance(rx.locations, list):
                M = rx.locations[0]
                N = rx.locations[1]
            else:
                M = rx.locations
                N = -np.inf * np.ones_like(rx.locations)

            n_rx = np.shape(M)[0]

            A = matlib.repmat(a_loc, n_rx, 1)
            B = matlib.repmat(b_loc, n_rx, 1)

            # Compute distances
            AB.append(np.sqrt(np.sum((A - B) ** 2.0, axis=1)))
            MN.append(np.sqrt(np.sum((M - N) ** 2.0, axis=1)))
            AM.append(np.sqrt(np.sum((A - M) ** 2.0, axis=1)))
            AN.append(np.sqrt(np.sum((A - N) ** 2.0, axis=1)))
            BM.append(np.sqrt(np.sum((B - M) ** 2.0, axis=1)))
            BN.append(np.sqrt(np.sum((B - N) ** 2.0, axis=1)))

    # Stack to vector and define in dictionary
    if "AB" in electrode_pair:
        if AB:
            AB = np.hstack(AB)
        elecSepDict["AB"] = AB
    if "MN" in electrode_pair:
        if MN:
            MN = np.hstack(MN)
        elecSepDict["MN"] = MN
    if "AM" in electrode_pair:
        if AM:
            AM = np.hstack(AM)
        elecSepDict["AM"] = AM
    if "AN" in electrode_pair:
        if AN:
            AN = np.hstack(AN)
        elecSepDict["AN"] = AN
    if "BM" in electrode_pair:
        if BM:
            BM = np.hstack(BM)
        elecSepDict["BM"] = BM
    if "BN" in electrode_pair:
        if BN:
            BN = np.hstack(BN)
        elecSepDict["BN"] = BN

    return elecSepDict

def pseudo_locations(survey, **kwargs):
    """
    Calculates the pseudo-sensitivity locations for 2D and 3D surveys.

    Input:
    survey : SimPEG.electromagnetics.static.resistivity.Survey
        A DC or IP survey

    Output:
    tuple of numpy.ndarray of the form (midxy, midz)
        For 2D surveys, *midxy* is a vector containing the along line position.
        For 3D surveys, *midxy* is an (n, 2) numpy array containing the (x,y) positions.
        In eithere case, *midz* is a vector containing the pseudo-depth locations.

    """

    if not isinstance(survey, dc.Survey):
        raise ValueError("Input must be of type {}".format(dc.Survey))

    if len(kwargs) > 0:
        warnings.warn(
            "The keyword arguments of this function have been deprecated."
            " All of the necessary information is now in the DC survey class",
            DeprecationWarning,
        )

    # Pre-allocate
    pseudo_locations = []

    midpoints = []
    ds = []

    for ii, source in enumerate(survey.source_list):
        src_loc = source.location
        if isinstance(src_loc, list):
            src_midpoint = (src_loc[0] + src_loc[1])/2
        else:
            src_midpoint = src_loc
        src_midpoint = src_midpoint.reshape((1, len(src_midpoint)))

        for receiver in source.receiver_list:
            rx_locs = receiver.locations
            if isinstance(rx_locs, list):
                rx_midpoints = (rx_locs[0] + rx_locs[1])/2
            else:
                rx_midpoints = rx_locs
            n_loc = rx_midpoints.shape[0]

            # Midpoint locations
            midpoints.append(
                (np.tile(src_midpoint, (n_loc, 1)) + rx_midpoints)/2
            )

            # Vector path from source midpoint to receiver midpoints
            ds.append(
                (rx_midpoints - np.tile(src_midpoint, (n_loc, 1)))
            )
    
    midpoints = np.vstack(midpoints)
    ds = np.vstack(ds)
    pseudo_depth = np.zeros_like(midpoints)

    # wenner-like electrode groups (are source and rx midpoints in same place)
    is_wenner = np.sqrt(np.sum(ds[:,:-1]**2, axis=1)) < 0.1

    # Pseudo depth is AB/2
    if np.any(is_wenner):
        temp = np.abs(electrode_separations(survey, ['AB'])['AB'])/2
        pseudo_depth[is_wenner, -1] = temp[is_wenner]

    # Takes into account topography.
    if np.any(~is_wenner):
        L = np.sqrt(np.sum(ds[~is_wenner, :]**2, axis=1))/2
        dz = ds[~is_wenner, -1]
        pseudo_depth[~is_wenner, 0] = (dz/2) * (ds[~is_wenner, 0] / L)
        if np.shape(ds)[1] > 2:
            pseudo_depth[~is_wenner, 1] = (dz/2) * (ds[~is_wenner, 1] / L)
        pseudo_depth[~is_wenner, -1] = np.sqrt(np.sum(ds[~is_wenner,:-1]**2, axis=1))/2

    return midpoints-pseudo_depth


def geometric_factor(survey_object, space_type="half_space", **kwargs):
    """
        Calculate Geometric Factor. Assuming that data are normalized voltages

        Input:
        :param SimPEG.electromagnetics.static.resistivity.Survey dc_survey: DC survey object
        :param str survey_type: Either 'dipole-dipole' | 'pole-dipole'
                               | 'dipole-pole' | 'pole-pole'
        :param str space_type: Assuming whole-space or half-space
                              ('whole-space' | 'half-space')

        Output:
        :return numpy.ndarray G: Geometric Factor

    """
    if "survey_type" in kwargs:
        warnings.warn(
            "The survey_type is no longer necessary to calculate geometric factor. "
            "Feel free to remove it from the call. This option will be removed in SimPEG 0.15.0",
            DeprecationWarning,
        )
    # Set factor for whole-space or half-space assumption
    if space_type.lower() in SPACE_TYPES["whole space"]:
        spaceFact = 4.0
    elif space_type.lower() in SPACE_TYPES["half space"]:
        spaceFact = 2.0
    else:
        raise Exception("'space_type must be 'whole space' | 'half space'")

    elecSepDict = electrode_separations(
        survey_object, electrode_pair=["AM", "BM", "AN", "BN"]
    )
    AM = elecSepDict["AM"]
    BM = elecSepDict["BM"]
    AN = elecSepDict["AN"]
    BN = elecSepDict["BN"]

    # Determine geometric factor G based on electrode separation distances.
    # For case where source and/or receivers are pole, terms will be
    # divided by infinity.
    G = 1 / AM - 1 / BM - 1 / AN + 1 / BN

    return G / (spaceFact * np.pi)

def apparent_resistivity_from_voltage(survey, volts, space_type="half_space", eps=1e-10):
    """
    Calculate apparent resistivities from normalized voltages.

    Input:
    :param SimPEG.electromagnetics.static.resistivity.Survey: DC survey
    :param numpy.ndarray volts: normalized voltage measurements [V/A]
    :param String space_type: 'half_space' or 'whole_space'
    :param float eps: Regularizer in case of a null geometric factor

    Output:
    :return rhoApp: apparent resistivity
    """

    G = geometric_factor(survey, space_type=space_type)

    # Calculate apparent resistivity
    # absolute value is required because of the regularizer
    rhoApp = np.abs(volts * (1.0 / (G + eps)))

    return rhoApp


def convert_survey_3d_to_2d_lines(survey, lineID, data_type='volt', output_indexing=False):
    """
    Convert a 3D survey into a list of local 2D surveys.

    Here, the user provides a Survey whose geometry is defined
    for use in a 3D simulation and a 1D numpy.array which defines the
    line ID for each datum. The function returns a list of local
    2D survey objects. The change of coordinates for electrodes is
    [x, y, z] to [s, z], where s is the distance along the profile
    line. For each line, s = 0 defines the A-electrode location
    for the first source in the source list.

    Input:
    :param survey: DC survey class object
    :param lineID: A numpy.array (nD,) containing the line ID for each datum

    Output:
    :param survey: List of 2D DC survey class object
    :rtype: List of SimPEG.electromagnetics.static.resistivity.Survey
    """
    

    # Find all unique line id
    unique_lineID = np.unique(lineID)
    
    # If you output indexing to keep track of possible sorting
    if output_indexing:
        k = np.arange(0, survey.nD)
        out_indices_list = []
    
    ab_locs_all = np.c_[survey.a_locations, survey.b_locations]
    mn_locs_all = np.c_[survey.m_locations, survey.n_locations]
    
    # For each unique lineID
    survey_list = []
    for ID in unique_lineID:
        
        source_list = []
        
        # Source locations for this line
        lineID_index = np.where(lineID == ID)[0]
        ab_locs, ab_index = np.unique(ab_locs_all[lineID_index, :], axis=0, return_index=True)
        
        # Find s=0 location and heading for line
        start_index = lineID_index[ab_index]
        if output_indexing:
            out_indices = []
            kID = k[lineID_index]                     # data indices part of this line
        r0 = mkvc(ab_locs_all[start_index[0], 0:2])   # (x0, y0) for the survey line
        rN = mkvc(ab_locs_all[start_index[-1], 0:2])  # (x, y) for last electrode
        uvec = (rN - r0)/np.sqrt(np.sum((rN-r0)**2))  # unit vector for line orientation
        
        # Along line positions and elevation for electrodes on current line
        # in terms of position elevation
        a_locs_s = np.c_[
            np.dot(ab_locs_all[lineID_index, 0:2]-r0[0], uvec), ab_locs_all[lineID_index, 2]
        ]
        b_locs_s = np.c_[
            np.dot(ab_locs_all[lineID_index, 3:5]-r0[0], uvec), ab_locs_all[lineID_index, -1]
        ]
        m_locs_s = np.c_[
            np.dot(mn_locs_all[lineID_index, 0:2]-r0[0], uvec), mn_locs_all[lineID_index, 2]
        ]
        n_locs_s = np.c_[
            np.dot(mn_locs_all[lineID_index, 3:5]-r0[0], uvec), mn_locs_all[lineID_index, -1]
        ]
        
        # For each source in the line
        for ii, ind in enumerate(ab_index):
            
            # Get source location
            src_loc_a = mkvc(a_locs_s[ind, :])
            src_loc_b = mkvc(b_locs_s[ind, :])
            
            # Get receiver locations
            rx_index = np.where(
                np.isclose(a_locs_s[:, 0], src_loc_a[0], atol=1e-3) & np.isclose(b_locs_s[:, 0], src_loc_b[0], atol=1e-3)
            )[0]
            rx_loc_m = m_locs_s[rx_index, :]
            rx_loc_n = n_locs_s[rx_index, :]
            
            if output_indexing:
                out_indices.append(kID[rx_index])
            
            # Define Pole or Dipole Receivers
            if np.all(np.isclose(rx_loc_m[:, 0], rx_loc_n[:, 0], atol=1e-3)):
                rx_list = [dc.receivers.Pole(rx_loc_m)]
            elif np.all(np.isclose(rx_loc_m[:, 0], rx_loc_n[:, 0], atol=1e-3)==False):
                rx_list = [dc.receivers.Dipole(rx_loc_m, rx_loc_n)]
            else:
                raise NotImplementedError("An individual source cannot have a mix of Pole and Dipole receivers")
            
            # Define Pole or Dipole Sources
            if np.all(np.isclose(src_loc_a, src_loc_b, atol=1e-3)):
                source_list.append(dc.sources.Pole(rx_list, src_loc_a))
            else:
                source_list.append(
                    dc.sources.Dipole(rx_list, src_loc_a, src_loc_b)
                )
            
        # Create a 2D survey and add to list
        survey_list.append(dc.survey.Survey(source_list))
        if output_indexing:
            out_indices_list.append(np.hstack(out_indices))
    
    if output_indexing:
        return survey_list, out_indices_list
    else:
        return survey_list


#####################################################################
#                               PLOTTING
#####################################################################


def plot_1d_layer_model(
    thicknesses, values, z0=0, scale="log", ax=None, plot_elevation=False, show_layers=False, **kwargs
):
    """
    Plot the vertical conductivity or resistivity profile for a 1D layered Earth model.
    
    Input:
    thicknesses : List[Float]
        A list or numpy.array containing the layer thicknesses from the top layer down
    values : List[Float]
        A list or numpy.array containing the physical property values from the top layer down
    z0 : Float
        Elevation of the surface
    scale: str
        scale {'linear', 'log'}. Plot physical property values on a linear or log10 scale.
    ax: mpl_toolkits.mplot3d.axes3d.Axes3D, optional
        A 3D axis object for the 3D plot
    plot_elevation : bool
        If False, the yaxis will be the depth. If True, the yaxis is the elevation.
    show_layers : bool
        Plot horizontal lines to denote layers.
    line_opts : dict
        Dictionary defining kwargs for scatter plot if plot_type='scatter'
    

    Output:
    mpl_toolkits.mplot3d.axes3d.Axes3D
        The axis object that holds the plot

    """

    if np.median(values) > 1.:
        x_label = "Resistivity ($\Omega$m)"
    else:
        x_label = "Conductivity (S/m)"

    if len(thicknesses) < len(values):
        thicknesses = np.r_[thicknesses, thicknesses[-1]]
    z_grid = np.r_[0., np.cumsum(thicknesses)]
    resistivity = np.repeat(values, 2)
    rho_min = 0.9*np.min(values)
    rho_max = 1.1*np.max(values)

    z = []
    for i in range(0, len(thicknesses)):
        z.append(np.r_[z_grid[i], z_grid[i + 1]])
    z = np.hstack(z)
    
    if plot_elevation:
        y_label = "Elevation (m)"
        z = z0 - z
        z_grid = z0 - z_grid
        flip_axis = False
    else:
        y_label = "Depth (m)"
        flip_axis = True
        
    if ax == None:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_axes([0.15, 0.15, 0.75, 0.75])
    
    if show_layers:
        for locz in z_grid:
            plt.plot(
                np.linspace(rho_min, rho_max, 100),
                np.ones(100) * locz,
                "k--",
                lw=0.5,
                label='_nolegend_'
            )

    ax.plot(resistivity, z, **kwargs)
    ax.set_xscale(scale)
    ax.set_xlim(rho_min, rho_max)
    if flip_axis:
        ax.set_ylim(z.max(), z.min())
    else:
        ax.set_ylim(z.min(), z.max())
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    
    return ax


def plot_2d_pseudosection(
    survey,
    dvec,
    plot_type,
    ax=None,
    cax=None,
    vlim=None,
    scale="linear",
    units="",
    create_colorbar=True,
    mask_topography=False,
    marker_size=40,
    scatter_opts={},
    tricontourf_opts={},
    cbar_opts={}
):
    """
    Plot 2D DC/IP data in pseudo-section.

    This utility allows the user to image 2D DC/IP data in pseudosection as
    either a scatter plot or as a filled contour plot.

    Input:
    survey : SimPEG.electromagnetics.static.survey.Survey
        A DC or IP survey object defining a 2D survey line
    dvec : numpy.ndarray (ndata,)
        A data vector containing volts, integrated chargeabilities, apparent
        resistivities, apparent chargeabilities or data misfits.
    plot_type: str
        Plot type {'scatter', 'tricontourf'}. 'scatter' creates a scatter plot
        and 'tricontourf' creates a filled contour plot.
    ax: mpl_toolkits.mplot3d.axes3d.Axes3D, optional
        A 3D axis object for the 3D plot
    cax : mpl_toolkits.mplot3d.axes.Axes or mpl_toolkits.mplot3d.axes3d.Axes3D, optional
        An axis object for the colorbar
    vlim : list
        list containing the minimum and maximum value for the color range,
        i.e. [vmin, vmax]
    scale: str
        Plot on linear or log base 10 scale {'linear','log'}
    units : str
        A LateX formatted string stating the desired units for the
        data; e.g. 'S/m', '$\Omega m$', '%'
    create_colorbar : bool
        If *True*, a colorbar is automatically generated. If *False*, it is not.
        If multiple planes are being plotted, only set the first scatter plot
        to *True*
    mask_topography : bool
        This freature should be set to True when there is significant topography and the user
        would like to mask interpolated locations in the filled contour plot that lie
        above the surface topography.
    marker_size : int
        If plot_type=='scatter', this argument can be used to set the marker size for
        the scatter plot.
    scatter_opts : dict
        Dictionary defining kwargs for scatter plot if plot_type='scatter'
    tricontourf_opts : dict
        Dictionary defining kwargs for filled contour plot if plot_type='tricontourf'
    cbar_opts : dict
        Dictionary defining kwargs for the colorbar
    

    Output:
    mpl_toolkits.mplot3d.axes3d.Axes3D
        The axis object that holds the plot

    """

    # Get plotting locations from survey geometry
    locations = pseudo_locations(survey)

    # Log or linear scale
    if scale == "log":
        dvec = np.log10(dvec)
        if vlim != None:
            vlim[0] = np.log10(vlim[0])
            vlim[1] = np.log10(vlim[1])

    # Create an axis for the pseudosection if None
    if ax == None:
        fig = plt.figure(figsize=(10, 4))
        ax = fig.add_axes([0.1, 0.1, 0.7, 0.8])
        cax = fig.add_axes([0.85, 0.1, 0.03, 0.8])

    if vlim == None:
        norm = mpl.colors.Normalize(vmin=dvec.min(), vmax=dvec.max())
    else:
        norm = mpl.colors.Normalize(vmin=vlim[0], vmax=vlim[1])

    # Scatter plot
    if plot_type == 'scatter':
        data_plot = ax.scatter(
            locations[:, 0],
            locations[:, -1],
            marker_size,
            c=dvec,
            norm=norm,
            **scatter_opts
        )

    # Filled contour plot
    elif plot_type == 'tricontourf':

        # Make initial contour plot
        data_plot = ax.tricontourf(
            locations[:, 0],
            locations[:, -1],
            dvec,
            norm=norm,
            **tricontourf_opts,
        )

    else:
        raise NotImplementedError("plot_type must be 'scatter' or 'tricontourf'")
        
    # Use a filled polygon to mask everything above
    # that has a pseudo-location above the positions
    # for nearest electrode spacings    
    
    if mask_topography:
        
        electrode_locations = np.unique(np.r_[
            survey.locations_a,
            survey.locations_b,
            survey.locations_m,
            survey.locations_n
        ], axis=0)
        
        zmin = np.min(electrode_locations[:, 1])
        zmax = np.max(electrode_locations[:, 1])

        tree = cKDTree(locations)
        _, nodeInds = tree.query(electrode_locations)

        poly_locations = locations[nodeInds, :]

        poly_locations = np.r_[
            np.c_[np.min(poly_locations[:, 0]), zmax],
            poly_locations,
            np.c_[np.max(poly_locations[:, 0]), zmax]
        ]

        ax.fill(
            poly_locations[:, 0], poly_locations[:, 1],
            facecolor='w', linewidth=0.5
        )    
    
    z_top = np.max(locations[:, -1])
    z_bot = np.min(locations[:, -1])
    ax.set_ylim(z_bot - 0.03*(z_top-z_bot), z_top + 0.03*(z_top-z_bot))
    ax.set_xlabel("Line position (m)")
    ax.set_ylabel("Pseudo-elevation (m)")

    # Define colorbar
    if create_colorbar:
        if cax == None:
            if scale == "log":
                cbar = plt.colorbar(
                    data_plot,
                    format="$10^{%.2f}$",
                    fraction=0.06,
                    orientation="vertical",
                    ax=ax,
                    **cbar_opts,
                )
            elif scale == "linear":
                cbar = plt.colorbar(
                    data_plot,
                    format="%.2e",
                    fraction=0.06,
                    orientation="vertical",
                    ax=ax,
                    **cbar_opts,
                )

        else:
            if scale == "log":
                cbar = plt.colorbar(
                    data_plot, format="$10^{%.2f}$", cax=cax, **cbar_opts,
                )
            elif scale == "linear":
                cbar = plt.colorbar(
                    data_plot, format="%.2e", cax=cax, **cbar_opts,
                )

        ticks = np.linspace(norm.vmin, norm.vmax, 7)

        cbar.set_ticks(ticks)
        cbar.set_label(units, labelpad=10)
        cbar.ax.tick_params()

    return ax


def plot_3d_pseudosection(
    survey,
    dvec,
    ax=None,
    cax=None,
    marker_size=50,
    vlim=None,
    scale="linear",
    units="",
    plane_points=None,
    plane_distance=10.0,
    create_colorbar=True,
    scatter_opts={},
    cbar_opts={},
):
    """
    Plot 3D DC/IP data in pseudo-section as a scatter plot.

    This utility allows the user to produce a scatter plot of 3D DC/IP data at
    all pseudo-locations. If a plane is specified, the user may create a scatter
    plot using points near that plane.

    Input:
    survey : SimPEG.electromagnetics.static.survey.Survey
        A DC or IP survey object
    dvec : numpy.ndarray
        A data vector containing volts, integrated chargeabilities, apparent
        resistivities or apparent chargeabilities.
    ax: mpl_toolkits.mplot3d.axes3d.Axes3D, optional
        A 3D axis object for the 3D plot
    cax : mpl_toolkits.mplot3d.axes.Axes or mpl_toolkits.mplot3d.axes3d.Axes3D, optional
        An axis object for the colorbar
    marker_size : int
        Sets the marker size for the points on the scatter plot
    vlim : list
        list containing the minimum and maximum value for the color range,
        i.e. [vmin, vmax]
    scale: str
        Plot on linear or log base 10 scale {'linear','log'}
    units : str
        A LateX formatted string stating the desired units for the
        data; e.g. 'S/m', '$\Omega m$', '%'
    plane_points : list of numpy.ndarray
        A list of length 3 which contains the three xyz locations required to
        define a plane; i.e. [xyz1, xyz2, xyz3]. This functionality is used to
        plot only data that lie near this plane. A list of [xyz1, xyz2, xyz3]
        can be entered for multiple planes.
    plane_distance : float or list of float
        Distance tolerance for plotting data that are near the plane(s) defined by
        **plane_points**. A list is used if the *plane_distance* is different
        for each plane.
    create_colorbar : bool
        If *True*, a colorbar is automatically generated. If *False*, it is not.
        If multiple planes are being plotted, only set the first scatter plot
        to *True*
    scatter_opts : dict
        Dictionary defining kwargs for the scatter plot
    cbar_opts : dict
        Dictionary defining kwargs for the colorbar
    

    Output:
    mpl_toolkits.mplot3d.axes3d.Axes3D
        The axis object that holds the plot

    """

    locations = pseudo_locations(survey)

    if scale == "log":
        plot_vec = np.log10(dvec)
        if vlim != None:
            vlim[0] = np.log10(vlim[0])
            vlim[1] = np.log10(vlim[1])
    else:
        plot_vec = dvec

    if ax == None:
        fig = plt.figure(figsize=(10, 4))
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], projection="3d", azim=-60, elev=30)
        cax = fig.add_axes([0.85, 0.1, 0.05, 0.8])

    # 3D scatter plot
    if plane_points == None:

        if vlim == None:
            norm = mpl.colors.Normalize(vmin=plot_vec.min(), vmax=plot_vec.max())
        else:
            norm = mpl.colors.Normalize(vmin=vlim[0], vmax=vlim[1])

        data_plot = ax.scatter(
            locations[:, 0],
            locations[:, 1],
            locations[:, 2],
            s=s,
            c=plot_vec,
            edgecolors='none',
            depthshade=False,
            norm=norm,
            **scatter_opts,
        )
    else:
        # Place in list if only one plane defined
        if isinstance(plane_points[0], np.ndarray):
            plane_points = [plane_points]

        # Expand to list of only one plane distance for all planes
        if isinstance(plane_distance, list) != True:
            plane_distance = len(plane_points) * [plane_distance]

        # Pre-allocate index for points on plane(s)
        k = np.zeros(len(plot_vec), dtype=bool)
        for ii in range(0, len(plane_points)):

            p1, p2, p3 = plane_points[ii]
            a, b, c, d = define_plane_from_points(p1, p2, p3)

            k = k | (
                np.abs(a * locations[:, 0] + b * locations[:, 1] + c * locations[:, 2] + d)
                / np.sqrt(a ** 2 + b ** 2 + c ** 2)
                < plane_distance[ii]
            )

        if np.all(k == 0):
            raise Exception(
                """No locations are within *plane_distance* of any plane(s)
                defined by *plane_points*. Try increasing *plane_distance*."""
            )

        if vlim == None:
            norm = mpl.colors.Normalize(vmin=plot_vec[k].min(), vmax=plot_vec[k].max())
        else:
            norm = mpl.colors.Normalize(vmin=vlim[0], vmax=vlim[1])

        data_plot = ax.scatter(
            locations[k, 0],
            locations[k, 1],
            locations[k, 2],
            s=marker_size,
            c=plot_vec[k],
            edgecolors='none',
            depthshade=False,
            norm=norm,
            **scatter_opts,
        )

    # Define colorbar
    if create_colorbar:
        if cax == None:
            if scale == "log":
                cbar = plt.colorbar(
                    data_plot,
                    format="$10^{%.2f}$",
                    fraction=0.06,
                    orientation="vertical",
                    ax=ax,
                    shrink=0.7,
                    **cbar_opts,
                )
            elif scale == "linear":
                cbar = plt.colorbar(
                    data_plot,
                    format="%.2e",
                    fraction=0.06,
                    orientation="vertical",
                    ax=ax,
                    shrink=0.7,
                    **cbar_opts,
                )

        else:
            if scale == "log":
                cbar = plt.colorbar(
                    data_plot, format="$10^{%.2f}$", cax=cax, **cbar_opts,
                )
            elif scale == "linear":
                cbar = plt.colorbar(
                    data_plot, format="%.2e", cax=cax, **cbar_opts,
                )

        ticks = np.linspace(norm.vmin, norm.vmax, 5)

        cbar.set_ticks(ticks)
        cbar.set_label(units, labelpad=12)
        cbar.ax.tick_params()

    return ax



#########################################################################
#                      GENERATE SURVEYS
#########################################################################

# def generate_dcip_survey(endl, survey_type, a, b, n, dim=3, **kwargs):

#     """
#         Load in endpoints and survey specifications to generate Tx, Rx location
#         stations.

#         Assumes flat topo for now...

#         Input:
#         :param numpy.ndarray endl: input endpoints [x1, y1, z1, x2, y2, z2]
#         :param discretize.base.BaseMesh mesh: discretize mesh object
#         :param str survey_type: 'dipole-dipole' | 'pole-dipole' |
#             'dipole-pole' | 'pole-pole' | 'gradient'
#         :param int a: pole seperation
#         :param int b: dipole separation
#         :param int n: number of rx dipoles per tx

#         Output:
#         :return SimPEG.electromagnetics.static.resistivity.Survey dc_survey: DC survey object
#     """
#     if "d2flag" in kwargs:
#         warnings.warn(
#             "The d2flag is no longer necessary to construct a survey. "
#             "Feel free to remove it from the call. This option will be removed in SimPEG 0.15.0",
#             DeprecationWarning,
#         )

#     def xy_2_r(x1, x2, y1, y2):
#         r = np.sqrt(np.sum((x2 - x1) ** 2.0 + (y2 - y1) ** 2.0))
#         return r

#     # Evenly distribute electrodes and put on surface
#     # Mesure survey length and direction
#     dl_len = xy_2_r(endl[0, 0], endl[1, 0], endl[0, 1], endl[1, 1])

#     dl_x = (endl[1, 0] - endl[0, 0]) / dl_len
#     dl_y = (endl[1, 1] - endl[0, 1]) / dl_len

#     nstn = int(np.floor(dl_len / a))

#     # Compute discrete pole location along line
#     stn_x = endl[0, 0] + np.array(range(int(nstn))) * dl_x * a
#     stn_y = endl[0, 1] + np.array(range(int(nstn))) * dl_y * a

#     if dim == 2:
#         ztop = np.linspace(endl[0, 1], endl[0, 1], nstn)
#         # Create line of P1 locations
#         M = np.c_[stn_x, ztop]
#         # Create line of P2 locations
#         N = np.c_[stn_x + a * dl_x, ztop]

#     elif dim == 3:
#         stn_z = np.linspace(endl[0, 2], endl[0, 2], nstn)
#         # Create line of P1 locations
#         M = np.c_[stn_x, stn_y, stn_z]
#         # Create line of P2 locations
#         N = np.c_[stn_x + a * dl_x, stn_y + a * dl_y, stn_z]

#     # Build list of Tx-Rx locations depending on survey type
#     # Dipole-dipole: Moving tx with [a] spacing -> [AB a MN1 a MN2 ... a MNn]
#     # Pole-dipole: Moving pole on one end -> [A a MN1 a MN2 ... MNn a B]
#     SrcList = []

#     if survey_type != "gradient":

#         for ii in range(0, int(nstn) - 1):

#             if survey_type.lower() in ["dipole-dipole", "dipole-pole"]:
#                 tx = np.c_[M[ii, :], N[ii, :]]
#                 # Current elctrode separation
#                 AB = xy_2_r(tx[0, 1], endl[1, 0], tx[1, 1], endl[1, 1])
#             elif survey_type.lower() in ["pole-dipole", "pole-pole"]:
#                 tx = np.r_[M[ii, :]]
#                 # Current elctrode separation
#                 AB = xy_2_r(tx[0], endl[1, 0], tx[1], endl[1, 1])
#             else:
#                 raise Exception(
#                     "survey_type must be 'dipole-dipole' | 'pole-dipole' | "
#                     "'dipole-pole' | 'pole-pole' not {}".format(survey_type)
#                 )

#             # Rx.append(np.c_[M[ii+1:indx, :], N[ii+1:indx, :]])

#             # Number of receivers to fit
#             nstn = int(np.min([np.floor((AB - b) / a), n]))

#             # Check if there is enough space, else break the loop
#             if nstn <= 0:
#                 continue

#             # Compute discrete pole location along line
#             stn_x = N[ii, 0] + dl_x * b + np.array(range(int(nstn))) * dl_x * a
#             stn_y = N[ii, 1] + dl_y * b + np.array(range(int(nstn))) * dl_y * a

#             # Create receiver poles

#             if dim == 3:
#                 stn_z = np.linspace(endl[0, 2], endl[0, 2], nstn)

#                 # Create line of P1 locations
#                 P1 = np.c_[stn_x, stn_y, stn_z]
#                 # Create line of P2 locations
#                 P2 = np.c_[stn_x + a * dl_x, stn_y + a * dl_y, stn_z]
#                 if survey_type.lower() in ["dipole-dipole", "pole-dipole"]:
#                     rxClass = dc.Rx.Dipole(P1, P2)
#                 elif survey_type.lower() in ["dipole-pole", "pole-pole"]:
#                     rxClass = dc.Rx.Pole(P1)

#             elif dim == 2:
#                 ztop = np.linspace(endl[0, 1], endl[0, 1], nstn)
#                 # Create line of P1 locations
#                 P1 = np.c_[stn_x, np.ones(nstn).T * ztop]
#                 # Create line of P2 locations
#                 P2 = np.c_[stn_x + a * dl_x, np.ones(nstn).T * ztop]
#                 if survey_type.lower() in ["dipole-dipole", "pole-dipole"]:
#                     rxClass = dc.Rx.Dipole(P1, P2)
#                 elif survey_type.lower() in ["dipole-pole", "pole-pole"]:
#                     rxClass = dc.Rx.Pole(P1)

#             if survey_type.lower() in ["dipole-dipole", "dipole-pole"]:
#                 srcClass = dc.Src.Dipole([rxClass], M[ii, :], N[ii, :])
#             elif survey_type.lower() in ["pole-dipole", "pole-pole"]:
#                 srcClass = dc.Src.Pole([rxClass], M[ii, :])
#             SrcList.append(srcClass)

#     elif survey_type.lower() == "gradient":

#         # Gradient survey takes the "b" parameter to define the limits of a
#         # square survey grid. The pole seperation within the receiver grid is
#         # define the "a" parameter.

#         # Get the edge limit of survey area
#         min_x = endl[0, 0] + dl_x * b
#         min_y = endl[0, 1] + dl_y * b

#         max_x = endl[1, 0] - dl_x * b
#         max_y = endl[1, 1] - dl_y * b

#         # Define the size of the survey grid (square for now)
#         box_l = np.sqrt((min_x - max_x) ** 2.0 + (min_y - max_y) ** 2.0)
#         box_w = box_l / 2.0

#         nstn = int(np.floor(box_l / a))

#         # Compute discrete pole location along line
#         stn_x = min_x + np.array(range(int(nstn))) * dl_x * a
#         stn_y = min_y + np.array(range(int(nstn))) * dl_y * a

#         # Define number of cross lines
#         nlin = int(np.floor(box_w / a))
#         lind = range(-nlin, nlin + 1)

#         npoles = int(nstn * len(lind))

#         rx = np.zeros([npoles, 6])
#         for ii in range(len(lind)):

#             # Move station location to current survey line This is a
#             # perpendicular move then line survey orientation, hence the y, x
#             # switch
#             lxx = stn_x - lind[ii] * a * dl_y
#             lyy = stn_y + lind[ii] * a * dl_x

#             M = np.c_[lxx, lyy, np.ones(nstn).T * ztop]
#             N = np.c_[lxx + a * dl_x, lyy + a * dl_y, np.ones(nstn).T * ztop]
#             rx[(ii * nstn) : ((ii + 1) * nstn), :] = np.c_[M, N]

#             if dim == 3:
#                 rxClass = dc.Rx.Dipole(rx[:, :3], rx[:, 3:])
#             elif dim == 2:
#                 M = M[:, [0, 2]]
#                 N = N[:, [0, 2]]
#                 rxClass = dc.Rx.Dipole(rx[:, [0, 2]], rx[:, [3, 5]])
#             srcClass = dc.Src.Dipole([rxClass], (endl[0, :]), (endl[1, :]))
#         SrcList.append(srcClass)
#         survey_type = "dipole-dipole"

#     survey = dc.Survey(SrcList, survey_type=survey_type.lower())
#     return survey


def generate_dcip_sources_line(
    survey_type,
    data_type,
    dimension_type,
    end_points,
    topo,
    num_rx_per_src,
    station_spacing,
):
    """
    Generate the source list for a 2D or 3D DC/IP survey line.

    This utility will create the list of DC/IP source objects for a single line of
    2D or 3D data. The topography, orientation, spacing and number of receivers
    can be specified by the user. This function can be used to define multiple lines
    of DC/IP, which can be appended to create the sources for an entire survey.

    Input:
    :param str survey_type: 'dipole-dipole' | 'pole-dipole' |
        'dipole-pole' | 'pole-pole'
    :param str data_type: 'volt' | 'apparent_conductivity' |
        'apparent_resistivity' | 'apparent_chargeability'
    :param str dimension_type: '2D' or '3D'
    :param np.array end_points: horizontal end points [x1, x2] or [x1, x2, y1, y2]
    :param float, (N, 2) np.array for 2D or (N, 3) np.array for 3D: topography
    :param int num_rx_per_src: number of receivers per souces
    :param float station_spacing : distance between stations

    Output:
    :return SimPEG.electromagnetics.static.resistivity.Survey dc_survey: DC survey object
    """

    assert survey_type.lower() in [
        "pole-pole",
        "pole-dipole",
        "dipole-pole",
        "dipole-dipole",
    ], "survey_type must be one of 'pole-pole', 'pole-dipole', 'dipole-pole', 'dipole-dipole'"

    assert data_type.lower() in [
        "volt",
        "apparent_conductivity",
        "apparent_resistivity",
        "apparent_chargeability",
    ], "data_type must be one of 'volt', 'apparent_conductivity', 'apparent_resistivity', 'apparent_chargeability'"

    assert dimension_type.upper() in [
        "2D",
        "3D",
    ], "dimension_type must be one of '2D' or '3D'"

    def xy_2_r(x1, x2, y1, y2):
        r = np.sqrt(np.sum((x2 - x1) ** 2.0 + (y2 - y1) ** 2.0))
        return r

    # Compute horizontal locations of sources and receivers
    x1 = end_points[0]
    x2 = end_points[1]

    if dimension_type == "3D":

        # Station locations
        y1 = end_points[2]
        y2 = end_points[3]
        L = xy_2_r(x1, x2, y1, y2)
        nstn = int(np.floor(L / station_spacing) + 1)
        dl_x = (x2 - x1) / L
        dl_y = (y2 - y1) / L
        stn_x = x1 + np.array(range(int(nstn))) * dl_x * station_spacing
        stn_y = y1 + np.array(range(int(nstn))) * dl_y * station_spacing

        # Station xyz locations
        P = np.c_[stn_x, stn_y]
        if np.size(topo) == 1:
            P = np.c_[P, topo * np.ones((nstn))]
        else:
            fun_interp = LinearNDInterpolator(topo[:, 0:2], topo[:, -1])
            P = np.c_[P, fun_interp(P)]

    else:

        # Station locations
        y1 = 0.0
        y2 = 0.0
        L = xy_2_r(x1, x2, y1, y2)
        nstn = int(np.floor(L / station_spacing) + 1)
        stn_x = x1 + np.array(range(int(nstn))) * station_spacing

        # Station xyz locations
        if np.size(topo) == 1:
            P = np.c_[stn_x, topo * np.ones((nstn))]
        else:
            fun_interp = interp1d(topo[:, 0], topo[:, -1])
            P = np.c_[stn_x, fun_interp(stn_x)]

    # Build list of Tx-Rx locations depending on survey type
    # Dipole-dipole: Moving tx with [a] spacing -> [AB a MN1 a MN2 ... a MNn]
    # Pole-dipole: Moving pole on one end -> [A a MN1 a MN2 ... MNn a B]
    source_list = []

    if survey_type.lower() == "pole-pole":
        rx_shift = 0
    elif survey_type.lower() in ["pole-dipole", "dipole-pole"]:
        rx_shift = 1
    elif survey_type.lower() == "dipole-dipole":
        rx_shift = 2

    for ii in range(0, int(nstn - rx_shift)):

        if dimension_type == "3D":
            D = xy_2_r(stn_x[ii + rx_shift], x2, stn_y[ii + rx_shift], y2)
        else:
            D = xy_2_r(stn_x[ii + rx_shift], x2, y1, y2)

        # Number of receivers to fit
        nrec = int(np.min([np.floor(D / station_spacing), num_rx_per_src]))

        # Check if there is enough space, else break the loop
        if nrec <= 0:
            continue

        # Create receivers
        if survey_type.lower() in ["dipole-pole", "pole-pole"]:
            rxClass = dc.receivers.Pole(
                P[ii + rx_shift + 1 : ii + rx_shift + nrec + 1, :], data_type=data_type
            )
        elif survey_type.lower() in ["dipole-dipole", "pole-dipole"]:
            rxClass = dc.receivers.Dipole(
                P[ii + rx_shift : ii + rx_shift + nrec, :],
                P[ii + rx_shift + 1 : ii + rx_shift + nrec + 1, :],
                data_type=data_type,
            )

        # Create sources
        if survey_type.lower() in ["pole-dipole", "pole-pole"]:
            srcClass = dc.sources.Pole([rxClass], P[ii, :])
        elif survey_type.lower() in ["dipole-dipole", "dipole-pole"]:
            srcClass = dc.sources.Dipole([rxClass], P[ii, :], P[ii + 1, :])

        source_list.append(srcClass)

    return source_list



########################################################################

def writeUBC_DCobs(
    fileName,
    data,
    dim,
    format_type,
    survey_type="dipole-dipole",
    ip_type=0,
    comment_lines="",
):
    """
    Write UBC GIF DCIP 2D or 3D observation file

    Input:
    :param str fileName: including path where the file is written out
    :param SimPEG.Data data: DC data object
    :param int dim:  either 2 | 3
    :param str format_type:  either 'surface' | 'general' | 'simple'
    :param str survey_type: 'dipole-dipole' | 'pole-dipole' |
        'dipole-pole' | 'pole-pole' | 'gradient'

    Output:
    :return: UBC2D-Data file
    :rtype: file
    """

    if not isinstance(data, Data):
        raise Exception(
            "A Data instance ({datacls}: <{datapref}.{datacls}>) must be "
            "provided as the second input. The provided input is a "
            "{providedcls} <{providedpref}.{providedcls}>".format(
                datacls=Data.__name__,
                datapref=Data.__module__,
                providedcls=data.__class__.__name__,
                providedpref=data.__module__,
            )
        )

    if not ((dim == 2) | (dim == 3)):
        raise Exception("""dim must be either 2 or 3""" " not {}".format(dim))

    format_type = format_type.lower()
    if format_type not in ["surface", "general", "simple"]:
        raise Exception(
            "format_type must be 'surface' | 'general' | 'simple' "
            " not {}".format(format_type)
        )

    # if(isinstance(dc_survey.std, float)):
    #     print(
    #         """survey.std was a float computing standard_deviation vector
    #         (survey.std*survey.dobs + survey.eps)"""
    #     )

    # if(isinstance(dc_survey.eps, float)):
    #     epsValue = dc_survey.eps
    #     dc_survey.eps = epsValue*np.ones_like(dc_survey.dobs)

    fid = open(fileName, "w")

    if format_type.lower() in ["surface", "general"] and dim == 2:
        fid.write("COMMON_CURRENT\n")

    fid.write("! " + format_type + " FORMAT\n")

    if comment_lines:
        fid.write(comment_lines)

    if dim == 2:
        fid.write("{:d}\n".format(data.survey.nSrc))

    if ip_type != 0:
        fid.write("IPTYPE=%i\n" % ip_type)

    fid.close()

    count = 0

    for src in data.survey.source_list:

        rx = src.receiver_list[0].locations
        nD = src.nD

        if survey_type.lower() in ["pole-dipole", "pole-pole"]:
            tx = np.r_[src.location]
            tx = np.repeat(np.r_[[tx]], 2, axis=0)
        elif survey_type.lower() in ["dipole-dipole", "dipole-pole"]:
            tx = np.c_[src.location]

        if survey_type.lower() in ["pole-dipole", "dipole-dipole"]:
            M = rx[0]
            N = rx[1]
        elif survey_type.lower() in ["pole-pole", "dipole-pole"]:
            M = rx
            N = rx

        # Adapt source-receiver location for dim and survey_type
        if dim == 2:
            if format_type == "simple":
                # fid.writelines("%e " % ii for ii in mkvc(tx[0, :]))
                A = np.repeat(tx[0, 0], M.shape[0], axis=0)

                if survey_type == "pole-dipole":
                    B = np.repeat(tx[0, 0], M.shape[0], axis=0)

                else:
                    B = np.repeat(tx[1, 0], M.shape[0], axis=0)

                M = M[:, 0]
                N = N[:, 0]

                fid = open(fileName, "ab")
                np.savetxt(
                    fid,
                    np.c_[
                        A,
                        B,
                        M,
                        N,
                        data.dobs[count : count + nD],
                        data.relative_error[count : count + nD],
                    ],
                    delimiter=str(" "),
                    newline=str("\n"),
                )
                fid.close()

            else:
                fid = open(fileName, "a")
                if format_type == "surface":
                    fid.writelines("%f " % ii for ii in mkvc(tx[:, 0]))
                    M = M[:, 0]
                    N = N[:, 0]

                if format_type == "general":
                    # Flip sign for z-elevation to depth
                    tx[2::2, :] = -tx[2::2, :]

                    fid.writelines(
                        ("{:e} {:e} ").format(ii, jj) for ii, jj in tx[:, :2]
                    )
                    M = M[:, :2]
                    N = N[:, :2]

                    # Flip sign for z-elevation to depth
                    M[:, 1::2] = -M[:, 1::2]
                    N[:, 1::2] = -N[:, 1::2]

                fid.write("%i\n" % nD)
                fid.close()

                fid = open(fileName, "ab")
                np.savetxt(
                    fid,
                    np.c_[
                        M,
                        N,
                        data.dobs[count : count + nD],
                        data.relative_error[count : count + nD],
                    ],
                    delimiter=str(" "),
                    newline=str("\n"),
                )

        if dim == 3:
            fid = open(fileName, "a")
            # Flip sign of z value for UBC DCoctree code
            # tx[:, 2] = -tx[:, 2]
            # print(tx)

            # Flip sign of z value for UBC DCoctree code
            # M[:, 2] = -M[:, 2]
            # N[:, 2] = -N[:, 2]

            if format_type.lower() == "surface":

                fid.writelines("%e " % ii for ii in mkvc(tx[:, 0:2].T))
                M = M[:, 0:2]
                N = N[:, 0:2]

            if format_type.lower() == "general":

                fid.writelines("%e " % ii for ii in mkvc(tx.T))

            fid.write("%i\n" % nD)

            fid.close()

            fid = open(fileName, "ab")
            if isinstance(data.relative_error, np.ndarray):
                np.savetxt(
                    fid,
                    np.c_[
                        M,
                        N,
                        data.dobs[count : count + nD],
                        (
                            data.relative_error[count : count + nD]
                            + data.noise_floor[count : count + nD]
                        ),
                    ],
                    fmt=str("%e"),
                    delimiter=str(" "),
                    newline=str("\n"),
                )
            else:
                raise Exception(
                    """Uncertainities SurveyObject.std should be set.
                    Either float or nunmpy.ndarray is expected, """
                    "not {}".format(type(data.relative_error))
                )

            fid.close()
            fid = open(fileName, "a")
            fid.write("\n")
            fid.close()

        count += nD

    fid.close()


def writeUBC_DClocs(
    fileName,
    dc_survey,
    dim,
    format_type,
    survey_type="dipole-dipole",
    ip_type=0,
    comment_lines="",
):
    """
        Write UBC GIF DCIP 2D or 3D locations file

        Input:
        :param str fileName: including path where the file is written out
        :param SimPEG.electromagnetics.static.resistivity.Survey dc_survey: DC survey object
        :param int dim:  either 2 | 3
        :param str survey_type:  either 'SURFACE' | 'GENERAL'

        Output:
        :rtype: file
        :return: UBC 2/3D-locations file
    """

    if not ((dim == 2) | (dim == 3)):
        raise Exception("""dim must be either 2 or 3""" " not {}".format(dim))

    if format_type.lower() not in ["surface", "general", "simple"]:
        raise Exception(
            "format_type must be 'SURFACE' | 'GENERAL' | 'SIMPLE' "
            " not {}".format(format_type)
        )

    fid = open(fileName, "w")

    if format_type.lower() in ["surface", "general"] and dim == 2:
        fid.write("COMMON_CURRENT\n")

    fid.write("! " + format_type + " FORMAT\n")

    if comment_lines:
        fid.write(comment_lines)

    if dim == 2:
        fid.write("{:d}\n".format(dc_survey.nSrc))

    if ip_type != 0:
        fid.write("IPTYPE=%i\n" % ip_type)

    fid.close()

    count = 0

    for ii in range(len(dc_survey.source_list)):

        rx = dc_survey.source_list[ii].receiver_list[0].locations
        nD = dc_survey.source_list[ii].nD

        if survey_type.lower() in ["pole-dipole", "pole-pole"]:
            tx = np.r_[dc_survey.source_list[ii].locations]
            tx = np.repeat(np.r_[[tx]], 2, axis=0)
        elif survey_type.lower() in ["dipole-dipole", "dipole-pole"]:
            tx = np.c_[dc_survey.source_list[ii].locations]

        if survey_type.lower() in ["pole-dipole", "dipole-dipole"]:
            M = rx[0]
            N = rx[1]
        elif survey_type.lower() in ["pole-pole", "dipole-pole"]:
            M = rx
            N = rx

        # Adapt source-receiver location for dim and survey_type
        if dim == 2:

            if format_type.lower() == "simple":
                # fid.writelines("%e " % ii for ii in mkvc(tx[0, :]))
                A = np.repeat(tx[0, 0], M.shape[0], axis=0)

                if survey_type.lower() == "pole-dipole":
                    B = np.repeat(tx[0, 0], M.shape[0], axis=0)

                else:
                    B = np.repeat(tx[1, 0], M.shape[0], axis=0)

                M = M[:, 0]
                N = N[:, 0]

                fid = open(fileName, "ab")
                np.savetxt(
                    fid, np.c_[A, B, M, N], delimiter=str(" "), newline=str("\n")
                )
                fid.close()

            else:
                fid = open(fileName, "a")
                if format_type.lower() == "surface":

                    fid.writelines("%f " % ii for ii in mkvc(tx[:, 0]))
                    M = M[:, 0]
                    N = N[:, 0]

                if format_type.lower() == "general":

                    # Flip sign for z-elevation to depth
                    tx[2::2, :] = -tx[2::2, :]

                    fid.writelines(
                        ("{:e} {:e} ").format(ii, jj) for ii, jj in tx[:, :2]
                    )
                    M = M[:, :2]
                    N = N[:, :2]

                    # Flip sign for z-elevation to depth
                    M[:, 1::2] = -M[:, 1::2]
                    N[:, 1::2] = -N[:, 1::2]

                fid.write("%i\n" % nD)
                fid.close()

                fid = open(fileName, "ab")
                np.savetxt(fid, np.c_[M, N,], delimiter=str(" "), newline=str("\n"))

        if dim == 3:
            fid = open(fileName, "a")
            # Flip sign of z value for UBC DCoctree code
            tx[:, 2] = -tx[:, 2]
            # print(tx)

            # Flip sign of z value for UBC DCoctree code
            M[:, 2] = -M[:, 2]
            N[:, 2] = -N[:, 2]

            if format_type.lower() == "surface":

                fid.writelines("%e " % ii for ii in mkvc(tx[:, 0:2].T))
                M = M[:, 0:2]
                N = N[:, 0:2]

            if format_type.lower() == "general":

                fid.writelines("%e " % ii for ii in mkvc(tx.T))

            fid.write("%i\n" % nD)

            fid.close()

            fid = open(fileName, "ab")
            np.savetxt(
                fid, np.c_[M, N], fmt=str("%e"), delimiter=str(" "), newline=str("\n")
            )
            fid.close()

            fid = open(fileName, "a")
            fid.write("\n")
            fid.close()

        count += nD

    fid.close()





def readUBC_DC2Dpre(fileName):
    """
        Read UBC GIF DCIP 2D observation file and generate arrays
        for tx-rx location

        Input:
        :param string fileName: path to the UBC GIF 3D obs file

        Output:
        :return survey: 2D DC survey class object
        :rtype: SimPEG.electromagnetics.static.resistivity.Survey

        Created on Mon March 9th, 2016 << Doug's 70th Birthday !! >>

        @author: dominiquef

    """

    # Load file
    obsfile = np.genfromtxt(fileName, delimiter=" \n", dtype=np.str, comments="!")

    # Pre-allocate
    srcLists = []
    Rx = []
    d = []
    zflag = True  # Flag for z value provided

    for ii in range(obsfile.shape[0]):

        if not obsfile[ii]:
            continue

        # First line is transmitter with number of receivers

        temp = np.fromstring(obsfile[ii], dtype=float, sep=" ").T

        # Check if z value is provided, if False -> nan
        if len(temp) == 5:
            tx = np.r_[temp[0], np.nan, np.nan, temp[1], np.nan, np.nan]
            zflag = False

        else:
            tx = np.r_[temp[0], np.nan, temp[1], temp[2], np.nan, temp[3]]

        if zflag:
            rx = np.c_[temp[4], np.nan, temp[5], temp[6], np.nan, temp[7]]

        else:
            rx = np.c_[temp[2], np.nan, np.nan, temp[3], np.nan, np.nan]
            # Check if there is data with the location

        d.append(temp[-1])

        Rx = dc.Rx.Dipole(rx[:, :3], rx[:, 3:])
        srcLists.append(dc.Src.Dipole([Rx], tx[:3], tx[3:]))

    # Create survey class
    survey = dc.Survey(srcLists)
    data = Data(survey=survey, dobs=np.asarray(d))

    return data


def readUBC_DC3Dobs(fileName):
    """
        Read UBC GIF DCIP 3D observation file and generate arrays
        for tx-rx location
        Input:
        :param string fileName: path to the UBC GIF 3D obs file
        Output:
        :param rx, tx, d, wd
        :return
    """

    # Load file
    obsfile = np.genfromtxt(fileName, delimiter=" \n", dtype=np.str, comments="!")

    # Pre-allocate
    srcLists = []
    Rx = []
    d = []
    wd = []
    # Flag for z value provided
    zflag = True
    poletx = False
    polerx = False

    # Countdown for number of obs/tx
    count = 0
    for ii in range(obsfile.shape[0]):

        if not obsfile[ii]:
            continue

        # First line is transmitter with number of receivers
        if count == 0:
            rx = []
            temp = np.fromstring(obsfile[ii], dtype=float, sep=" ").T
            count = int(temp[-1])
            # Check if z value is provided, if False -> nan
            if len(temp) == 5:
                # check if pole-dipole
                if np.allclose(temp[0:2], temp[2:4]):
                    tx = np.r_[temp[0:2], np.nan]
                    poletx = True

                else:
                    tx = np.r_[temp[0:2], np.nan, temp[2:4], np.nan]
                zflag = False

            else:
                # check if pole-dipole
                if np.allclose(temp[0:3], temp[3:6]):
                    tx = np.r_[temp[0:3]]
                    poletx = True
                    temp[2] = -temp[2]
                else:
                    # Flip z values
                    temp[2] = -temp[2]
                    temp[5] = -temp[5]
                    tx = temp[:-1]

            continue

        temp = np.fromstring(obsfile[ii], dtype=float, sep=" ")

        if zflag:

            # Check if Pole Receiver
            if np.allclose(temp[0:3], temp[3:6]):
                polerx = True
                # Flip z values
                temp[2] = -temp[2]
                rx.append(temp[:3])
            else:
                temp[2] = -temp[2]
                temp[5] = -temp[5]
                rx.append(temp[:-2])

            # Check if there is data with the location
            if len(temp) == 8:
                d.append(temp[-2])
                wd.append(temp[-1])

        else:
            # Check if Pole Receiver
            if np.allclose(temp[0:2], temp[2:4]):
                polerx = True
                # Flip z values
                rx.append(temp[:2])
            else:
                rx.append(np.r_[temp[0:2], np.nan, temp[2:4], np.nan])

            # Check if there is data with the location
            if len(temp) == 6:
                d.append(temp[-2])
                wd.append(temp[-1])

        count = count - 1

        # Reach the end of transmitter block
        if count == 0:
            rx = np.asarray(rx)
            if polerx:
                Rx = dc.Rx.Pole(rx[:, :3])
            else:
                Rx = dc.Rx.Dipole(rx[:, :3], rx[:, 3:])
            if poletx:
                srcLists.append(dc.Src.Pole([Rx], tx[:3]))
            else:
                srcLists.append(dc.Src.Dipole([Rx], tx[:3], tx[3:]))

    # Define survey type
    if poletx:
        str1 = "pole-"
    else:
        str1 = "dipole-"

    if polerx:
        str2 = "pole"
    else:
        str2 = "dipole"

    survey_type = str1 + str2

    survey = dc.Survey(srcLists, survey_type=survey_type)
    data = Data(survey=survey, dobs=np.asarray(d), relative_error=np.asarray(wd))
    return data


def xy_2_lineID(dc_survey):
    """
        Read DC survey class and append line ID.
        Assumes that the locations are listed in the order
        they were collected. May need to generalize for random
        point locations, but will be more expensive

        Input:
        :param DCdict Vectors of station location

        Output:
        :return LineID Vector of integers
    """

    # Compute unit vector between two points
    nstn = dc_survey.nSrc

    # Pre-allocate space
    lineID = np.zeros(nstn)

    linenum = 0
    indx = 0

    for ii in range(nstn):

        if ii == 0:

            A = dc_survey.source_list[ii].location[0]
            B = dc_survey.source_list[ii].location[1]

            xout = np.mean([A[0:2], B[0:2]], axis=0)

            xy0 = A[:2]
            xym = xout

            # Deal with replicate pole location
            if np.all(xy0 == xym):

                xym[0] = xym[0] + 1e-3

            continue

        A = dc_survey.source_list[ii].location[0]
        B = dc_survey.source_list[ii].location[1]

        xin = np.mean([A[0:2], B[0:2]], axis=0)

        vec1, r1 = r_unit(xout, xin)  # Compute vector between neighbours
        vec2, r2 = r_unit(xym, xin)  # Compute vector between current stn and mid-point
        vec3, r3 = r_unit(xy0, xin)  # Compute vector between current stn and start line
        vec4, r4 = r_unit(xym, xy0)  # Compute vector between mid-point and start line

        # Compute dot product
        ang1 = np.abs(vec1.dot(vec2))
        ang2 = np.abs(vec3.dot(vec4))

        # If the angles are smaller then 45d, than next point is on a new line
        if ((ang1 < np.cos(np.pi / 4.0)) | (ang2 < np.cos(np.pi / 4.0))) & (
            np.all(np.r_[r1, r2, r3, r4] > 0)
        ):

            # Re-initiate start and mid-point location
            xy0 = A[:2]
            xym = xin

            # Deal with replicate pole location
            if np.all(xy0 == xym):

                xym[0] = xym[0] + 1e-3

            linenum += 1
            indx = ii

        else:
            xym = np.mean([xy0, xin], axis=0)

        lineID[ii] = linenum
        xout = xin

    return lineID


def r_unit(p1, p2):
    """
    r_unit(x, y) : Function computes the unit vector
    between two points with coordinates p1(x1, y1) and p2(x2, y2)

    """

    assert len(p1) == len(p2), "locs must be the same shape."

    dx = []
    for ii in range(len(p1)):
        dx.append((p2[ii] - p1[ii]))

    # Compute length of vector
    r = np.linalg.norm(np.asarray(dx))

    if r != 0:
        vec = dx / r

    else:
        vec = np.zeros(len(p1))

    return vec, r


def getSrc_locs(survey):
    """
        Read in a DC survey class and extract the xyz location of all
        sources.

        Input:
        :param survey: DC survey class object
        :rtype: SimPEG.electromagnetics.static.resistivity.Survey

        Output:
        :return numpy.ndarray srcMat: Array containing the locations of sources

    """

    srcMat = []

    for src in survey.source_list:

        srcMat.append(np.hstack(src.location))

    srcMat = np.vstack(srcMat)

    return srcMat


def gettopoCC(mesh, actind, option="top"):
    """
        Get topography from active indices of mesh.
    """

    if mesh._meshType == "TENSOR":

        if mesh.dim == 3:

            mesh2D = discretize.TensorMesh([mesh.hx, mesh.hy], mesh.x0[:2])
            zc = mesh.cell_centers[:, 2]
            ACTIND = actind.reshape((mesh.vnC[0] * mesh.vnC[1], mesh.vnC[2]), order="F")
            ZC = zc.reshape((mesh.vnC[0] * mesh.vnC[1], mesh.vnC[2]), order="F")
            topoCC = np.zeros(ZC.shape[0])

            for i in range(ZC.shape[0]):
                ind = np.argmax(ZC[i, :][ACTIND[i, :]])
                if option == "top":
                    dz = mesh.hz[ACTIND[i, :]][ind] * 0.5
                elif option == "center":
                    dz = 0.0
                else:
                    raise Exception()
                topoCC[i] = ZC[i, :][ACTIND[i, :]].max() + dz
            return mesh2D, topoCC

        elif mesh.dim == 2:

            mesh1D = discretize.TensorMesh([mesh.hx], [mesh.x0[0]])
            yc = mesh.cell_centers[:, 1]
            ACTIND = actind.reshape((mesh.vnC[0], mesh.vnC[1]), order="F")
            YC = yc.reshape((mesh.vnC[0], mesh.vnC[1]), order="F")
            topoCC = np.zeros(YC.shape[0])
            for i in range(YC.shape[0]):
                ind = np.argmax(YC[i, :][ACTIND[i, :]])
                if option == "top":
                    dy = mesh.hy[ACTIND[i, :]][ind] * 0.5
                elif option == "center":
                    dy = 0.0
                else:
                    raise Exception()
                topoCC[i] = YC[i, :][ACTIND[i, :]].max() + dy
            return mesh1D, topoCC

    elif mesh._meshType == "TREE":

        inds = mesh.get_boundary_cells(actind, direction="zu")[0]

        if option == "top":
            dz = mesh.h_gridded[inds, -1] * 0.5
        elif option == "center":
            dz = 0.0
        return mesh.cell_centers[inds, :-1], mesh.cell_centers[inds, -1] + dz


def drapeTopotoLoc(mesh, pts, actind=None, option="top", topo=None):
    """
        Drape location right below (cell center) the topography
    """
    if mesh.dim == 2:
        # if shape is (*, 1) or (*, 2) just grab first column
        if pts.ndim == 2 and pts.shape[1] in [1, 2]:
            pts = pts[:, 0]
        if pts.ndim > 1:
            raise ValueError("pts should be 1d array")
    elif mesh.dim == 3:
        if pts.shape[1] not in [2, 3]:
            raise ValueError("shape of pts should be (x, 3) or (x, 2)")
        # just grab the xy locations in the first two columns
        pts = pts[:, :2]
    else:
        raise NotImplementedError()
    if actind is None:
        actind = surface2ind_topo(mesh, topo)
    if mesh._meshType == "TENSOR":
        meshtemp, topoCC = gettopoCC(mesh, actind, option=option)
        inds = closestPoints(meshtemp, pts)
        topo = topoCC[inds]
        out = np.c_[pts, topo]

    elif mesh._meshType == "TREE":
        if mesh.dim == 3:
            uniqXYlocs, topoCC = gettopoCC(mesh, actind, option=option)
            inds = closestPointsGrid(uniqXYlocs, pts)
            out = np.c_[uniqXYlocs[inds, :], topoCC[inds]]
        else:
            uniqXlocs, topoCC = gettopoCC(mesh, actind, option=option)
            inds = closestPointsGrid(uniqXlocs, pts, dim=1)
            out = np.c_[uniqXlocs[inds], topoCC[inds]]
    else:
        raise NotImplementedError()

    return out


def genTopography(mesh, zmin, zmax, seed=None, its=100, anisotropy=None):
    if mesh.dim == 3:
        mesh2D = discretize.TensorMesh([mesh.hx, mesh.hy], x0=[mesh.x0[0], mesh.x0[1]])
        out = model_builder.randomModel(
            mesh.vnC[:2], bounds=[zmin, zmax], its=its, seed=seed, anisotropy=anisotropy
        )
        return out, mesh2D
    elif mesh.dim == 2:
        mesh1D = discretize.TensorMesh([mesh.hx], x0=[mesh.x0[0]])
        out = model_builder.randomModel(
            mesh.vnC[:1], bounds=[zmin, zmax], its=its, seed=seed, anisotropy=anisotropy
        )
        return out, mesh1D
    else:
        raise Exception("Only works for 2D and 3D models")


def closestPointsGrid(grid, pts, dim=2):
    """Move a list of points to the closest points on a grid.

    :param numpy.ndarray pts: Points to move
    :rtype: numpy.ndarray
    :return: nodeInds
    """
    if dim == 1:
        nodeInds = np.asarray(
            [np.abs(pt - grid).argmin() for pt in pts.tolist()], dtype=int
        )
    else:
        tree = cKDTree(grid)
        _, nodeInds = tree.query(pts)

    return nodeInds


def gen_3d_survey_from_2d_lines(
    survey_type,
    a,
    b,
    n_spacing,
    n_lines=5,
    line_length=200.0,
    line_spacing=20.0,
    x0=0,
    y0=0,
    z0=0,
    src_offset_y=0.0,
    dim=3,
    is_IO=True,
):
    """
        Generate 3D DC survey using gen_DCIPsurvey function.

        Input:
        :param str survey_type: 'dipole-dipole' | 'pole-dipole' |
            'dipole-pole' | 'pole-pole' | 'gradient'
        :param int a: pole seperation
        :param int b: dipole separation
        :param int n_spacing: number of rx dipoles per tx

        Output:
        :return SimPEG.dc.SurveyDC.Survey survey_3d: 3D DC survey object
    """
    ylocs = np.arange(n_lines) * line_spacing + y0

    survey_lists_2d = []
    srcList = []
    line_inds = []
    for i, y in enumerate(ylocs):
        # Generate DC survey object
        xmin, xmax = x0, x0 + line_length
        ymin, ymax = y, y
        zmin, zmax = 0, 0
        IO_2d = dc.IO()
        endl = np.array([[xmin, ymin, zmin], [xmax, ymax, zmax]])
        survey_2d = gen_DCIPsurvey(endl, survey_type, a, b, n_spacing, dim=3,)

        srcList.append(survey_2d.source_list)
        survey_2d = IO_2d.from_ambn_locations_to_survey(
            survey_2d.locations_a[:, [0, 2]],
            survey_2d.locations_b[:, [0, 2]],
            survey_2d.locations_m[:, [0, 2]],
            survey_2d.locations_n[:, [0, 2]],
            survey_type,
            dimension=2,
        )
        survey_lists_2d.append(survey_2d)
        line_inds.append(np.ones(survey_2d.nD, dtype=int) * i)
    line_inds = np.hstack(line_inds)
    srcList = sum(srcList, [])
    survey_3d = dc.Survey(srcList)
    IO_3d = dc.IO()

    survey_3d.locations_a[:, 1] += src_offset_y
    survey_3d.locations_b[:, 1] += src_offset_y

    survey_3d = IO_3d.from_ambn_locations_to_survey(
        survey_3d.locations_a,
        survey_3d.locations_b,
        survey_3d.locations_m,
        survey_3d.locations_n,
        survey_type,
        dimension=3,
        line_inds=line_inds,
    )
    return IO_3d, survey_3d





############
# Deprecated
############

def plot_pseudosection(
    data,
    ax=None,
    survey_type="dipole-dipole",
    data_type="apparent conductivity",
    space_type="half space",
    plot_type="pcolor",
    clim=None,
    scale="linear",
    sameratio=True,
    pcolor_opts={},
    contour_opts={},
    cbar_opts={},
    data_locations=False,
    dobs=None,
    dim=2,
    y_values="n-spacing",
    **kwargs,
):

    warnings.warn(
        "The plot_pseudosection method has been deprecated. Please use "
        "plot_2d_pseudosection instead. This will be removed in version"
        " 0.15.0 of SimPEG",
        DeprecationWarning,
    )

    return plot_2d_pseudosection(data.survey, data.dobs, 'scatter', ax=None, scale=scale)

def apparent_resistivity(data_object, space_type='half space', dobs=None, eps=1e-10, **kwargs):

    warnings.warn(
        "The apparent_resistivity method has been deprecated. Please use "
        "apparent_resistivity_from_voltage instead. This will be removed in version"
        " 0.15.0 of SimPEG",
        DeprecationWarning,
    )


def source_receiver_midpoints(survey, **kwargs):
    warnings.warn(
        "The source_receiver_midpoint method has been deprecated. Please use "
        "pseudo_locations instead. This will be removed in version"
        " 0.15.0 of SimPEG",
        DeprecationWarning,
    )

    return pseudo_locations(survey, kwargs)

def plot_layer(rho, mesh, **kwargs):
    warnings.warn(
        "The plot_layer method has been deprecated. Please use "
        "plot_1d_layer_model instead. This will be removed in version"
        " 0.15.0 of SimPEG",
        DeprecationWarning,
    )

    return plot_1d_layer_model(mesh.hx, rho)

def convertObs_DC3D_to_2D(survey, lineID, flag="local"):
    warnings.warn(
        "The convertObs_DC3D_to_2D method has been deprecated. Please use "
        "convert_3d_survey_to_2d. This will be removed in version"
        " 0.15.0 of SimPEG",
        DeprecationWarning,
    )

    return convert_3d_survey_to_2d_lines(survey, lineID)

