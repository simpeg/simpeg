import numpy as np
from scipy.interpolate import LinearNDInterpolator, interp1d
from scipy.spatial import cKDTree
from numpy import matlib
import discretize
import matplotlib.pyplot as plt
import warnings

from ....data import Data
from .. import resistivity as dc
from ....utils import (
    closestPoints,
    mkvc,
    surface2ind_topo,
    model_builder,
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


def electrode_separations(dc_survey, survey_type="dipole-dipole", electrode_pair="all"):
    """
    Calculate electrode separation distances.

    Input:
    :param SimPEG.electromagnetics.static.resistivity.survey.Survey dc_survey: DC survey object
    :param str survey_type: Either 'pole-dipole' | 'dipole-dipole'
                                  | 'dipole-pole' | 'pole-pole'

    Output:
    :return list ***: electrodes [A,B] separation distances

    """

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

    for ii, src in enumerate(dc_survey.source_list):
        Tx = src.location
        Rx = src.receiver_list[0].locations
        nDTx = src.receiver_list[0].nD

        if survey_type.lower() == "dipole-dipole":
            A = matlib.repmat(Tx[0], nDTx, 1)
            B = matlib.repmat(Tx[1], nDTx, 1)
            M = Rx[0]
            N = Rx[1]

            AB.append(np.sqrt(np.sum((A[:, :] - B[:, :]) ** 2.0, axis=1)))
            MN.append(np.sqrt(np.sum((M[:, :] - N[:, :]) ** 2.0, axis=1)))
            AM.append(np.sqrt(np.sum((A[:, :] - M[:, :]) ** 2.0, axis=1)))
            AN.append(np.sqrt(np.sum((A[:, :] - N[:, :]) ** 2.0, axis=1)))
            BM.append(np.sqrt(np.sum((B[:, :] - M[:, :]) ** 2.0, axis=1)))
            BN.append(np.sqrt(np.sum((B[:, :] - N[:, :]) ** 2.0, axis=1)))

        elif survey_type.lower() == "pole-dipole":
            A = matlib.repmat(Tx, nDTx, 1)
            M = Rx[0]
            N = Rx[1]

            MN.append(np.sqrt(np.sum((M[:, :] - N[:, :]) ** 2.0, axis=1)))
            AM.append(np.sqrt(np.sum((A[:, :] - M[:, :]) ** 2.0, axis=1)))
            AN.append(np.sqrt(np.sum((A[:, :] - N[:, :]) ** 2.0, axis=1)))

        elif survey_type.lower() == "dipole-pole":
            A = matlib.repmat(Tx[0], nDTx, 1)
            B = matlib.repmat(Tx[1], nDTx, 1)
            M = Rx

            AB.append(np.sqrt(np.sum((A[:, :] - B[:, :]) ** 2.0, axis=1)))
            AM.append(np.sqrt(np.sum((A[:, :] - M[:, :]) ** 2.0, axis=1)))
            BM.append(np.sqrt(np.sum((B[:, :] - M[:, :]) ** 2.0, axis=1)))

        elif survey_type.lower() == "pole-pole":
            A = matlib.repmat(Tx, nDTx, 1)
            M = Rx

            AM.append(np.sqrt(np.sum((A[:, :] - M[:, :]) ** 2.0, axis=1)))

        else:
            raise Exception(
                "survey_type must be 'dipole-dipole' | 'pole-dipole' | "
                "'dipole-pole' | 'pole-pole' not {}".format(survey_type)
            )

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


def source_receiver_midpoints(survey, **kwargs):
    """
        Calculate source receiver midpoints.

        Input:
        :param SimPEG.electromagnetics.static.resistivity.Survey survey: DC survey object

        Output:
        :return numpy.ndarray midx: midpoints x location
        :return numpy.ndarray midz: midpoints z location
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
    midxy = []
    midz = []

    for ii, source in enumerate(survey.source_list):
        tx_locs = source.location
        if isinstance(tx_locs, list):
            Cmid = (tx_locs[0][:-1] + tx_locs[1][:-1]) / 2
            zsrc = (tx_locs[0][-1] + tx_locs[1][-1]) / 2
            tx_sep = np.linalg.norm((tx_locs[0][:-1] - tx_locs[1][:-1]))
        else:
            Cmid = tx_locs[:-1]
            zsrc = tx_locs[-1]

        Pmids = []
        for receiver in source.receiver_list:
            rx_locs = receiver.locations
            if isinstance(rx_locs, list):
                Pmid = (rx_locs[0][:, :-1] + rx_locs[1][:, :-1]) / 2
            else:
                Pmid = rx_locs[:, :-1]
            Pmids.append(Pmid)
        Pmid = np.vstack(Pmids)

        midxy.append((Cmid + Pmid) / 2)
        diffs = np.linalg.norm((Cmid - Pmid), axis=1)
        if np.allclose(diffs, 0.0):  # likely a wenner type survey.
            midz = zsrc - tx_sep / 2 * np.ones_like(diffs)
        else:
            midz.append(zsrc - diffs / 2)

    return np.vstack(midxy), np.hstack(midz)


def geometric_factor(dc_survey, survey_type="dipole-dipole", space_type="half space"):
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
    # Set factor for whole-space or half-space assumption
    if space_type.lower() in SPACE_TYPES["whole space"]:
        spaceFact = 4.0
    elif space_type.lower() in SPACE_TYPES["half space"]:
        spaceFact = 2.0
    else:
        raise Exception("'space_type must be 'whole space' | 'half space'")

    elecSepDict = electrode_separations(
        dc_survey, survey_type=survey_type, electrode_pair=["AM", "BM", "AN", "BN"]
    )
    AM = elecSepDict["AM"]
    BM = elecSepDict["BM"]
    AN = elecSepDict["AN"]
    BN = elecSepDict["BN"]

    # Determine geometric factor G based on electrode separation distances
    if survey_type.lower() == "dipole-dipole":
        G = 1 / AM - 1 / BM - 1 / AN + 1 / BN

    elif survey_type.lower() == "pole-dipole":
        G = 1 / AM - 1 / AN

    elif survey_type.lower() == "dipole-pole":
        G = 1 / AM - 1 / BM

    elif survey_type.lower() == "pole-pole":
        G = 1 / AM

    else:
        raise Exception(
            "survey_type must be 'dipole-dipole' | 'pole-dipole' | "
            "'dipole-pole' | 'pole-pole' not {}".format(survey_type)
        )

    return G / (spaceFact * np.pi)


def apparent_resistivity(data, space_type="half space", dobs=None, eps=1e-10, **kwargs):
    """
    Calculate apparent resistivity. Assuming that data are normalized
    voltages - Vmn/I (Potential difference [V] divided by injection
    current [A]). For fwd modelled data an injection current of 1A is
    assumed in SimPEG.

    Input:
    :param SimPEG.Data: DC data object
    :param numpy.ndarray dobs: normalized voltage measurements [V/A]
    :param str survey_type: Either 'dipole-dipole' | 'pole-dipole' |
        'dipole-pole' | 'pole-pole'
    :param float eps: Regularizer in case of a null geometric factor

    Output:
    :return rhoApp: apparent resistivity
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

    if dobs is None:
        dobs = data.dobs

    # Calculate Geometric Factor
    G = geometric_factor(
        data.survey, survey_type=data.survey.survey_type, space_type=space_type
    )

    # Calculate apparent resistivity
    # absolute value is required because of the regularizer
    rhoApp = np.abs(dobs * (1.0 / (G + eps)))

    return rhoApp


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
    pcolorOpts=None,
    data_location=None,
    y_values="n-spacing",
):
    """
        Read list of 2D tx-rx location and plot a pseudo-section of apparent
        resistivity.

        Assumes flat topo for now...

        Input:
        :param SimPEG.Data: DC data
        :param matplotlib.pyplot.axes ax: figure axes on which to plot
        :param str survey_type: Either 'dipole-dipole' | 'pole-dipole' |
            'dipole-pole' | 'pole-pole'
        :param str data_type: Either 'appResistivity' | 'appConductivity' |
            'volt' (potential)
        :param str space_type: Either 'half-space' (default) or 'whole-space'
        :param str scale: Either 'linear' (default) or 'log'
        :param y_values: Either "n-spacing"

        Output:
        :return  matplotlib.pyplot.figure plot overlayed on image
    """
    import pylab as plt
    from scipy.interpolate import griddata

    if pcolorOpts is not None:
        warnings.warn(
            "The pcolorOpts keyword has been deprecated. Please use "
            "pcolor_opts instead. This will be removed in version"
            " 0.15.0 of SimPEG",
            DeprecationWarning,
        )

    if data_location is not None:
        warnings.warn(
            "The data_location keyword has been deprecated. Please use "
            "data_locations instead. This will be removed in version"
            " 0.15.0 of SimPEG",
            DeprecationWarning,
        )

    if plot_type.lower() not in ["pcolor", "contourf"]:
        raise ValueError(
            "plot_type must be 'pcolor' or 'contourf'. The input value of "
            f"{plot_type} is not recognized"
        )

    # Set depth to 0 for now
    z0 = 0.0
    rho = []

    if not isinstance(data, Data):
        raise Exception(
            "A Data instance ({datacls}: <{datapref}.{datacls}>) must be "
            "provided as the second input. The provided input is a "
            "{providedcls} <{providedpref}.{providedcls}>".format(
                datacls=Data.__name__,
                datapref=Data.__module__,
                providedcls=data.__name__,
                providedpref=data.__module__,
            )
        )
    # Use dobs in survey if dobs is None
    if dobs is None:
        dobs = data.dobs

    midx, midz = source_receiver_midpoints(data.survey)
    if midx.shape[1] == 2:
        min_x, min_y = np.min(midx, axis=0)
        max_x, max_y = np.max(midx, axis=0)
        if max_x - min_x > max_y - min_y:
            midx = midx[:, 0]
        else:
            midx = midx[:, 1]
    else:
        midx = midx[:, 0]

    if data_type.lower() in (
        DATA_TYPES["potential"]
        + DATA_TYPES["apparent chargeability"]
        + ["misfit", "misfitmap"]
    ):
        if scale == "linear":
            rho = dobs
        elif scale == "log":
            rho = np.log10(abs(dobs))

    elif data_type.lower() in DATA_TYPES["apparent conductivity"]:
        rhoApp = apparent_resistivity(
            data, dobs=dobs, survey_type=survey_type, space_type=space_type
        )
        if scale == "linear":
            rho = 1.0 / rhoApp
        elif scale == "log":
            rho = np.log10(1.0 / rhoApp)

    elif data_type.lower() in DATA_TYPES["apparent resistivity"]:
        rhoApp = apparent_resistivity(
            data, dobs=dobs, survey_type=survey_type, space_type=space_type
        )
        if scale == "linear":
            rho = rhoApp
        elif scale == "log":
            rho = np.log10(rhoApp)

    else:
        print()
        raise Exception(
            """data_type must be 'potential' | 'apparent resistivity' |
                'apparent conductivity' | 'apparent chargeability' | misfit"""
            " not {}".format(data_type)
        )

    # Grid points
    grid_x, grid_z = np.mgrid[np.min(midx) : np.max(midx), np.min(midz) : np.max(midz)]

    grid_rho = griddata(np.c_[midx, midz], rho.T, (grid_x, grid_z), method="linear")

    if clim is None:
        vmin, vmax = rho.min(), rho.max()
    else:
        vmin, vmax = clim[0], clim[1]

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(15, 3))

    ph = grid_rho = np.ma.masked_where(np.isnan(grid_rho), grid_rho)
    if plot_type.lower() == "pcolor":
        ph = ax.pcolormesh(
            grid_x[:, 0],
            grid_z[0, :],
            grid_rho.T,
            clim=(vmin, vmax),
            vmin=vmin,
            vmax=vmax,
            **pcolor_opts,
        )
    elif plot_type.lower() == "contourf":
        ph = ax.contourf(
            grid_x[:, 0], grid_z[0, :], grid_rho.T, vmin=vmin, vmax=vmax, **contour_opts
        )

    if scale == "log":
        cbar = plt.colorbar(
            ph,
            format="$10^{%.2f}$",
            fraction=0.06,
            orientation="horizontal",
            ax=ax,
            **cbar_opts,
        )
    elif scale == "linear":
        cbar = plt.colorbar(
            ph,
            format="%.2f",
            fraction=0.06,
            orientation="horizontal",
            ax=ax,
            **cbar_opts,
        )

    ticks = np.linspace(vmin, vmax, 3)
    cbar.set_ticks(ticks)
    cbar.ax.tick_params()

    # Plot apparent resistivity
    if data_location:
        ax.plot(midx, midz, "k.", ms=1, alpha=0.4)

    if sameratio:
        ax.set_aspect("equal", adjustable="box")

    if y_values == "n-spacing":
        ticks = ax.get_yticks() * 2  # pseudo-depth divides by 2
        spacing = np.abs(midz).min()
        ax.set_yticklabels(-ticks / spacing)
        ax.set_ylabel("n-spacing")
    elif y_values == "pseudo-depth":
        ax.set_ylabel("pseudo-depth")

    if data_type.lower() in DATA_TYPES["apparent conductivity"]:
        cbar.set_label("Apparent Conductivity (S/m)")

    elif data_type.lower() in DATA_TYPES["apparent resistivity"]:
        cbar.set_label("Apparent Resistivity ($\\Omega$m)")

    elif data_type.lower() in DATA_TYPES["potential"]:
        cbar.set_label("Voltage (V)")

    elif data_type.lower() in DATA_TYPES["apparent chargeability"]:
        cbar.set_label("Apparent Chargeability (V/V)")

    elif data_type.lower() in ["misfit", "misfitmap"]:
        cbar.set_label("Misfit (V)")

    return ax


def generate_dcip_survey(endl, survey_type, a, b, n, dim=3, d2flag="2.5D"):

    """
        Load in endpoints and survey specifications to generate Tx, Rx location
        stations.

        Assumes flat topo for now...

        Input:
        :param numpy.ndarray endl: input endpoints [x1, y1, z1, x2, y2, z2]
        :param discretize.base.BaseMesh mesh: discretize mesh object
        :param str survey_type: 'dipole-dipole' | 'pole-dipole' |
            'dipole-pole' | 'pole-pole' | 'gradient'
        :param int a: pole seperation
        :param int b: dipole separation
        :param int n: number of rx dipoles per tx
        :param str d2flag: choose for 2D mesh between a '2D' or a '2.5D' survey

        Output:
        :return SimPEG.electromagnetics.static.resistivity.Survey dc_survey: DC survey object
    """

    def xy_2_r(x1, x2, y1, y2):
        r = np.sqrt(np.sum((x2 - x1) ** 2.0 + (y2 - y1) ** 2.0))
        return r

    # Evenly distribute electrodes and put on surface
    # Mesure survey length and direction
    dl_len = xy_2_r(endl[0, 0], endl[1, 0], endl[0, 1], endl[1, 1])

    dl_x = (endl[1, 0] - endl[0, 0]) / dl_len
    dl_y = (endl[1, 1] - endl[0, 1]) / dl_len

    nstn = int(np.floor(dl_len / a))

    # Compute discrete pole location along line
    stn_x = endl[0, 0] + np.array(range(int(nstn))) * dl_x * a
    stn_y = endl[0, 1] + np.array(range(int(nstn))) * dl_y * a

    if dim == 2:
        ztop = np.linspace(endl[0, 1], endl[0, 1], nstn)
        # Create line of P1 locations
        M = np.c_[stn_x, ztop]
        # Create line of P2 locations
        N = np.c_[stn_x + a * dl_x, ztop]

    elif dim == 3:
        stn_z = np.linspace(endl[0, 2], endl[0, 2], nstn)
        # Create line of P1 locations
        M = np.c_[stn_x, stn_y, stn_z]
        # Create line of P2 locations
        N = np.c_[stn_x + a * dl_x, stn_y + a * dl_y, stn_z]

    # Build list of Tx-Rx locations depending on survey type
    # Dipole-dipole: Moving tx with [a] spacing -> [AB a MN1 a MN2 ... a MNn]
    # Pole-dipole: Moving pole on one end -> [A a MN1 a MN2 ... MNn a B]
    SrcList = []

    if survey_type != "gradient":

        for ii in range(0, int(nstn) - 1):

            if survey_type.lower() in ["dipole-dipole", "dipole-pole"]:
                tx = np.c_[M[ii, :], N[ii, :]]
                # Current elctrode separation
                AB = xy_2_r(tx[0, 1], endl[1, 0], tx[1, 1], endl[1, 1])
            elif survey_type.lower() in ["pole-dipole", "pole-pole"]:
                tx = np.r_[M[ii, :]]
                # Current elctrode separation
                AB = xy_2_r(tx[0], endl[1, 0], tx[1], endl[1, 1])
            else:
                raise Exception(
                    "survey_type must be 'dipole-dipole' | 'pole-dipole' | "
                    "'dipole-pole' | 'pole-pole' not {}".format(survey_type)
                )

            # Rx.append(np.c_[M[ii+1:indx, :], N[ii+1:indx, :]])

            # Number of receivers to fit
            nstn = int(np.min([np.floor((AB - b) / a), n]))

            # Check if there is enough space, else break the loop
            if nstn <= 0:
                continue

            # Compute discrete pole location along line
            stn_x = N[ii, 0] + dl_x * b + np.array(range(int(nstn))) * dl_x * a
            stn_y = N[ii, 1] + dl_y * b + np.array(range(int(nstn))) * dl_y * a

            # Create receiver poles

            if dim == 3:
                stn_z = np.linspace(endl[0, 2], endl[0, 2], nstn)

                # Create line of P1 locations
                P1 = np.c_[stn_x, stn_y, stn_z]
                # Create line of P2 locations
                P2 = np.c_[stn_x + a * dl_x, stn_y + a * dl_y, stn_z]
                if survey_type.lower() in ["dipole-dipole", "pole-dipole"]:
                    rxClass = dc.Rx.Dipole(P1, P2)
                elif survey_type.lower() in ["dipole-pole", "pole-pole"]:
                    rxClass = dc.Rx.Pole(P1)

            elif dim == 2:
                ztop = np.linspace(endl[0, 1], endl[0, 1], nstn)
                # Create line of P1 locations
                P1 = np.c_[stn_x, np.ones(nstn).T * ztop]
                # Create line of P2 locations
                P2 = np.c_[stn_x + a * dl_x, np.ones(nstn).T * ztop]
                if survey_type.lower() in ["dipole-dipole", "pole-dipole"]:
                    rxClass = dc.Rx.Dipole(P1, P2)
                elif survey_type.lower() in ["dipole-pole", "pole-pole"]:
                    rxClass = dc.Rx.Pole(P1)

            if survey_type.lower() in ["dipole-dipole", "dipole-pole"]:
                srcClass = dc.Src.Dipole([rxClass], M[ii, :], N[ii, :])
            elif survey_type.lower() in ["pole-dipole", "pole-pole"]:
                srcClass = dc.Src.Pole([rxClass], M[ii, :])
            SrcList.append(srcClass)

    elif survey_type.lower() == "gradient":

        # Gradient survey takes the "b" parameter to define the limits of a
        # square survey grid. The pole seperation within the receiver grid is
        # define the "a" parameter.

        # Get the edge limit of survey area
        min_x = endl[0, 0] + dl_x * b
        min_y = endl[0, 1] + dl_y * b

        max_x = endl[1, 0] - dl_x * b
        max_y = endl[1, 1] - dl_y * b

        # Define the size of the survey grid (square for now)
        box_l = np.sqrt((min_x - max_x) ** 2.0 + (min_y - max_y) ** 2.0)
        box_w = box_l / 2.0

        nstn = int(np.floor(box_l / a))

        # Compute discrete pole location along line
        stn_x = min_x + np.array(range(int(nstn))) * dl_x * a
        stn_y = min_y + np.array(range(int(nstn))) * dl_y * a

        # Define number of cross lines
        nlin = int(np.floor(box_w / a))
        lind = range(-nlin, nlin + 1)

        npoles = int(nstn * len(lind))

        rx = np.zeros([npoles, 6])
        for ii in range(len(lind)):

            # Move station location to current survey line This is a
            # perpendicular move then line survey orientation, hence the y, x
            # switch
            lxx = stn_x - lind[ii] * a * dl_y
            lyy = stn_y + lind[ii] * a * dl_x

            M = np.c_[lxx, lyy, np.ones(nstn).T * ztop]
            N = np.c_[lxx + a * dl_x, lyy + a * dl_y, np.ones(nstn).T * ztop]
            rx[(ii * nstn) : ((ii + 1) * nstn), :] = np.c_[M, N]

            if dim == 3:
                rxClass = dc.Rx.Dipole(rx[:, :3], rx[:, 3:])
            elif dim == 2:
                M = M[:, [0, 2]]
                N = N[:, [0, 2]]
                rxClass = dc.Rx.Dipole(rx[:, [0, 2]], rx[:, [3, 5]])
            srcClass = dc.Src.Dipole([rxClass], (endl[0, :]), (endl[1, :]))
        SrcList.append(srcClass)
        survey_type = "dipole-dipole"

    survey = dc.Survey(SrcList, survey_type=survey_type.lower())
    return survey


def generate_dcip_survey_line(
    survey_type, data_type, endl, topo, ds, dh, n, dim_flag="2.5D", sources_only=False
):
    """
        Generate DCIP survey line for modeling in 2D, 2.5D or 3D. Takes into accounted true surface
        topography.

        Input:
        :param str survey_type: 'dipole-dipole' | 'pole-dipole' |
            'dipole-pole' | 'pole-pole'
        :param str data_type: 'volt' | 'apparent_conductivity' |
        	'apparent_resistivity' | 'apparent_chargeability'
        :param np.array endl: horizontal end points [x1, x2] or [x1, x2, y1, y2]
        :param float , (N, 2) np.array or (N, 3) np.array: topography
        :param int ds: station seperation
        :param int dh: dipole separation (unused if pole-pole)
        :param int n: number of rx per tx
        :param str dim: '2D', '2.5D' or '3D'
        :param bool sources_only: Outputs a survey object if False. Outputs sources list if True.

        Output:
        :return SimPEG.electromagnetics.static.resistivity.Survey dc_survey: DC survey object
    """

    accepted_surveys = ["pole-pole", "pole-dipole", "dipole-pole", "dipole-dipole"]

    if survey_type.lower() not in accepted_surveys:
        raise Exception(
            "survey_type must be 'dipole-dipole' | 'pole-dipole' | "
            "'dipole-pole' | 'pole-pole' not {}".format(survey_type)
        )

    def xy_2_r(x1, x2, y1, y2):
        r = np.sqrt(np.sum((x2 - x1) ** 2.0 + (y2 - y1) ** 2.0))
        return r

    # Compute horizontal locations of sources and receivers
    x1 = endl[0]
    x2 = endl[1]

    if dim_flag == "3D":

        # Station locations
        y1 = endl[2]
        y2 = endl[3]
        L = xy_2_r(x1, x2, y1, y2)
        nstn = int(np.floor(L / ds) + 1)
        dl_x = (x2 - x1) / L
        dl_y = (y2 - y1) / L
        stn_x = x1 + np.array(range(int(nstn))) * dl_x * ds
        stn_y = y1 + np.array(range(int(nstn))) * dl_y * ds

        # Locations of poles and dipoles
        if survey_type.lower() in ["pole-pole", "pole-dipole", "dipole-pole"]:
            P = np.c_[stn_x, stn_y]
            if np.size(topo) == 1:
                P = np.c_[P, topo * np.ones((nstn))]
            else:
                fun_interp = LinearNDInterpolator(topo[:, 0:2], topo[:, -1])
                P = np.c_[P, fun_interp(P)]

        if survey_type.lower() in ["pole-dipole", "dipole-pole", "dipole-dipole"]:
            DP1 = np.c_[stn_x - 0.5 * dl_x * dh, stn_y - 0.5 * dl_y * dh]
            DP2 = np.c_[stn_x + 0.5 * dl_x * dh, stn_y + 0.5 * dl_y * dh]
            if np.size(topo) == 1:
                DP1 = np.c_[DP1, topo * np.ones((nstn))]
                DP2 = np.c_[DP2, topo * np.ones((nstn))]
            else:
                fun_interp = LinearNDInterpolator(topo[:, 0:2], topo[:, -1])
                DP1 = np.c_[DP1, fun_interp(DP1)]
                DP2 = np.c_[DP2, fun_interp(DP2)]

    else:

        # Station locations
        y1 = 0.0
        y2 = 0.0
        L = xy_2_r(x1, x2, y1, y2)
        nstn = int(np.floor(L / ds) + 1)
        stn_x = x1 + np.array(range(int(nstn))) * ds

        # Locations of poles and dipoles
        if survey_type.lower() in ["pole-pole", "pole-dipole", "dipole-pole"]:
            P = np.c_[stn_x, stn_y]
            if np.size(topo) == 1:
                P = np.c_[stn_x, topo * np.ones((nstn))]
            else:
                fun_interp = LinearNDInterpolator(topo[:, 0:2], topo[:, -1])
                P = np.c_[stn_x, fun_interp(stn_x)]

        if survey_type.lower() in ["pole-dipole", "dipole-pole", "dipole-dipole"]:
            DP1 = stn_x - 0.5 * dh
            DP2 = stn_x + 0.5 * dh
            if np.size(topo) == 1:
                DP1 = np.c_[DP1, topo * np.ones((nstn))]
                DP2 = np.c_[DP2, topo * np.ones((nstn))]
            else:
                fun_interp = interp1d(topo[:, 0], topo[:, -1])
                DP1 = np.c_[DP1, fun_interp(DP1)]
                DP2 = np.c_[DP2, fun_interp(DP2)]

    # Build list of Tx-Rx locations depending on survey type
    # Dipole-dipole: Moving tx with [a] spacing -> [AB a MN1 a MN2 ... a MNn]
    # Pole-dipole: Moving pole on one end -> [A a MN1 a MN2 ... MNn a B]
    SrcList = []

    for ii in range(0, int(nstn)):

        if dim_flag == "3D":
            D = xy_2_r(stn_x[ii], x2, stn_y[ii], y2)
        else:
            D = xy_2_r(stn_x[ii], x2, y1, y2)

        # Number of receivers to fit
        nrec = int(np.min([np.floor(D / ds), n]))

        # Check if there is enough space, else break the loop
        if nrec <= 0:
            continue

        # Create receivers
        if survey_type.lower() in ["dipole-pole", "pole-pole"]:
            rxClass = dc.receivers.Pole(
                P[ii + 1 : ii + nrec + 1, :], data_type=data_type
            )
        elif survey_type.lower() in ["dipole-dipole", "pole-dipole"]:
            rxClass = dc.receivers.Dipole(
                DP1[ii + 1 : ii + nrec + 1, :],
                DP2[ii + 1 : ii + nrec + 1, :],
                data_type=data_type,
            )

        # Create sources
        if survey_type.lower() in ["pole-dipole", "pole-pole"]:
            srcClass = dc.sources.Pole([rxClass], P[ii, :])
        elif survey_type.lower() in ["dipole-dipole", "dipole-pole"]:
            srcClass = dc.sources.Dipole([rxClass], DP1[ii, :], DP2[ii, :])

        SrcList.append(srcClass)

    if sources_only:

        return SrcList

    else:
        survey = dc.Survey(SrcList, survey_type=survey_type.lower())

        return survey


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

    for ii in range(dc_survey.nSrc):

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


def convertObs_DC3D_to_2D(survey, lineID, flag="local"):
    """
        Read DC survey and projects the coordinate system
        according to the flag = 'Xloc' | 'Yloc' | 'local' (default)
        In the 'local' system, station coordinates are referenced
        to distance from the first srcLoc[0].location[0]

        The Z value is preserved, but Y coordinates zeroed.

        Input:
        :param survey: 3D DC survey class object
        :rtype: SimPEG.electromagnetics.static.resistivity.Survey

        Output:
        :param survey: 2D DC survey class object
        :rtype: SimPEG.electromagnetics.static.resistivity.Survey
    """

    def stn_id(v0, v1, r):
        """
        Compute station ID along line
        """

        dl = int(v0.dot(v1)) * r

        return dl

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

    srcList2D = []

    srcMat = getSrc_locs(survey)

    # Find all unique line id
    uniqueID = np.unique(lineID)

    for jj in range(len(uniqueID)):

        indx = np.where(lineID == uniqueID[jj])[0]

        # Find origin of survey
        r = 1e8  # Initialize to some large number

        Tx = srcMat[indx]

        if np.all(Tx[0:3] == Tx[3:]):
            survey_type = "pole-dipole"

        else:
            survey_type = "dipole-dipole"

        x0 = Tx[0][0:2]  # Define station zero along line

        vecTx, r1 = r_unit(x0, Tx[-1][0:2])

        for ii in range(len(indx)):

            # Get all receivers
            Rx = survey.source_list[indx[ii]].receiver_list[0].locations
            nrx = Rx[0].shape[0]

            if flag == "local":
                # Find A electrode along line
                vec, r = r_unit(x0, Tx[ii][0:2])
                A = stn_id(vecTx, vec, r)

                if survey_type != "pole-dipole":
                    # Find B electrode along line
                    vec, r = r_unit(x0, Tx[ii][3:5])
                    B = stn_id(vecTx, vec, r)

                M = np.zeros(nrx)
                N = np.zeros(nrx)
                for kk in range(nrx):

                    # Find all M electrodes along line
                    vec, r = r_unit(x0, Rx[0][kk, 0:2])
                    M[kk] = stn_id(vecTx, vec, r)

                    # Find all N electrodes along line
                    vec, r = r_unit(x0, Rx[1][kk, 0:2])
                    N[kk] = stn_id(vecTx, vec, r)
            elif flag == "Yloc":
                """ Flip the XY axis locs"""
                A = Tx[ii][1]

                if survey_type != "pole-dipole":
                    B = Tx[ii][4]

                M = Rx[0][:, 1]
                N = Rx[1][:, 1]

            elif flag == "Xloc":
                """ Copy the rx-tx locs"""
                A = Tx[ii][0]

                if survey_type != "pole-dipole":
                    B = Tx[ii][3]

                M = Rx[0][:, 0]
                N = Rx[1][:, 0]

            rxClass = dc.Rx.Dipole(
                np.c_[M, np.zeros(nrx), Rx[0][:, 2]],
                np.c_[N, np.zeros(nrx), Rx[1][:, 2]],
            )

            if survey_type == "pole-dipole":
                srcList2D.append(dc.Src.Pole([rxClass], np.asarray([A, 0, Tx[ii][2]])))

            elif survey_type == "dipole-dipole":
                srcList2D.append(
                    dc.Src.Dipole(
                        [rxClass], np.r_[A, 0, Tx[ii][2]], np.r_[B, 0, Tx[ii][5]]
                    )
                )

    survey2D = dc.Survey(srcList2D)

    return survey2D


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

    survey = dc.Survey(srcLists)
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
            zc = mesh.gridCC[:, 2]
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
            yc = mesh.gridCC[:, 1]
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
        return mesh.gridCC[inds, :-1], mesh.gridCC[inds, -1] + dz


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


def plot_layer(
    rho,
    mesh,
    xscale="log",
    ax=None,
    showlayers=False,
    xlim=None,
    depth_axis=True,
    **kwargs,
):
    """
        Plot Conductivity model for the layered earth model
    """

    n_rho = rho.size

    z_grid = -mesh.vectorNx
    resistivity = np.repeat(rho, 2)

    z = []
    for i in range(n_rho):
        z.append(np.r_[z_grid[i], z_grid[i + 1]])
    z = np.hstack(z)
    z = z + mesh.x0[0]
    if xlim == None:
        rho_min = rho[~np.isnan(rho)].min() * 0.5
        rho_max = rho[~np.isnan(rho)].max() * 2
    else:
        rho_min, rho_max = xlim

    if xscale == "linear" and rho.min() == 0.0:
        if xlim == None:
            rho_min = -rho[~np.isnan(rho)].max() * 0.5
            rho_max = rho[~np.isnan(rho)].max() * 2

    if ax == None:
        plt.xscale(xscale)
        plt.xlim(rho_min, rho_max)
        plt.ylim(z.min(), z.max())
        plt.xlabel("Resistivity ($\Omega$m)", fontsize=14)
        plt.ylabel("Depth (m)", fontsize=14)
        plt.ylabel("Depth (m)", fontsize=14)
        if showlayers:
            for locz in z_grid:
                plt.plot(
                    np.linspace(rho_min, rho_max, 100),
                    np.ones(100) * locz,
                    "b--",
                    lw=0.5,
                )
        return plt.plot(resistivity, z, "k-", **kwargs)

    else:
        ax.set_xscale(xscale)
        ax.set_xlim(rho_min, rho_max)
        ax.set_ylim(z.min(), z.max())
        ax.set_xlabel("Resistivity ($\Omega$m)", fontsize=14)
        ax.set_ylabel("Depth (m)", fontsize=14)
        if showlayers:
            for locz in z_grid:
                ax.plot(
                    np.linspace(rho_min, rho_max, 100),
                    np.ones(100) * locz,
                    "b--",
                    lw=0.5,
                )
        return ax.plot(resistivity, z, "k-", **kwargs)


############
# Deprecated
############


def plot_pseudoSection(
    data,
    ax=None,
    survey_type="dipole-dipole",
    data_type="appConductivity",
    space_type="half-space",
    clim=None,
    scale="linear",
    sameratio=True,
    pcolorOpts={},
    data_location=False,
    dobs=None,
    dim=2,
):

    warnings.warn(
        "The plot_pseudoSection method has been deprecated. Please use "
        "plot_pseudosection instead. This will be removed in version"
        " 0.15.0 of SimPEG",
        DeprecationWarning,
    )

    return plot_pseudosection(
        data=data,
        ax=ax,
        survey_type=survey_type,
        data_type=data_type,
        space_type=space_type,
        clim=clim,
        scale=scale,
        sameratio=sameratio,
        pcolorOpts=pcolorOpts,
        data_locations=data_location,
        dobs=dobs,
        dim=dim,
    )


def gen_DCIPsurvey(endl, survey_type, a, b, n, dim=3, d2flag="2.5D"):
    warnings.warn(
        "The gen_DCIPsurvey method has been deprecated. Please use "
        "generate_dcip_survey instead. This will be removed in version"
        " 0.15.0 of SimPEG",
        DeprecationWarning,
    )

    return generate_dcip_survey(endl, survey_type, a, b, n, dim, d2flag)
