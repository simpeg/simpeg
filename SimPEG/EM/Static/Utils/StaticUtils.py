from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

from SimPEG import Utils, Mesh
from SimPEG.EM.Static import DC
from SimPEG.Utils import asArray_N_x_Dim, uniqueRows


def electrode_separations(
    dc_survey, survey_type='dipole-dipole', electrode_pair='All'
):
    """
        Calculate electrode separation distances.

        Input:
        :param SimPEG.EM.Static.DC.SurveyDC.Survey dc_survey: DC survey object
        :param str survey_type: Either 'pole-dipole' | 'dipole-dipole'
                                      | 'dipole-pole' | 'pole-pole'

        Output:
        :return list ***: electrodes [A,B] separation distances

    """

    if not isinstance(electrode_pair, np.ndarray):
        if electrode_pair == 'All':
            electrode_pair = np.r_[['AB', 'MN', 'AM', 'AN', 'BM', 'BN']]
        elif isinstance(electrode_pair, list) or isinstance(electrode_pair, str):
                electrode_pair = np.r_[electrode_pair]
        else:
            raise Exception(
                """electrode_pair must be either a string, list of strings, or an
                ndarray containing the electrode separation distances you would
                like to calculate""" " not {}".format(type(electrode_pair))
            )

    elecSepDict = {}
    AB = []
    MN = []
    AM = []
    AN = []
    BM = []
    BN = []

    for ii in range(dc_survey.nSrc):

        Tx = dc_survey.srcList[ii].loc
        Rx = dc_survey.srcList[ii].rxList[0].locs
        nDTx = dc_survey.srcList[ii].rxList[0].nD

        if survey_type == 'dipole-dipole':
            A = np.matlib.repmat(Tx[0], nDTx, 1)
            B = np.matlib.repmat(Tx[1], nDTx, 1)
            M = Rx[0]
            N = Rx[1]

            AB.append(np.sqrt(np.sum((A[:, :] - B[:, :])**2., axis=1)))
            MN.append(np.sqrt(np.sum((M[:, :] - N[:, :])**2., axis=1)))
            AM.append(np.sqrt(np.sum((A[:, :] - M[:, :])**2., axis=1)))
            AN.append(np.sqrt(np.sum((A[:, :] - N[:, :])**2., axis=1)))
            BM.append(np.sqrt(np.sum((B[:, :] - M[:, :])**2., axis=1)))
            BN.append(np.sqrt(np.sum((B[:, :] - N[:, :])**2., axis=1)))

        elif survey_type == 'pole-dipole':
            A = np.matlib.repmat(Tx, nDTx, 1)
            M = Rx[0]
            N = Rx[1]

            MN.append(np.sqrt(np.sum((M[:, :] - N[:, :])**2., axis=1)))
            AM.append(np.sqrt(np.sum((A[:, :] - M[:, :])**2., axis=1)))
            AN.append(np.sqrt(np.sum((A[:, :] - N[:, :])**2., axis=1)))

        elif survey_type == 'dipole-pole':
            A = np.matlib.repmat(Tx[0], nDTx, 1)
            B = np.matlib.repmat(Tx[1], nDTx, 1)
            M = Rx

            AB.append(np.sqrt(np.sum((A[:, :] - B[:, :])**2., axis=1)))
            AM.append(np.sqrt(np.sum((A[:, :] - M[:, :])**2., axis=1)))
            BM.append(np.sqrt(np.sum((B[:, :] - M[:, :])**2., axis=1)))

        elif survey_type == 'pole-pole':
            A = np.matlib.repmat(Tx, nDTx, 1)
            M = Rx

            AM.append(np.sqrt(np.sum((A[:, :] - M[:, :])**2., axis=1)))

        else:
            raise Exception(
                """survey_type must be 'dipole-dipole' | 'pole-dipole' |
                'dipole-pole' | 'pole-pole'"""
                " not {}".format(survey_type)
            )

    if np.any(electrode_pair == 'AB'):
        if AB:
            AB = np.hstack(AB)
        elecSepDict['AB'] = AB
    if np.any(electrode_pair == 'MN'):
        if MN:
            MN = np.hstack(MN)
        elecSepDict['MN'] = MN
    if np.any(electrode_pair == 'AM'):
        if AM:
            AM = np.hstack(AM)
        elecSepDict['AM'] = AM
    if np.any(electrode_pair == 'AN'):
        if AN:
            AN = np.hstack(AN)
        elecSepDict['AN'] = AN
    if np.any(electrode_pair == 'BM'):
        if BM:
            BM = np.hstack(BM)
        elecSepDict['BM'] = BM
    if np.any(electrode_pair == 'BN'):
        if BN:
            BN = np.hstack(BN)
        elecSepDict['BN'] = BN

    return elecSepDict


def source_receiver_midpoints(dc_survey, survey_type='dipole-dipole', dim=2):
    """
        Calculate source receiver midpoints.

        Input:
        :param SimPEG.EM.Static.DC.SurveyDC.Survey dc_survey: DC survey object
        :param str survey_type: Either 'pole-dipole' | 'dipole-dipole'
                                      | 'dipole-pole' | 'pole-pole'

        Output:
        :return numpy.ndarray midx: midpoints x location
        :return numpy.ndarray midz: midpoints  z location
    """

    # Pre-allocate
    midx = []
    midz = []

    for ii in range(dc_survey.nSrc):
        Tx = dc_survey.srcList[ii].loc
        Rx = dc_survey.srcList[ii].rxList[0].locs

        # Get distances between each poles A-B-M-N
        if survey_type == 'pole-dipole':
            # Create mid-point location
            Cmid = Tx[0]
            Pmid = (Rx[0][:, 0] + Rx[1][:, 0])/2
            if dim == 2:
                zsrc = Tx[1]
            elif dim == 3:
                zsrc = Tx[2]
            else:
                raise Exception()

        elif survey_type == 'dipole-dipole':
            # Create mid-point location
            Cmid = (Tx[0][0] + Tx[1][0])/2
            Pmid = (Rx[0][:, 0] + Rx[1][:, 0])/2
            if dim == 2:
                zsrc = (Tx[0][1] + Tx[1][1])/2
            elif dim == 3:
                zsrc = (Tx[0][2] + Tx[1][2])/2
            else:
                raise Exception()

        elif survey_type == 'pole-pole':
            # Create mid-point location
            Cmid = Tx[0]
            Pmid = Rx[:, 0]
            if dim == 2:
                zsrc = Tx[1]
            elif dim == 3:
                zsrc = Tx[2]
            else:
                raise Exception()

        elif survey_type == 'dipole-pole':
            # Create mid-point location
            Cmid = (Tx[0][0] + Tx[1][0])/2
            Pmid = Rx[:, 0]
            if dim == 2:
                zsrc = (Tx[0][1] + Tx[1][1])/2
            elif dim == 3:
                zsrc = (Tx[0][2] + Tx[1][2])/2
            else:
                raise Exception()
        else:
            raise Exception(
                """survey_type must be 'dipole-dipole' | 'pole-dipole' |
                'dipole-pole' | 'pole-pole'"""
                " not {}".format(survey_type)
            )

        midx = np.hstack([midx, (Cmid + Pmid)/2])
        midz = np.hstack([midz, -np.abs(Cmid-Pmid)/2 + zsrc])

    return midx, midz


def geometric_factor(
    dc_survey, survey_type='dipole-dipole', space_type='half-space'
):
    """
        Calculate Geometric Factor. Assuming that data are normalized voltages

        Input:
        :param SimPEG.EM.Static.DC.SurveyDC.Survey dc_survey: DC survey object
        :param str survey_type: Either 'dipole-dipole' | 'pole-dipole'
                               | 'dipole-pole' | 'pole-pole'
        :param str space_type: Assuming whole-space or half-space
                              ('whole-space' | 'half-space')

        Output:
        :return numpy.ndarray G: Geometric Factor

    """
    # Set factor for whole-space or half-space assumption
    if space_type == 'whole-space':
        spaceFact = 4.
    elif space_type == 'half-space':
        spaceFact = 2.
    else:
        raise Exception("""'space_type must be 'whole-space' | 'half-space'""")

    elecSepDict = electrode_separations(
            dc_survey, survey_type=survey_type,
            electrode_pair=['AM', 'BM', 'AN', 'BN']
    )
    AM = elecSepDict['AM']
    BM = elecSepDict['BM']
    AN = elecSepDict['AN']
    BN = elecSepDict['BN']

    # Determine geometric factor G based on electrode separation distances
    if survey_type == 'dipole-dipole':
        G = 1/AM - 1/BM - 1/AN + 1/BN

    elif survey_type == 'pole-dipole':
        G = 1/AM - 1/AN

    elif survey_type == 'dipole-pole':
        G = 1/AM - 1/BM

    elif survey_type == 'pole-pole':
        G = 1/AM

    else:
        raise Exception(
                """survey_type must be 'dipole-dipole' | 'pole-dipole' |
                'dipole-pole' | 'pole-pole'"""
                " not {}".format(survey_type)
            )

    return (G/(spaceFact*np.pi))


def apparent_resistivity(
    dc_survey, survey_type='dipole-dipole',
    space_type='half-space', dobs=None,
    eps=1e-10
):
    """
        Calculate apparent resistivity. Assuming that data are normalized
        voltages - Vmn/I (Potential difference [V] divided by injection
        current [A]). For fwd modelled data an injection current of 1A is
        assumed in SimPEG.

        Input:
        :param SimPEG.EM.Static.DC.SurveyDC.Survey dc_survey: DC survey object
        :param numpy.ndarray dobs: normalized voltage measurements [V/A]
        :param str survey_type: Either 'dipole-dipole' | 'pole-dipole' |
            'dipole-pole' | 'pole-pole'
        :param float eps: Regularizer in case of a null geometric factor

        Output:
        :return rhoApp: apparent resistivity
    """
    # Use dobs in survey if dobs is None
    if dobs is None:
        if dc_survey.dobs is None:
            raise Exception()
        else:
            dobs = dc_survey.dobs

    # Calculate Geometric Factor
    G = geometric_factor(
        dc_survey, survey_type=survey_type, space_type=space_type
    )

    # Calculate apparent resistivity
    # absolute value is required because of the regularizer
    rhoApp = np.abs(dobs*(1./(G+eps)))

    return rhoApp


def plot_pseudoSection(
    dc_survey, ax=None, survey_type='dipole-dipole',
    data_type="appConductivity", space_type='half-space',
    clim=None, scale="linear", sameratio=True,
    pcolorOpts={}, data_location=False, dobs=None, dim=2
):
    """
        Read list of 2D tx-rx location and plot a speudo-section of apparent
        resistivity.

        Assumes flat topo for now...

        Input:
        :param SimPEG.EM.Static.DC.SurveyDC.Survey dc_survey: DC survey object
        :param matplotlib.pyplot.axes ax: figure axes on which to plot
        :param str survey_type: Either 'dipole-dipole' | 'pole-dipole' |
            'dipole-pole' | 'pole-pole'
        :param str data_type: Either 'appResistivity' | 'appConductivity' |
            'volt' (potential)
        :param str space_type: Either 'half-space' (default) or 'whole-space'
        :param str scale: Either 'linear' (default) or 'log'

        Output:
        :return  matplotlib.pyplot.figure plot overlayed on image
    """
    import pylab as plt
    from scipy.interpolate import griddata
    # Set depth to 0 for now
    z0 = 0.
    rho = []

    # Use dobs in survey if dobs is None
    if dobs is None:
        if dc_survey.dobs is None:
            raise Exception()
        else:
            dobs = dc_survey.dobs

    rhoApp = apparent_resistivity(
                dc_survey, dobs=dobs,
                survey_type=survey_type,
                space_type=space_type
    )
    midx, midz = source_receiver_midpoints(
                    dc_survey,
                    survey_type=survey_type,
                    dim=dim
    )

    if data_type == 'volt':
        if scale == "linear":
            rho = dobs
        elif scale == "log":
            rho = np.log10(abs(dobs))

    elif data_type == 'appConductivity':
        if scale == "linear":
            rho = 1./rhoApp
        elif scale == "log":
            rho = np.log10(1./rhoApp)

    elif data_type == 'appResistivity':
        if scale == "linear":
            rho = rhoApp
        elif scale == "log":
            rho = np.log10(rhoApp)

    else:
        print()
        raise Exception(
                """data_type must be 'appResistivity' |
                'appConductivity' | 'volt' """
                " not {}".format(data_type)
        )

    # Grid points
    grid_x, grid_z = np.mgrid[np.min(midx):np.max(midx),
                              np.min(midz):np.max(midz)]

    grid_rho = griddata(np.c_[midx, midz], rho.T, (grid_x, grid_z),
                        method='linear')

    if clim is None:
        vmin, vmax = rho.min(), rho.max()
    else:
        vmin, vmax = clim[0], clim[1]

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(15, 3))

    grid_rho = np.ma.masked_where(np.isnan(grid_rho), grid_rho)
    ph = ax.pcolormesh(
         grid_x[:, 0], grid_z[0, :], grid_rho.T,
         clim=(vmin, vmax), vmin=vmin, vmax=vmax, **pcolorOpts
    )

    if scale == "log":
        cbar = plt.colorbar(
               ph, format="$10^{%.1f}$",
               fraction=0.04, orientation="horizontal"
        )
    elif scale == "linear":
        cbar = plt.colorbar(
               ph, format="%.1f",
               fraction=0.04, orientation="horizontal"
        )

    if data_type == 'appConductivity':
        cbar.set_label("App.Cond", size=12)

    elif data_type == 'appResistivity':
        cbar.set_label("App.Res.", size=12)

    elif data_type == 'volt':
        cbar.set_label("Potential (V)", size=12)

    cmin, cmax = cbar.get_clim()
    ticks = np.linspace(cmin, cmax, 3)
    cbar.set_ticks(ticks)
    cbar.ax.tick_params(labelsize=10)

    # Plot apparent resistivity
    if data_location:
        ax.plot(midx, midz, 'k.', ms=1, alpha=0.4)

    if sameratio:
        ax.set_aspect('equal', adjustable='box')

    return ax


def gen_DCIPsurvey(endl, survey_type, a, b, n, dim=3, d2flag='2.5D'):
    """
        Load in endpoints and survey specifications to generate Tx, Rx location
        stations.

        Assumes flat topo for now...

        Input:
        :param numpy.ndarray endl: input endpoints [x1, y1, z1, x2, y2, z2]
        :param discretize.BaseMesh mesh: discretize mesh object
        :param str survey_type: 'dipole-dipole' | 'pole-dipole' |
            'dipole-pole' | 'pole-pole' | 'gradient'
        :param int a: pole seperation
        :param int b: dipole separation
        :param int n: number of rx dipoles per tx
        :param str d2flag: choose for 2D mesh between a '2D' or a '2.5D' survey

        Output:
        :return SimPEG.EM.Static.DC.SurveyDC.Survey dc_survey: DC survey object
    """

    def xy_2_r(x1, x2, y1, y2):
        r = np.sqrt(np.sum((x2 - x1)**2. + (y2 - y1)**2.))
        return r

    # Evenly distribute electrodes and put on surface
    # Mesure survey length and direction
    dl_len = xy_2_r(endl[0, 0], endl[1, 0], endl[0, 1], endl[1, 1])

    dl_x = (endl[1, 0] - endl[0, 0]) / dl_len
    dl_y = (endl[1, 1] - endl[0, 1]) / dl_len

    nstn = int(np.floor(dl_len / a))

    # Compute discrete pole location along line
    stn_x = endl[0, 0] + np.array(range(int(nstn)))*dl_x*a
    stn_y = endl[0, 1] + np.array(range(int(nstn)))*dl_y*a

    if dim == 2:
        ztop = np.linspace(endl[0, 1], endl[0, 1], nstn)
        # Create line of P1 locations
        M = np.c_[stn_x, ztop]
        # Create line of P2 locations
        N = np.c_[stn_x+a*dl_x, ztop]

    elif dim == 3:
        stn_z = np.linspace(endl[0, 2], endl[0, 2], nstn)
        # Create line of P1 locations
        M = np.c_[stn_x, stn_y, stn_z]
        # Create line of P2 locations
        N = np.c_[stn_x+a*dl_x, stn_y+a*dl_y, stn_z]

    # Build list of Tx-Rx locations depending on survey type
    # Dipole-dipole: Moving tx with [a] spacing -> [AB a MN1 a MN2 ... a MNn]
    # Pole-dipole: Moving pole on one end -> [A a MN1 a MN2 ... MNn a B]
    SrcList = []

    if survey_type != 'gradient':

        for ii in range(0, int(nstn)-1):

            if survey_type == 'dipole-dipole' or survey_type == 'dipole-pole':
                tx = np.c_[M[ii, :], N[ii, :]]
                # Current elctrode separation
                AB = xy_2_r(tx[0, 1], endl[1, 0], tx[1, 1], endl[1, 1])
            elif survey_type == 'pole-dipole' or survey_type == 'pole-pole':
                tx = np.r_[M[ii, :]]
                # Current elctrode separation
                AB = xy_2_r(tx[0], endl[1, 0], tx[1], endl[1, 1])
            else:
                raise Exception(
                    """survey_type must be 'dipole-dipole' | 'pole-dipole' |
                    'dipole-pole' | 'pole-pole'"""
                    " not {}".format(survey_type)
                )

            # Rx.append(np.c_[M[ii+1:indx, :], N[ii+1:indx, :]])

            # Number of receivers to fit
            nstn = int(np.min([np.floor((AB - b) / a), n]))

            # Check if there is enough space, else break the loop
            if nstn <= 0:
                continue

            # Compute discrete pole location along line
            stn_x = N[ii, 0] + dl_x*b + np.array(range(int(nstn)))*dl_x*a
            stn_y = N[ii, 1] + dl_y*b + np.array(range(int(nstn)))*dl_y*a

            # Create receiver poles

            if dim == 3:
                stn_z = np.linspace(endl[0, 2], endl[0, 2], nstn)

                # Create line of P1 locations
                P1 = np.c_[stn_x, stn_y, stn_z]
                # Create line of P2 locations
                P2 = np.c_[stn_x+a*dl_x, stn_y+a*dl_y, stn_z]
                if survey_type == 'dipole-dipole' or survey_type == 'pole-dipole':
                    rxClass = DC.Rx.Dipole(P1, P2)
                elif survey_type == 'dipole-pole' or survey_type == 'pole-pole':
                    rxClass = DC.Rx.Pole(P1)

            elif dim == 2:
                ztop = np.linspace(endl[0, 1], endl[0, 1], nstn)
                # Create line of P1 locations
                P1 = np.c_[stn_x, np.ones(nstn).T*ztop]
                # Create line of P2 locations
                P2 = np.c_[stn_x+a*dl_x, np.ones(nstn).T*ztop]
                if survey_type == 'dipole-dipole' or survey_type == 'pole-dipole':
                    if d2flag == '2.5D':
                        rxClass = DC.Rx.Dipole_ky(P1, P2)
                    elif d2flag == '2D':
                        rxClass = DC.Rx.Dipole(P1, P2)
                elif survey_type == 'dipole-pole' or survey_type == 'pole-pole':
                    if d2flag == '2.5D':
                        rxClass = DC.Rx.Pole_ky(P1)
                    elif d2flag == '2D':
                        rxClass = DC.Rx.Pole(P1)

            if survey_type == 'dipole-dipole' or survey_type == 'dipole-pole':
                srcClass = DC.Src.Dipole([rxClass], M[ii, :], N[ii, :])
            elif survey_type == 'pole-dipole' or survey_type == 'pole-pole':
                srcClass = DC.Src.Pole([rxClass], M[ii, :])
            SrcList.append(srcClass)

    elif survey_type == 'gradient':

        # Gradient survey takes the "b" parameter to define the limits of a
        # square survey grid. The pole seperation within the receiver grid is
        # define the "a" parameter.

        # Get the edge limit of survey area
        min_x = endl[0, 0] + dl_x * b
        min_y = endl[0, 1] + dl_y * b

        max_x = endl[1, 0] - dl_x * b
        max_y = endl[1, 1] - dl_y * b

        # Define the size of the survey grid (square for now)
        box_l = np.sqrt((min_x - max_x)**2. + (min_y - max_y)**2.)
        box_w = box_l/2.

        nstn = int(np.floor(box_l / a))

        # Compute discrete pole location along line
        stn_x = min_x + np.array(range(int(nstn)))*dl_x*a
        stn_y = min_y + np.array(range(int(nstn)))*dl_y*a

        # Define number of cross lines
        nlin = int(np.floor(box_w / a))
        lind = range(-nlin, nlin+1)

        npoles = int(nstn * len(lind))

        rx = np.zeros([npoles, 6])
        for ii in range(len(lind)):

            # Move station location to current survey line This is a
            # perpendicular move then line survey orientation, hence the y, x
            # switch
            lxx = stn_x - lind[ii]*a*dl_y
            lyy = stn_y + lind[ii]*a*dl_x

            M = np.c_[lxx, lyy, np.ones(nstn).T*ztop]
            N = np.c_[lxx+a*dl_x, lyy+a*dl_y, np.ones(nstn).T*ztop]
            rx[(ii*nstn):((ii+1)*nstn), :] = np.c_[M, N]

            if mesh.dim == 3:
                rxClass = DC.Rx.Dipole(rx[:, :3], rx[:, 3:])
            elif mesh.dim == 2:
                M = M[:, [0, 2]]
                N = N[:, [0, 2]]
                if d2flag == '2.5D':
                    rxClass = DC.Rx.Dipole_ky(rx[:, [0, 2]], rx[:, [3, 5]])
                elif d2flag == '2D':
                    rxClass = DC.Rx.Dipole(rx[:, [0, 2]], rx[:, [3, 5]])
            srcClass = DC.Src.Dipole([rxClass],
                                     (endl[0, :]),
                                     (endl[1, :]))
        SrcList.append(srcClass)
    else:
        raise Exception(
            """survey_type must be either 'pole-dipole', 'dipole-dipole',
            'dipole-pole','pole-pole' or 'gradient'"""
            " not {}".format(survey_type)
        )
    if (d2flag == '2.5D') and (dim == 2):
        survey = DC.Survey_ky(SrcList)
    else:
        survey = DC.Survey(SrcList)

    return survey


def writeUBC_DCobs(
    fileName, dc_survey, dim, format_type,
    survey_type='dipole-dipole', ip_type=0,
    comment_lines=''
):
    """
        Write UBC GIF DCIP 2D or 3D observation file

        Input:
        :param str fileName: including path where the file is written out
        :param SimPEG.EM.Static.DC.SurveyDC.Survey dc_survey: DC survey object
        :param int dim:  either 2 | 3
        :param str format_type:  either 'SURFACE' | 'GENERAL'
        :param str survey_type: 'dipole-dipole' | 'pole-dipole' |
            'dipole-pole' | 'pole-pole' | 'gradient'

        Output:
        :return: UBC2D-Data file
        :rtype: file
    """

    if not((dim == 2) | (dim == 3)):
        raise Exception(
            """dim must be either 2 or 3"""
            " not {}".format(dim)
        )

    if not (
        (format_type == 'SURFACE') |
        (format_type == 'GENERAL') |
        (format_type == 'SIMPLE')
    ):
        raise Exception(
            """format_type must be 'SURFACE' | 'GENERAL' |
            'SIMPLE' | 'pole-pole'"""
            " not {}".format(format_type)
        )

    if(isinstance(dc_survey.std, float)):
        print(
            """survey.std was a float computing uncertainty vector
            (survey.std*survey.dobs + survey.eps)"""
        )

    if(isinstance(dc_survey.eps, float)):
        epsValue = dc_survey.eps
        dc_survey.eps = epsValue*np.ones_like(dc_survey.dobs)

    fid = open(fileName, 'w')

    if format_type in ['SURFACE', 'GENERAL'] and dim == 2:
        fid.write('COMMON_CURRENT\n')

    fid.write('! ' + format_type + ' FORMAT\n')

    if comment_lines:
        fid.write(comment_lines)

    if dim == 2:
        fid.write('{:d}\n'.format(dc_survey.nSrc))

    if ip_type != 0:
        fid.write('IPTYPE=%i\n' % ip_type)

    fid.close()

    count = 0

    for ii in range(dc_survey.nSrc):

        rx = dc_survey.srcList[ii].rxList[0].locs
        nD = dc_survey.srcList[ii].nD

        if survey_type == 'pole-dipole' or survey_type == 'pole-pole':
            tx = np.r_[dc_survey.srcList[ii].loc]
            tx = np.repeat(np.r_[[tx]], 2, axis=0)
        elif survey_type == 'dipole-dipole' or survey_type == 'dipole-pole':
            tx = np.c_[dc_survey.srcList[ii].loc]

        if survey_type == 'pole-dipole' or survey_type == 'dipole-dipole':
            M = rx[0]
            N = rx[1]
        elif survey_type == 'pole-pole' or survey_type == 'dipole-pole':
            M = rx
            N = rx

        # Adapt source-receiver location for dim and survey_type
        if dim == 2:

            if format_type == 'SIMPLE':

                # fid.writelines("%e " % ii for ii in Utils.mkvc(tx[0, :]))
                A = np.repeat(tx[0, 0], M.shape[0], axis=0)

                if survey_type == 'pole-dipole':
                    B = np.repeat(tx[0, 0], M.shape[0], axis=0)

                else:
                    B = np.repeat(tx[1, 0], M.shape[0], axis=0)

                M = M[:, 0]
                N = N[:, 0]

                fid = open(fileName, 'ab')
                np.savetxt(
                    fid,
                    np.c_[
                        A, B, M, N,
                        dc_survey.dobs[count:count+nD],
                        dc_survey.std[count:count+nD]
                    ],
                    delimiter=str(' '), newline=str('\n'))
                fid.close()

            else:
                fid = open(fileName, 'a')
                if format_type == 'SURFACE':

                    fid.writelines("%f " % ii for ii in Utils.mkvc(tx[:, 0]))
                    M = M[:, 0]
                    N = N[:, 0]

                if format_type == 'GENERAL':

                    # Flip sign for z-elevation to depth
                    tx[2::2, :] = -tx[2::2, :]

                    fid.writelines(('{:e} {:e} ').format(ii, jj) for ii, jj in tx[:, :2])
                    M = M[:, :2]
                    N = N[:, :2]

                    # Flip sign for z-elevation to depth
                    M[:, 1::2] = -M[:, 1::2]
                    N[:, 1::2] = -N[:, 1::2]

                fid.write('%i\n' % nD)
                fid.close()

                fid = open(fileName, 'ab')
                np.savetxt(
                    fid,
                    np.c_[
                        M, N,
                        dc_survey.dobs[count:count+nD],
                        dc_survey.std[count:count+nD]
                    ],
                    delimiter=str(' '), newline=str('\n'))

        if dim == 3:
            fid = open(fileName, 'a')
            # Flip sign of z value for UBC DCoctree code
            tx[:, 2] = -tx[:, 2]
            # print(tx)

            # Flip sign of z value for UBC DCoctree code
            M[:, 2] = -M[:, 2]
            N[:, 2] = -N[:, 2]

            if format_type == 'SURFACE':

                fid.writelines("%e " % ii for ii in Utils.mkvc(tx[:, 0:2].T))
                M = M[:, 0:2]
                N = N[:, 0:2]

            if format_type == 'GENERAL':

                fid.writelines("%e " % ii for ii in Utils.mkvc(tx.T))

            fid.write('%i\n' % nD)

            fid.close()

            fid = open(fileName, 'ab')
            if isinstance(dc_survey.std, np.ndarray):
                np.savetxt(
                    fid,
                    np.c_[
                        M, N,
                        dc_survey.dobs[count:count+nD],
                        dc_survey.std[count:count+nD] + dc_survey.eps[count:count+nD]
                    ],
                    fmt=str('%e'), delimiter=str(' '), newline=str('\n')
                )
            elif (isinstance(dc_survey.std, float)):
                np.savetxt(
                    fid,
                    np.c_[
                        M, N,
                        dc_survey.dobs[count:count+nD],
                        dc_survey.std*np.abs(dc_survey.dobs[count:count+nD]) + dc_survey.eps[count:count+nD]
                    ],
                    fmt=str('%e'), delimiter=str(' '), newline=str('\n')
                )
            else:
                raise Exception(
                    """Uncertainities SurveyObject.std should be set.
                    Either float or nunmpy.ndarray is expected, """
                    "not {}".format(type(dc_survey.std)))

            fid.close()

            fid = open(fileName, 'a')
            fid.write('\n')
            fid.close()

        count += nD

    fid.close()


def writeUBC_DClocs(
    fileName, dc_survey, dim, format_type,
    survey_type='dipole-dipole', ip_type=0,
    comment_lines=''):
    """
        Write UBC GIF DCIP 2D or 3D locations file

        Input:
        :param str fileName: including path where the file is written out
        :param SimPEG.EM.Static.DC.SurveyDC.Survey dc_survey: DC survey object
        :param int dim:  either 2 | 3
        :param str survey_type:  either 'SURFACE' | 'GENERAL'

        Output:
        :rtype: file
        :return: UBC 2/3D-locations file
    """

    if not((dim == 2) | (dim == 3)):
        raise Exception(
            """dim must be either 2 or 3"""
            " not {}".format(dim)
        )

    if not (
        (format_type == 'SURFACE') |
        (format_type == 'GENERAL') |
        (format_type == 'SIMPLE')
    ):
        raise Exception(
            """format_type must be 'SURFACE' | 'GENERAL' |
            'SIMPLE' | 'pole-pole'"""
            " not {}".format(format_type)
        )

    if(isinstance(dc_survey.std, float)):
        print(
            """survey.std was a float computing uncertainty vector
            (survey.std*survey.dobs + survey.eps)"""
        )

    if(isinstance(dc_survey.eps, float)):
        epsValue = dc_survey.eps
        dc_survey.eps = epsValue*np.ones_like(dc_survey.dobs)

    fid = open(fileName, 'w')

    if format_type in ['SURFACE', 'GENERAL'] and dim == 2:
        fid.write('COMMON_CURRENT\n')

    fid.write('! ' + format_type + ' FORMAT\n')

    if comment_lines:
        fid.write(comment_lines)

    if dim == 2:
        fid.write('{:d}\n'.format(dc_survey.nSrc))

    if ip_type != 0:
        fid.write('IPTYPE=%i\n' % ip_type)

    fid.close()

    count = 0

    for ii in range(dc_survey.nSrc):

        rx = dc_survey.srcList[ii].rxList[0].locs
        nD = dc_survey.srcList[ii].nD

        if survey_type == 'pole-dipole' or survey_type == 'pole-pole':
            tx = np.r_[dc_survey.srcList[ii].loc]
            tx = np.repeat(np.r_[[tx]], 2, axis=0)
        elif survey_type == 'dipole-dipole' or survey_type == 'dipole-pole':
            tx = np.c_[dc_survey.srcList[ii].loc]

        if survey_type == 'pole-dipole' or survey_type == 'dipole-dipole':
            M = rx[0]
            N = rx[1]
        elif survey_type == 'pole-pole' or survey_type == 'dipole-pole':
            M = rx
            N = rx

        # Adapt source-receiver location for dim and survey_type
        if dim == 2:

            if format_type == 'SIMPLE':

                # fid.writelines("%e " % ii for ii in Utils.mkvc(tx[0, :]))
                A = np.repeat(tx[0, 0], M.shape[0], axis=0)

                if survey_type == 'pole-dipole':
                    B = np.repeat(tx[0, 0], M.shape[0], axis=0)

                else:
                    B = np.repeat(tx[1, 0], M.shape[0], axis=0)

                M = M[:, 0]
                N = N[:, 0]

                fid = open(fileName, 'ab')
                np.savetxt(
                    fid,
                    np.c_[
                        A, B, M, N,
                    ],
                    delimiter=str(' '), newline=str('\n'))
                fid.close()

            else:
                fid = open(fileName, 'a')
                if format_type == 'SURFACE':

                    fid.writelines("%f " % ii for ii in Utils.mkvc(tx[:, 0]))
                    M = M[:, 0]
                    N = N[:, 0]

                if format_type == 'GENERAL':

                    # Flip sign for z-elevation to depth
                    tx[2::2, :] = -tx[2::2, :]

                    fid.writelines(('{:e} {:e} ').format(ii, jj) for ii, jj in tx[:, :2])
                    M = M[:, :2]
                    N = N[:, :2]

                    # Flip sign for z-elevation to depth
                    M[:, 1::2] = -M[:, 1::2]
                    N[:, 1::2] = -N[:, 1::2]

                fid.write('%i\n' % nD)
                fid.close()

                fid = open(fileName, 'ab')
                np.savetxt(
                    fid,
                    np.c_[
                        M, N,
                    ],
                    delimiter=str(' '), newline=str('\n'))

        if dim == 3:
            fid = open(fileName, 'a')
            # Flip sign of z value for UBC DCoctree code
            tx[:, 2] = -tx[:, 2]
            # print(tx)

            # Flip sign of z value for UBC DCoctree code
            M[:, 2] = -M[:, 2]
            N[:, 2] = -N[:, 2]

            if format_type == 'SURFACE':

                fid.writelines("%e " % ii for ii in Utils.mkvc(tx[:, 0:2].T))
                M = M[:, 0:2]
                N = N[:, 0:2]

            if format_type == 'GENERAL':

                fid.writelines("%e " % ii for ii in Utils.mkvc(tx.T))

            fid.write('%i\n' % nD)

            fid.close()

            fid = open(fileName, 'ab')
            if isinstance(dc_survey.std, np.ndarray):
                np.savetxt(
                    fid,
                    np.c_[
                        M, N,
                    ],
                    fmt=str('%e'), delimiter=str(' '), newline=str('\n')
                )
            elif (isinstance(dc_survey.std, float)):
                np.savetxt(
                    fid,
                    np.c_[
                        M, N,
                    ],
                    fmt=str('%e'), delimiter=str(' '), newline=str('\n')
                )

            fid.close()

            fid = open(fileName, 'a')
            fid.write('\n')
            fid.close()

        count += nD

    fid.close()


def convertObs_DC3D_to_2D(survey, lineID, flag='local'):
    """
        Read DC survey and projects the coordinate system
        according to the flag = 'Xloc' | 'Yloc' | 'local' (default)
        In the 'local' system, station coordinates are referenced
        to distance from the first srcLoc[0].loc[0]

        The Z value is preserved, but Y coordinates zeroed.

        Input:
        :param survey: 3D DC survey class object
        :rtype: SimPEG.EM.Static.DC.SurveyDC.Survey

        Output:
        :param survey: 2D DC survey class object
        :rtype: SimPEG.EM.Static.DC.SurveyDC.Survey
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

        assert len(p1) == len(p2), 'locs must be the same shape.'

        dx = []
        for ii in range(len(p1)):
            dx.append((p2[ii] - p1[ii]))

        # Compute length of vector
        r = np.linalg.norm(np.asarray(dx))

        if r != 0:
            vec = dx/r

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
        r = 1e+8  # Initialize to some large number

        Tx = srcMat[indx]

        if np.all(Tx[0:3] == Tx[3:]):
            survey_type = 'pole-dipole'

        else:
            survey_type = 'dipole-dipole'

        x0 = Tx[0][0:2]  # Define station zero along line

        vecTx, r1 = r_unit(x0, Tx[-1][0:2])

        for ii in range(len(indx)):

            # Get all receivers
            Rx = survey.srcList[indx[ii]].rxList[0].locs
            nrx = Rx[0].shape[0]

            if flag == 'local':
                # Find A electrode along line
                vec, r = r_unit(x0, Tx[ii][0:2])
                A = stn_id(vecTx, vec, r)

                if survey_type != 'pole-dipole':
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
            elif flag == 'Yloc':
                """ Flip the XY axis locs"""
                A = Tx[ii][1]

                if survey_type != 'pole-dipole':
                    B = Tx[ii][4]

                M = Rx[0][:, 1]
                N = Rx[1][:, 1]

            elif flag == 'Xloc':
                """ Copy the rx-tx locs"""
                A = Tx[ii][0]

                if survey_type != 'pole-dipole':
                    B = Tx[ii][3]

                M = Rx[0][:, 0]
                N = Rx[1][:, 0]

            rxClass = DC.Rx.Dipole(np.c_[M, np.zeros(nrx), Rx[0][:, 2]],
                                   np.c_[N, np.zeros(nrx), Rx[1][:, 2]])

            if survey_type == 'pole-dipole':
                srcList2D.append(
                    DC.Src.Pole(
                        [rxClass],
                        np.asarray([A, 0, Tx[ii][2]])
                    )
                )

            elif survey_type == 'dipole-dipole':
                srcList2D.append(
                    DC.Src.Dipole(
                        [rxClass],
                        np.r_[A, 0, Tx[ii][2]],
                        np.r_[B, 0, Tx[ii][5]]
                    )
                )

    survey2D = DC.SurveyDC.Survey(srcList2D)
    survey2D.dobs = survey.dobs
    survey2D.std = survey.std

    return survey2D


def readUBC_DC2Dpre(fileName):
    """
        Read UBC GIF DCIP 2D observation file and generate arrays
        for tx-rx location

        Input:
        :param string fileName: path to the UBC GIF 3D obs file

        Output:
        :return survey: 2D DC survey class object
        :rtype: SimPEG.EM.Static.DC.SurveyDC.Survey

        Created on Mon March 9th, 2016 << Doug's 70th Birthday !! >>

        @author: dominiquef

    """

    # Load file
    obsfile = np.genfromtxt(
        fileName, delimiter=' \n',
        dtype=np.str, comments='!'
    )

    # Pre-allocate
    srcLists = []
    Rx = []
    d = []
    zflag = True  # Flag for z value provided

    for ii in range(obsfile.shape[0]):

        if not obsfile[ii]:
            continue

        # First line is transmitter with number of receivers

        temp = (np.fromstring(obsfile[ii], dtype=float, sep=' ').T)

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

        Rx = DC.Rx.Dipole(rx[:, :3], rx[:, 3:])
        srcLists.append(DC.Src.Dipole([Rx], tx[:3], tx[3:]))

    # Create survey class
    survey = DC.SurveyDC.Survey(srcLists)

    survey.dobs = np.asarray(d)

    return {'dc_survey': survey}


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
    obsfile = np.genfromtxt(
        fileName, delimiter=' \n',
        dtype=np.str, comments='!'
    )

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
            temp = (np.fromstring(obsfile[ii], dtype=float, sep=' ').T)
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

        temp = np.fromstring(obsfile[ii], dtype=float, sep=' ')

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
                Rx = DC.Rx.Pole(rx[:, :3])
            else:
                Rx = DC.Rx.Dipole(rx[:, :3], rx[:, 3:])
            if poletx:
                srcLists.append(DC.Src.Pole([Rx], tx[:3]))
            else:
                srcLists.append(DC.Src.Dipole([Rx], tx[:3], tx[3:]))

    survey = DC.SurveyDC.Survey(srcLists)
    survey.dobs = np.asarray(d)
    survey.std = np.asarray(wd)
    survey.eps = 0.

    return {'dc_survey': survey}


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

            A = dc_survey.srcList[ii].loc[0]
            B = dc_survey.srcList[ii].loc[1]

            xout = np.mean([A[0:2], B[0:2]], axis=0)

            xy0 = A[:2]
            xym = xout

            # Deal with replicate pole location
            if np.all(xy0 == xym):

                xym[0] = xym[0] + 1e-3

            continue

        A = dc_survey.srcList[ii].loc[0]
        B = dc_survey.srcList[ii].loc[1]

        xin = np.mean([A[0:2], B[0:2]], axis=0)

        # Compute vector between neighbours
        vec1, r1 = r_unit(xout, xin)

        # Compute vector between current stn and mid-point
        vec2, r2 = r_unit(xym, xin)

        # Compute vector between current stn and start line
        vec3, r3 = r_unit(xy0, xin)

        # Compute vector between mid-point and start line
        vec4, r4 = r_unit(xym, xy0)

        # Compute dot product
        ang1 = np.abs(vec1.dot(vec2))
        ang2 = np.abs(vec3.dot(vec4))

        # If the angles are smaller then 45d, than next point is on a new line
        if ((ang1 < np.cos(np.pi/4.)) | (ang2 < np.cos(np.pi/4.))) & (np.all(np.r_[r1, r2, r3, r4] > 0)):

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

    assert len(p1) == len(p2), 'locs must be the same shape.'

    dx = []
    for ii in range(len(p1)):
        dx.append((p2[ii] - p1[ii]))

    # Compute length of vector
    r = np.linalg.norm(np.asarray(dx))

    if r != 0:
        vec = dx/r

    else:
        vec = np.zeros(len(p1))

    return vec, r


def getSrc_locs(survey):
    """
        Read in a DC survey class and extract the xyz location of all
        sources.

        Input:
        :param survey: DC survey class object
        :rtype: SimPEG.EM.Static.DC.SurveyDC.Survey

        Output:
        :return numpy.array srcMat: Array containing the locations of sources

    """

    srcMat = []

    for src in survey.srcList:

        srcMat.append(np.hstack(src.loc))

    srcMat = np.vstack(srcMat)

    return srcMat


def gettopoCC(mesh, actind, option="top"):
    """
        Get topography from active indices of mesh.

    """

    if mesh._meshType == "TENSOR":

        if mesh.dim == 3:

            mesh2D = Mesh.TensorMesh([mesh.hx, mesh.hy], mesh.x0[:2])
            zc = mesh.gridCC[:, 2]
            ACTIND = actind.reshape(
                (mesh.vnC[0]*mesh.vnC[1], mesh.vnC[2]),
                order='F'
                )
            ZC = zc.reshape((mesh.vnC[0]*mesh.vnC[1], mesh.vnC[2]), order='F')
            topoCC = np.zeros(ZC.shape[0])

            for i in range(ZC.shape[0]):
                ind = np.argmax(ZC[i, :][ACTIND[i, :]])
                if option == "top":
                    dz = mesh.hz[ACTIND[i, :]][ind] * 0.5
                elif option == "center":
                    dz = 0.
                else:
                    raise Exception()
                topoCC[i] = (
                    ZC[i, :][ACTIND[i, :]].max() + dz
                    )
            return mesh2D, topoCC

        elif mesh.dim == 2:

            mesh1D = Mesh.TensorMesh([mesh.hx], [mesh.x0[0]])
            yc = mesh.gridCC[:, 1]
            ACTIND = actind.reshape((mesh.vnC[0], mesh.vnC[1]), order='F')
            YC = yc.reshape((mesh.vnC[0], mesh.vnC[1]), order='F')
            topoCC = np.zeros(YC.shape[0])
            for i in range(YC.shape[0]):
                ind = np.argmax(YC[i, :][ACTIND[i, :]])
                if option == "top":
                    dy = mesh.hy[ACTIND[i, :]][ind] * 0.5
                elif option == "center":
                    dy = 0.
                else:
                    raise Exception()
                topoCC[i] = (
                    YC[i, :][ACTIND[i, :]].max() + dy
                    )
            return mesh1D, topoCC

    elif mesh._meshType == "TREE":
        if mesh.dim == 3:
            uniqXY = uniqueRows(mesh.gridCC[:, :2])
            npts = uniqXY[0].shape[0]
            ZC = mesh.gridCC[:, 2]
            topoCC = np.zeros(npts)
            if option == "top":
                # TODO: this assume same hz, need to be modified
                dz = mesh.hz.min() * 0.5
            elif option == "center":
                dz = 0.
            for i in range(npts):
                inds = uniqXY[2] == i
                actind_z = actind[inds]
                if actind_z.sum() > 0.:
                    topoCC[i] = (ZC[inds][actind_z]).max() + dz
                else:
                    topoCC[i] = (ZC[inds]).max() + dz
            return uniqXY[0], topoCC
        else:
            raise NotImplementedError(
                "gettopoCC is not implemented for Quad tree mesh"
                )


def drapeTopotoLoc(mesh, pts, actind=None, option="top", topo=None):
    """
        Drape location right below (cell center) the topography
    """
    if mesh.dim == 2:
        if pts.ndim > 1:
            raise Exception("pts should be 1d array")
    elif mesh.dim == 3:
        if pts.shape[1] == 3:
            raise Exception("shape of pts should be (x,3)")
    else:
        raise NotImplementedError()
    if actind is None:
        actind = Utils.surface2ind_topo(mesh, topo)
    if mesh._meshType == "TENSOR":
        meshtemp, topoCC = gettopoCC(mesh, actind, option=option)
        inds = Utils.closestPoints(meshtemp, pts)

    elif mesh._meshType == "TREE":
        if mesh.dim == 3:
            uniqXYlocs, topoCC = gettopoCC(mesh, actind, option=option)
            inds = closestPointsGrid(uniqXYlocs, pts)
        else:
            raise NotImplementedError()
    else:
        raise NotImplementedError()
    out = np.c_[pts, topoCC[inds]]
    return out


def genTopography(mesh, zmin, zmax, seed=None, its=100, anisotropy=None):
    if mesh.dim == 3:
        mesh2D = Mesh.TensorMesh(
            [mesh.hx, mesh.hy], x0=[mesh.x0[0], mesh.x0[1]]
            )
        out = Utils.ModelBuilder.randomModel(
            mesh.vnC[:2], bounds=[zmin, zmax], its=its,
            seed=seed, anisotropy=anisotropy
            )
        return out, mesh2D
    elif mesh.dim == 2:
        mesh1D = Mesh.TensorMesh([mesh.hx], x0=[mesh.x0[0]])
        out = Utils.ModelBuilder.randomModel(
            mesh.vnC[:1], bounds=[zmin, zmax], its=its,
            seed=seed, anisotropy=anisotropy
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

    pts = asArray_N_x_Dim(pts, dim)
    nodeInds = np.empty(pts.shape[0], dtype=int)

    for i, pt in enumerate(pts):
        if dim == 1:
            nodeInds[i] = ((pt - grid)**2.).argmin()
        else:
            nodeInds[i] = (
                (np.tile(
                    pt, (grid.shape[0], 1)) - grid)**2.).sum(axis=1).argmin()

    return nodeInds
