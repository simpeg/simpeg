from SimPEG import np
from SimPEG.EM.Static import DC, IP

def plot_pseudoSection(DCsurvey, axs, surveyType='dipole-dipole', unitType="appc", clim=None, colorbar=True):
    """
        Read list of 2D tx-rx location and plot a speudo-section of apparent
        resistivity.

        Assumes flat topo for now...

        Input:
        :param d2D, z0
        :switch surveyType -> Either 'pdp' (pole-dipole) | 'dipole-dipole' (dipole-dipole)
        :switch unitType=-> Either 'appResistivity' (app. res) | 'appConductivity' (app. con) | 'volt' (potential)
        Output:
        :figure scatter plot overlayed on image

        Edited Feb 17th, 2016

        @author: dominiquef

    """
    from SimPEG import np
    from scipy.interpolate import griddata
    import pylab as plt

    # Set depth to 0 for now
    z0 = 0.

    # Pre-allocate
    midx = []
    midz = []
    rho = []
    LEG = []
    count = 0 # Counter for data
    for ii in range(DCsurvey.nSrc):

        Tx = DCsurvey.srcList[ii].loc
        Rx = DCsurvey.srcList[ii].rxList[0].locs

        nD = DCsurvey.srcList[ii].rxList[0].nD

        data = DCsurvey.dobs[count:count+nD]
        count += nD

        # Get distances between each poles A-B-M-N
        if surveyType == 'pole-dipole':
            MA = np.abs(Tx[0] - Rx[0][:,0])
            NA = np.abs(Tx[0] - Rx[1][:,0])
            MN = np.abs(Rx[1][:,0] - Rx[0][:,0])

            # Create mid-point location
            Cmid = Tx[0]
            Pmid = (Rx[0][:,0] + Rx[1][:,0])/2
            if DCsurvey.mesh.dim == 2:
                zsrc = Tx[1]
            elif DCsurvey.mesh.dim ==3:
                zsrc = Tx[2]

        elif surveyType == 'dipole-dipole':
            MA = np.abs(Tx[0] - Rx[0][:,0])
            MB = np.abs(Tx[3] - Rx[0][:,0])
            NA = np.abs(Tx[0] - Rx[1][:,0])
            NB = np.abs(Tx[3] - Rx[1][:,0])

            # Create mid-point location
            Cmid = (Tx[0] + Tx[3])/2
            Pmid = (Rx[0][:,0] + Rx[1][:,0])/2
            if DCsurvey.mesh.dim == 2:
                zsrc = (Tx[0][1] + Tx[1][1])/2
            elif DCsurvey.mesh.dim ==3:
                zsrc = (Tx[2] + Tx[5])/2

        # Change output for unitType
        if unitType == 'volt':

            rho = np.hstack([rho,data])

        else:

            # Compute pant leg of apparent rho
            if surveyType == 'pole-dipole':

                leg =  data * 2*np.pi  * MA * ( MA + MN ) / MN

            elif surveyType == 'dipole-dipole':

                leg = data * 2*np.pi / ( 1/MA - 1/MB + 1/NB - 1/NA )
                LEG.append(1./(2*np.pi) *( 1/MA - 1/MB + 1/NB - 1/NA ))
            else:
                print """unitType must be 'pole-dipole'(pole-dipole) | 'dipole-dipole' (dipole-dipole) """
                break


            if unitType == 'appConductivity':

                leg = np.log10(abs(1./leg))
                rho = np.hstack([rho,leg])

            elif unitType == 'appResistivity':

                leg = np.log10(abs(leg))
                rho = np.hstack([rho,leg])

            else:
                print """unitType must be 'appResistivity' | 'appConductivity' | 'volt' """
                break


        midx = np.hstack([midx, ( Cmid + Pmid )/2 ])
        if DCsurvey.mesh.dim==3:
            midz = np.hstack([midz, -np.abs(Cmid-Pmid)/2 + zsrc ])
        elif DCsurvey.mesh.dim==2:
            midz = np.hstack([midz, -np.abs(Cmid-Pmid)/2 + zsrc ])
    ax = axs

    # Grid points
    grid_x, grid_z = np.mgrid[np.min(midx):np.max(midx), np.min(midz):np.max(midz)]
    grid_rho = griddata(np.c_[midx,midz], rho.T, (grid_x, grid_z), method='linear')

    if clim == None:
        vmin, vmax = rho.min(), rho.max()
    else:
        vmin, vmax = clim[0], clim[1]

    grid_rho = np.ma.masked_where(np.isnan(grid_rho), grid_rho)
    ph = plt.pcolormesh(grid_x[:,0],grid_z[0,:],grid_rho.T, clim=(vmin, vmax), vmin=vmin, vmax=vmax)
    plt.contour(grid_rho.T,5, extent = (np.min(grid_x),np.max(grid_x),np.min(grid_z),np.max(grid_z))  ,origin='lower',colors='k')
    
    if colorbar:
        cbar = plt.colorbar(format="$10^{%.1f}$",fraction=0.04,orientation="horizontal")

        cmin,cmax = cbar.get_clim()
        ticks = np.linspace(cmin,cmax,3)
        cbar.set_ticks(ticks)
        cbar.ax.tick_params(labelsize=10)

        if unitType == 'appConductivity':
            cbar.set_label("App.Cond",size=12)
        elif unitType == 'appResistivity':
            cbar.set_label("App.Res.",size=12)
        elif unitType == 'volt':
            cbar.set_label("Potential (V)",size=12)

    # Plot apparent resistivity
    ax.scatter(midx,midz,s=10,c=rho.T, vmin =vmin, vmax = vmax, clim=(vmin, vmax))

    #ax.set_xticklabels([])
    #ax.set_yticklabels([])

    plt.gca().set_aspect('equal', adjustable='box')



    return ph, LEG, midx, midz

def gen_DCIPsurvey(endl, mesh, stype, a, b, n):
    """
        Load in endpoints and survey specifications to generate Tx, Rx location
        stations.

        Assumes flat topo for now...

        Input:
        :param endl -> input endpoints [x1, y1, z1, x2, y2, z2]
        :object mesh -> SimPEG mesh object
        :switch stype -> "dipole-dipole" (dipole-dipole) | "pdp" (pole-dipole) | 'gradient'
        : param a, n -> pole seperation, number of rx dipoles per tx

        Output:
        :param Tx, Rx -> List objects for each tx location
            Lines: P1x, P1y, P1z, P2x, P2y, P2z

        Created on Wed December 9th, 2015

        @author: dominiquef
        !! Require clean up to deal with DCsurvey
    """

    from SimPEG import np

    def xy_2_r(x1, x2, y1, y2):
        r = np.sqrt(np.sum((x2 - x1)**2 + (y2 - y1)**2))
        return r

    # Evenly distribute electrodes and put on surface
    # Mesure survey length and direction
    dl_len = xy_2_r(endl[0, 0], endl[1, 0], endl[0, 1], endl[1, 1])

    dl_x = (endl[1, 0] - endl[0, 0]) / dl_len
    dl_y = (endl[1, 1] - endl[0, 1]) / dl_len

    nstn = np.floor(dl_len / a)

    # Compute discrete pole location along line
    stn_x = endl[0, 0] + np.array(range(int(nstn)))*dl_x*a
    stn_y = endl[0, 1] + np.array(range(int(nstn)))*dl_y*a

    if mesh.dim == 2:
        ztop = mesh.vectorNy[-1]
        # Create line of P1 locations
        M = np.c_[stn_x, np.ones(nstn).T*ztop]
        # Create line of P2 locations
        N = np.c_[stn_x+a*dl_x, np.ones(nstn).T*ztop]

    elif mesh.dim == 3:
        ztop = mesh.vectorNz[-1]
        # Create line of P1 locations
        M = np.c_[stn_x, stn_y, np.ones(nstn).T*ztop]
        # Create line of P2 locations
        N = np.c_[stn_x+a*dl_x, stn_y+a*dl_y, np.ones(nstn).T*ztop]

    # Build list of Tx-Rx locations depending on survey type
    # Dipole-dipole: Moving tx with [a] spacing -> [AB a MN1 a MN2 ... a MNn]
    # Pole-dipole: Moving pole on one end -> [A a MN1 a MN2 ... MNn a B]
    SrcList = []


    if stype != 'gradient':

        for ii in range(0, int(nstn)-1):


            if stype == 'dipole-dipole':
                tx = np.c_[M[ii,:],N[ii,:]]
            elif stype == 'pole-dipole':
                tx = np.c_[M[ii,:],M[ii,:]]
            else:
                raise Exception('The stype must be "dipole-dipole" or "pole-dipole"')

            # Rx.append(np.c_[M[ii+1:indx,:],N[ii+1:indx,:]])

            # Current elctrode seperation
            AB = xy_2_r(tx[0,1],endl[1,0],tx[1,1],endl[1,1])

            # Number of receivers to fit
            nstn = np.min([np.floor( (AB - b) / a ) , n])

            # Check if there is enough space, else break the loop
            if nstn <= 0:
                continue

            # Compute discrete pole location along line
            stn_x = N[ii,0] + dl_x*b + np.array(range(int(nstn)))*dl_x*a
            stn_y = N[ii,1] + dl_y*b + np.array(range(int(nstn)))*dl_y*a

            # Create receiver poles

            if mesh.dim==3:
                # Create line of P1 locations
                P1 = np.c_[stn_x, stn_y, np.ones(nstn).T*ztop]
                # Create line of P2 locations
                P2 = np.c_[stn_x+a*dl_x, stn_y+a*dl_y, np.ones(nstn).T*ztop]
                rxClass = DC.Rx.Dipole(P1, P2)

            elif mesh.dim==2:
                # Create line of P1 locations
                P1 = np.c_[stn_x, np.ones(nstn).T*ztop]
                # Create line of P2 locations
                P2 = np.c_[stn_x+a*dl_x, np.ones(nstn).T*ztop]
                rxClass = DC.Rx.Dipole_ky(P1, P2)

            if stype == 'dipole-dipole':
                srcClass = DC.Src.Dipole([rxClass], M[ii,:],N[ii,:])
            elif stype == 'pole-dipole':
                srcClass = DC.Src.Pole([rxClass], M[ii,:])
            SrcList.append(srcClass)

    elif stype == 'gradient':

        # Gradient survey takes the "b" parameter to define the limits of a
        # square survey grid. The pole seperation within the receiver grid is
        # define the "a" parameter.

        # Get the edge limit of survey area
        min_x = endl[0, 0] + dl_x * b
        min_y = endl[0, 1] + dl_y * b

        max_x = endl[1, 0] - dl_x * b
        max_y = endl[1, 1] - dl_y * b

        # Define the size of the survey grid (square for now)
        box_l = np.sqrt((min_x - max_x)**2 + (min_y - max_y)**2)
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
            # perpendicular move then line survey orientation, hence the y,x
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
                rxClass = DC.Rx.Dipole_ky(rx[:, [0, 2]], rx[:, [3, 5]])
            srcClass = DC.Src.Dipole([rxClass],
                                     (endl[0, :]),
                                     (endl[1, :]))
        SrcList.append(srcClass)
    else:
        print """stype must be either 'pole-dipole', 'dipole-dipole' or 'gradient'. """

    return SrcList


def writeUBC_DCobs(fileName, DCsurvey, dim, surveyType, iptype=0):
    """
        Write UBC GIF DCIP 2D or 3D observation file

        :param string fileName: including path where the file is written out
        :param Survey DCsurvey: DC survey class object
        :param string dim:  either '2D' | '3D'
        :param string surveyType:  either 'SURFACE' | 'GENERAL'
        :rtype: file
        :return: UBC2D-Data file
    """

    from SimPEG import mkvc

    assert (dim == '2D') | (dim == '3D'), "Data must be either '2D' | '3D'"

    assert ((surveyType == 'SURFACE') |
            (surveyType == 'GENERAL') |
            (surveyType == 'SIMPLE')), "Data must be either 'SURFACE' | 'GENERAL' | 'SIMPLE'"

    fid = open(fileName, 'w')
    fid.write('! ' + surveyType + ' FORMAT\n')

    if iptype != 0:
        fid.write('IPTYPE=%i\n' % iptype)

    else:
        fid.write('! ' + surveyType + ' FORMAT\n')

    count = 0

    for ii in range(DCsurvey.nSrc):

        tx = np.c_[DCsurvey.srcList[ii].loc]

        if np.shape(tx)[0] == 3:
            stype = 'pole-dipole'

        else:
            stype = 'dipole-dipole'

        rx = DCsurvey.srcList[ii].rxList[0].locs

        nD = DCsurvey.srcList[ii].nD

        M = rx[0]
        N = rx[1]

        # Adapt source-receiver location for dim and surveyType
        if dim == '2D':

            if surveyType == 'SIMPLE':

                # fid.writelines("%e " % ii for ii in mkvc(tx[0,:]))
                A = np.repeat(tx[0], M.shape[0], axis=0)

                if stype == 'pole-dipole':
                    B = np.repeat(tx[0], M.shape[0], axis=0)

                else:
                    B = np.repeat(tx[3], M.shape[0], axis=0)

                M = M[:, 0]
                N = N[:, 0]

                np.savetxt(fid, np.c_[A, B, M, N,
                                      DCsurvey.dobs[count:count+nD],
                                      DCsurvey.std[count:count+nD]],
                           fmt='%e', delimiter=' ', newline='\n')

            else:

                if surveyType == 'SURFACE':

                    fid.writelines("%f " % ii for ii in mkvc(tx[0,:]))
                    M = M[:,0]
                    N = N[:,0]

                if surveyType == 'GENERAL':

                    # Flip sign for z-elevation to depth
                    tx[2::2,:] = -tx[2::2,:]

                    fid.writelines("%e " % ii for ii in mkvc(tx[::2,:]))
                    M = M[:,0::2]
                    N = N[:,0::2]

                    # Flip sign for z-elevation to depth
                    M[:,1::2] = -M[:,1::2]
                    N[:,1::2] = -N[:,1::2]

                fid.write('%i\n'% nD)
                np.savetxt(fid, np.c_[ M, N , DCsurvey.dobs[count:count+nD], DCsurvey.std[count:count+nD] ], fmt='%f',delimiter=' ',newline='\n')

        if dim=='3D':

            if surveyType == 'SURFACE':

                fid.writelines("%e " % ii for ii in mkvc(tx[0:2,:]))
                M = M[:,0:2]
                N = N[:,0:2]

            if surveyType == 'GENERAL':

                fid.writelines("%e " % ii for ii in mkvc(tx[0:3,:]))

            fid.write('%i\n'% nD)
            np.savetxt(fid, np.c_[ M, N , DCsurvey.dobs[count:count+nD], DCsurvey.std[count:count+nD] ], fmt='%e',delimiter=' ',newline='\n')
            fid.write('\n')

        count += nD

    fid.close()


def getSrc_locs(srcList):
    """

        Get all locations from DC survey
    """

    srcMat = []

    for src in srcList:

        srcMat.append(np.hstack(src.loc))

    srcMat = np.vstack(srcMat)

    return srcMat


def convertObs_DC3D_to_2D(srcList, lineID, flag='local'):
    """
        Read DC survey and projects the coordinate system
        according to the flag = 'Xloc' | 'Yloc' | 'local' (default)
        In the 'local' system, station coordinates are referenced
        to distance from the first srcLoc[0].loc[0]

        The Z value is preserved, but Y coordinates zeroed.

        Input:
        :param survey3D

        Output:
        :figure survey2D

        Edited April 6th, 2016

        @author: dominiquef

    """
    from SimPEG import np

    def stn_id(v0, v1, r):
        """
        Compute station ID along line
        """

        dl = int(v0.dot(v1)) * r

        return dl

    def r_unit(p1, p2):
        """
        r_unit(x,y) : Function computes the unit vector
        between two points with coordinates p1(x1,y1) and p2(x2,y2)

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

    srcMat = getSrc_locs(srcList)

    # Find all unique line id
    uniqueID = np.unique(lineID)

    for jj in range(len(uniqueID)):

        indx = np.where(lineID == uniqueID[jj])[0]

        # Find origin of survey
        r = 1e+8  # Initialize to some large number

        Tx = srcMat[indx]
        if np.shape(Tx)[1] == 3:
            stype = 'pole-dipole'

        else:
            stype = 'dipole-dipole'

        x0 = Tx[0][0:2]  # Define station zero along line

        vecTx, r1 = r_unit(x0, Tx[-1][0:2])

        for ii in range(len(indx)):

            # Get all receivers
            Rx = srcList[indx[ii]].rxList[0].locs
            nrx = Rx[0].shape[0]

            if flag == 'local':
                # Find A electrode along line
                vec, r = r_unit(x0, Tx[ii][0:2])
                A = stn_id(vecTx, vec, r)

                if stype != 'pole-dipole':
                    # Find B electrode along line
                    vec, r = r_unit(x0, Tx[ii][3:6])
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

                if stype != 'pole-dipole':
                    B = Tx[ii][4]

                M = Rx[0][:, 1]
                N = Rx[1][:, 1]

            elif flag == 'Xloc':
                """ Copy the rx-tx locs"""
                A = Tx[ii][0]

                if stype != 'pole-dipole':
                    B = Tx[ii][3]

                M = Rx[0][:, 0]
                N = Rx[1][:, 0]

            rxClass = DC.Rx.Dipole(np.c_[M, np.zeros(nrx), Rx[0][:, 2]],
                                   np.c_[N, np.zeros(nrx), Rx[1][:, 2]])

            if stype == 'pole-dipole':
                srcList2D.append(DC.Src.Pole([rxClass],
                                 np.asarray([A, 0, Tx[ii][2]])))

            elif stype == 'dipole-dipole':
                srcList2D.append(DC.Src.Pole([rxClass],
                                 np.asarray((A, 0, Tx[ii][2],
                                 B, 0, Tx[ii][5]))))

    return srcList2D


def readUBC_DC2DModel(fileName):
    """
        Read UBC GIF 2DTensor model and generate 2D Tensor model in simpeg

        Input:
        :param fileName, path to the UBC GIF 2D model file

        Output:
        :param SimPEG TensorMesh 2D object
        :return

        Created on Thu Nov 12 13:14:10 2015

        @author: dominiquef

    """
    from SimPEG import np, mkvc

    # Open fileand skip header... assume that we know the mesh already
    obsfile = np.genfromtxt(fileName, delimiter=' \n',
                            dtype=np.str, comments='!')

    dim = np.array(obsfile[0].split(), dtype=float)

    temp = np.array(obsfile[1].split(), dtype=float)

    if len(temp) > 1:
        model = np.zeros(dim)

        for ii in range(len(obsfile)-1):
            mm = np.array(obsfile[ii+1].split(), dtype=float)
            model[:, ii] = mm

        model = model[:, ::-1]

    else:

        if len(obsfile[1:]) == 1:
            mm = np.array(obsfile[1:].split(), dtype=float)

        else:
            mm = np.array(obsfile[1:], dtype=float)

        # Permute the second dimension to flip the order
        model = mm.reshape(dim[1], dim[0])

        model = model[::-1, :]
        model = np.transpose(model, (1, 0))

    model = mkvc(model)

    return model

def readUBC_DC2Dpre(fileName):
    """
        Read UBC GIF DCIP 2D observation file and generate arrays for tx-rx location

        Input:
        :param fileName, path to the UBC GIF 3D obs file

        Output:
        DCsurvey
        :return

        Created on Mon March 9th, 2016 << Doug's 70th Birthday !! >>

        @author: dominiquef

    """

    # Load file
    obsfile = np.genfromtxt(fileName,delimiter=' \n',dtype=np.str,comments='!')

    # Pre-allocate
    srcLists = []
    Rx = []
    d = []
    zflag = True # Flag for z value provided

    for ii in range(obsfile.shape[0]):

        if not obsfile[ii]:
            continue

        # First line is transmitter with number of receivers


        temp = (np.fromstring(obsfile[ii], dtype=float,sep=' ').T)


        # Check if z value is provided, if False -> nan
        if len(temp)==5:
            tx = np.r_[temp[0],np.nan,np.nan,temp[1],np.nan,np.nan]
            zflag = False

        else:
            tx = np.r_[temp[0],np.nan,temp[1],temp[2],np.nan,temp[3]]


        if zflag:
            rx = np.c_[temp[4],np.nan,temp[5],temp[6],np.nan,temp[7]]


        else:
            rx = np.c_[temp[2],np.nan,np.nan,temp[3],np.nan,np.nan]
            # Check if there is data with the location

        d.append(temp[-1])


        Rx = DC.Rx.Dipole(rx[:,:3],rx[:,3:])
        srcLists.append( DC.Src.Dipole( [Rx], tx[:3],tx[3:]) )

    # Create survey class
    survey = DC.SurveyDC.Survey(srcLists)

    survey.dobs = np.asarray(d)

    return {'DCsurvey': survey}
