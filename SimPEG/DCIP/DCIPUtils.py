from SimPEG import np, Utils
import BaseDC as DC
import BaseDC as IP
import warnings

def getActiveindfromTopo(mesh, topo):
# def genActiveindfromTopo(mesh, topo):
    """
        Get active indices from topography
    """
    warnings.warn(
            "`getActiveindfromTopo` is deprecated and will be removed in future versions. Use `SimPEG.Utils.surface2ind_topo` instead",
            FutureWarning)
    from scipy.interpolate import NearestNDInterpolator
    if mesh.dim==3:
        nCxy = mesh.nCx*mesh.nCy
        Zcc = mesh.gridCC[:,2].reshape((nCxy, mesh.nCz), order='F')
        Ftopo = NearestNDInterpolator(topo[:,:2], topo[:,2])
        XY = Utils.ndgrid(mesh.vectorCCx, mesh.vectorCCy)
        XY.shape
        topo = Ftopo(XY)
        actind = []
        for ixy in range(nCxy):
            actind.append(topo[ixy] <= Zcc[ixy,:])
    else:
        raise NotImplementedError("Only 3D is working")

    return Utils.mkvc(np.vstack(actind))

def gettopoCC(mesh, airind):
# def gettopoCC(mesh, airind):
    """
        Get topography from active indices of mesh.
    """
    warnings.warn(
            "`gettopoCC` is deprecated and will be removed in future versions. Use `SimPEG.Utils.surface2ind_topo` instead",
            FutureWarning)
    mesh2D = Mesh.TensorMesh([mesh.hx, mesh.hy], mesh.x0[:2])
    zc = mesh.gridCC[:,2]
    AIRIND = airind.reshape((mesh.vnC[0]*mesh.vnC[1],mesh.vnC[2]), order='F')
    ZC = zc.reshape((mesh.vnC[0]*mesh.vnC[1], mesh.vnC[2]), order='F')
    topo = np.zeros(ZC.shape[0])
    topoCC = np.zeros(ZC.shape[0])
    for i in range(ZC.shape[0]):
        ind  = np.argmax(ZC[i,:][~AIRIND[i,:]])
        topo[i] = ZC[i,:][~AIRIND[i,:]].max() + mesh.hz[~AIRIND[i,:]][ind]*0.5
        topoCC[i] = ZC[i,:][~AIRIND[i,:]].max()
    XY = Utils.ndgrid(mesh.vectorCCx, mesh.vectorCCy)
    return mesh2D, topoCC

def readUBC_DC3Dobstopo(filename,mesh,topo,probType="CC"):
    """
     Seogi's personal readObs function.

    """
    text_file = open(filename, "r")
    lines = text_file.readlines()
    text_file.close()
    SRC = []
    DATA = []
    srcLists = []
    isrc = 0
    # airind = getActiveindfromTopo(mesh, topo)
    # mesh2D, topoCC = gettopoCC(mesh, airind)

    for line in lines:
        if "!" in line.split(): continue
        elif line == '\n': continue
        elif line == ' \n': continue
        temp =  map(float, line.split())
        # Read a line for the current electrode
        if len(temp) == 5: # SRC: Only X and Y are provided (assume no topography)
            #TODO consider topography and assign the closest cell center in the earth
            if isrc == 0:
                DATA_temp = []
            else:
                DATA.append(np.asarray(DATA_temp))
                DATA_temp = []
                indM = Utils.closestPoints(mesh2D, DATA[isrc-1][:,1:3])
                indN = Utils.closestPoints(mesh2D, DATA[isrc-1][:,3:5])
                rx = DCIP.RxDipole(np.c_[DATA[isrc-1][:,1:3], topoCC[indM]], np.c_[DATA[isrc-1][:,3:5], topoCC[indN]])
                temp = np.asarray(temp)
                if [SRC[isrc-1][0], SRC[isrc-1][1]] == [SRC[isrc-1][2], SRC[isrc-1][3]]:
                    indA = Utils.closestPoints(mesh2D, [SRC[isrc-1][0], SRC[isrc-1][1]])
                    tx = DCIP.SrcDipole([rx], [SRC[isrc-1][0], SRC[isrc-1][1], topoCC[indA]],[mesh.vectorCCx.max(), mesh.vectorCCy.max(), topoCC[-1]])
                else:
                    indA = Utils.closestPoints(mesh2D, [SRC[isrc-1][0], SRC[isrc-1][1]])
                    indB = Utils.closestPoints(mesh2D, [SRC[isrc-1][2], SRC[isrc-1][3]])
                    tx = DCIP.SrcDipole([rx], [SRC[isrc-1][0], SRC[isrc-1][1], topoCC[indA]],[SRC[isrc-1][2], SRC[isrc-1][3], topoCC[indB]])
                srcLists.append(tx)
            SRC.append(temp)
            isrc += 1
        elif len(temp) == 7: # SRC: X, Y and Z are provided
            SRC.append(temp)
            isrc += 1
        elif len(temp) == 6: #
            DATA_temp.append(np.r_[isrc, np.asarray(temp)])
        elif len(temp) > 7:
            DATA_temp.append(np.r_[isrc, np.asarray(temp)])

    DATA.append(np.asarray(DATA_temp))
    DATA_temp = []
    indM = Utils.closestPoints(mesh2D, DATA[isrc-1][:,1:3])
    indN = Utils.closestPoints(mesh2D, DATA[isrc-1][:,3:5])
    rx = DCIP.RxDipole(np.c_[DATA[isrc-1][:,1:3], topoCC[indM]], np.c_[DATA[isrc-1][:,3:5], topoCC[indN]])
    temp = np.asarray(temp)
    if [SRC[isrc-1][0], SRC[isrc-1][1]] == [SRC[isrc-1][2], SRC[isrc-1][3]]:
        indA = Utils.closestPoints(mesh2D, [SRC[isrc-1][0], SRC[isrc-1][1]])
        tx = DCIP.SrcDipole([rx], [SRC[isrc-1][0], SRC[isrc-1][1], topoCC[indA]],[mesh.vectorCCx.max(), mesh.vectorCCy.max(), topoCC[-1]])
    else:
        indA = Utils.closestPoints(mesh2D, [SRC[isrc-1][0], SRC[isrc-1][1]])
        indB = Utils.closestPoints(mesh2D, [SRC[isrc-1][2], SRC[isrc-1][3]])
        tx = DCIP.SrcDipole([rx], [SRC[isrc-1][0], SRC[isrc-1][1], topoCC[indA]],[SRC[isrc-1][2], SRC[isrc-1][3], topoCC[indB]])
    srcLists.append(tx)
    text_file.close()
    survey = DCIP.SurveyDC(srcLists)

    # Do we need this?
    SRC = np.asarray(SRC)
    DATA = np.vstack(DATA)
    survey.dobs = np.vstack(DATA)[:,-2]


    return {'DCsurvey':survey, 'airind':airind, 'topoCC':topoCC, 'SRC':SRC}

def readUBC_DC2DModel(fileName):
    """
    Read UBC GIF 2DTensor model and generate 2D Tensor model in simpeg

    :param string fileName: path to the UBC GIF 2D model file
    :rtype: TensorMesh
    :return: SimPEG TensorMesh 2D object

    """
    from SimPEG import np, mkvc

    # Open fileand skip header... assume that we know the mesh already
    obsfile = np.genfromtxt(fileName, delimiter=' \n', dtype=np.str, comments='!')

    dim = np.array(obsfile[0].split(), dtype=float)

    temp = np.array(obsfile[1].split(), dtype=float)

    if len(temp) > 1:
        model = np.zeros(dim)

        for ii in range(len(obsfile)-1):
            mm = np.array(obsfile[ii+1].split(), dtype=float)
            model[:,ii] = mm

        model = model[:,::-1]

    else:

        if len(obsfile[1:])==1:
            mm = np.array(obsfile[1:].split(), dtype=float)

        else:
            mm = np.array(obsfile[1:], dtype=float)

        # Permute the second dimension to flip the order
        model = mm.reshape(dim[1],dim[0])

        model = model[::-1,:]
        model = np.transpose(model, (1, 0))

    model = mkvc(model)


    return model

def plot_pseudoSection(DCsurvey, axs, surveyType='dipole-dipole', unitType='volt', clim=None):
    """
    Read list of 2D tx-rx location and plot a speudo-section of apparent
    resistivity.

    Assumes flat topo for now...

    :param SurveyDC DCsurvey:
    :param string surveyType: Either 'pole-dipole'  | 'dipole-dipole'
    :param string unitType: Either 'appResistivity' | 'appConductivity'  | 'volt'
    :rtype: matplotlib.plt
    :return: figure scatter plot overlayed on image

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
    count = 0 # Counter for data
    for ii in range(DCsurvey.nSrc):

        Tx = DCsurvey.srcList[ii].loc
        Rx = DCsurvey.srcList[ii].rxList[0].locs

        nD = DCsurvey.srcList[ii].rxList[0].nD

        data = DCsurvey.dobs[count:count+nD]
        count += nD

        # Get distances between each poles A-B-M-N
        MA = np.abs(Tx[0][0] - Rx[0][:,0])
        MB = np.abs(Tx[1][0] - Rx[0][:,0])
        NB = np.abs(Tx[1][0] - Rx[1][:,0])
        NA = np.abs(Tx[0][0] - Rx[1][:,0])
        MN = np.abs(Rx[1][:,0] - Rx[0][:,0])

        # Create mid-point location
        Cmid = (Tx[0][0] + Tx[1][0])/2
        Pmid = (Rx[0][:,0] + Rx[1][:,0])/2

        # Change output for unitType
        if unitType == 'volt':

            rho = np.hstack([rho,data])

        else:

            # Compute pant leg of apparent rho
            if surveyType == 'pole-dipole':

                leg =  data * 2*np.pi  * MA * ( MA + MN ) / MN

            elif surveyType == 'dipole-dipole':

                leg = data * 2*np.pi / ( 1/MA - 1/MB - 1/NB + 1/NA )

            else:
                print """unitType must be 'pole-dipole' | 'dipole-dipole' """
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
        midz = np.hstack([midz, -np.abs(Cmid-Pmid)/2 + (Tx[0][2] + Tx[1][2])/2 ])

    ax = axs

    # Grid points
    grid_x, grid_z = np.mgrid[np.min(midx):np.max(midx), np.min(midz):np.max(midz)]
    grid_rho = griddata(np.c_[midx,midz], rho.T, (grid_x, grid_z), method='linear')

    if clim == None:
        vmin, vmax = rho.min(), rho.max()
    else:
        vmin, vmax = clim[0], clim[1]

    grid_rho = np.ma.masked_where(np.isnan(grid_rho), grid_rho)
    ph = plt.pcolormesh(grid_x[:,0],grid_z[0,:],grid_rho.T, clim=(vmin, vmax))
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



    return ph

def gen_DCIPsurvey(endl, mesh, surveyType, AM_sep, MN_sep, nrx):
    """
    Load in endpoints and survey specifications to generate Tx, Rx location
    stations.

    Assumes flat topo for now...

    :param numpy.array endl: input endpoints [[x1, y1] , [x2, y2]]
    :param Mesh mesh: SimPEG mesh object
    :param string surveyType: 'dipole-dipole' | 'pole-dipole' | 'gradient'
    :param float AM_sep: transmitter (A) - receiver (M) seperation
    :param float b: receiver dipole seperation
    :param float nrx: pole seperation, number of rx dipoles per tx

    :rtype: DC.Survey, Src, Rx
    :returns: DC survey, Source

    !! Require clean up to deal with DCsurvey
    """

    from SimPEG import np

    def xy_2_r(x1,x2,y1,y2):
        r = np.sqrt( np.sum((x2 - x1)**2 + (y2 - y1)**2) )
        return r

    ## Evenly distribute electrodes and put on surface
    # Mesure survey length and direction
    dl_len = xy_2_r(endl[0,0],endl[1,0],endl[0,1],endl[1,1])

    dl_x = ( endl[1,0] - endl[0,0] ) / dl_len
    dl_y = ( endl[1,1] - endl[0,1] ) / dl_len

    nstn = np.floor( dl_len / AM_sep )

    # Compute discrete pole location along line
    stn_x = endl[0,0] + np.array(range(int(nstn)))*dl_x*AM_sep
    stn_y = endl[0,1] + np.array(range(int(nstn)))*dl_y*AM_sep

    # Create line of P1 locations
    M = np.c_[stn_x, stn_y, np.ones(nstn).T*mesh.vectorNz[-1]]

    # Create line of P2 locations
    N = np.c_[stn_x+AM_sep*dl_x, stn_y+AM_sep*dl_y, np.ones(nstn).T*mesh.vectorNz[-1]]

    ## Build list of Tx-Rx locations depending on survey type
    # Dipole-dipole: Moving tx with [a] spacing -> [AB a MN1 a MN2 ... a MNn]
    # Pole-dipole: Moving pole on one end -> [A a MN1 a MN2 ... MNn a B]
    Tx = []
    Rx = []
    SrcList = []


    if surveyType != 'gradient':

        for ii in range(0, int(nstn)-1):


            if surveyType == 'dipole-dipole':
                tx = np.c_[M[ii,:],N[ii,:]]
            elif surveyType == 'pole-dipole':
                tx = np.c_[M[ii,:],M[ii,:]]

            # Rx.append(np.c_[M[ii+1:indx,:],N[ii+1:indx,:]])

            # Current elctrode seperation
            AB = xy_2_r(tx[0,1],endl[1,0],tx[1,1],endl[1,1])

            # Number of receivers to fit
            nstn = np.min([np.floor( (AB - MN_sep) / AM_sep ) , nrx])

            # Check if there is enough space, else break the loop
            if nstn <= 0:
                continue

            # Compute discrete pole location along line
            stn_x = N[ii,0] + dl_x*MN_sep + np.array(range(int(nstn)))*dl_x*AM_sep
            stn_y = N[ii,1] + dl_y*MN_sep + np.array(range(int(nstn)))*dl_y*AM_sep

            # Create receiver poles
            # Create line of P1 locations
            P1 = np.c_[stn_x, stn_y, np.ones(nstn).T*mesh.vectorNz[-1]]

            # Create line of P2 locations
            P2 = np.c_[stn_x+AM_sep*dl_x, stn_y+AM_sep*dl_y, np.ones(nstn).T*mesh.vectorNz[-1]]

            Rx.append(np.c_[P1,P2])
            rxClass = DC.RxDipole(P1, P2)
            Tx.append(tx)
            if surveyType == 'dipole-dipole':
                srcClass = DC.SrcDipole([rxClass], M[ii,:],N[ii,:])
            elif surveyType == 'pole-dipole':
                srcClass = DC.SrcDipole([rxClass], M[ii,:],M[ii,:])
            SrcList.append(srcClass)

    elif surveyType == 'gradient':

        # Gradient survey only requires Tx at end of line and creates a square
        # grid of receivers at in the middle at a pre-set minimum distance

        Tx.append(np.c_[M[0,:],N[-1,:]])

        # Get the edge limit of survey area
        min_x = endl[0,0] + dl_x * MN_sep
        min_y = endl[0,1] + dl_y * MN_sep

        max_x = endl[1,0] - dl_x * MN_sep
        max_y = endl[1,1] - dl_y * MN_sep

        box_l = np.sqrt( (min_x - max_x)**2 + (min_y - max_y)**2 )
        box_w = box_l/2.

        nstn = np.floor( box_l / AM_sep )

        # Compute discrete pole location along line
        stn_x = min_x + np.array(range(int(nstn)))*dl_x*AM_sep
        stn_y = min_y + np.array(range(int(nstn)))*dl_y*AM_sep

        # Define number of cross lines
        nlin = int(np.floor( box_w / AM_sep ))
        lind = range(-nlin,nlin+1)

        ngrad = nstn * len(lind)

        rx = np.zeros([ngrad,6])
        for ii in range( len(lind) ):

            # Move line in perpendicular direction by dipole spacing
            lxx = stn_x - lind[ii]*AM_sep*dl_y
            lyy = stn_y + lind[ii]*AM_sep*dl_x


            M = np.c_[ lxx, lyy , np.ones(nstn).T*mesh.vectorNz[-1]]
            N = np.c_[ lxx+AM_sep*dl_x, lyy+AM_sep*dl_y, np.ones(nstn).T*mesh.vectorNz[-1]]

            rx[(ii*nstn):((ii+1)*nstn),:] = np.c_[M,N]

        Rx.append(rx)
        rxClass = DC.RxDipole(rx[:,:3], rx[:,3:])
        srcClass = DC.SrcDipole([rxClass], M[0,:], N[-1,:])
        SrcList.append(srcClass)
    else:
        print """surveyType must be either 'pole-dipole', 'dipole-dipole' or 'gradient'. """

    survey = DC.SurveyDC(SrcList)
    return survey, Tx, Rx

def writeUBC_DCobs(fileName, DCsurvey, dim, surveyType):
    """
        Write UBC GIF DCIP 2D or 3D observation file

        Input:
        :string fileName -> including path where the file is written out
        :DCsurvey -> DC survey class object
        :string dim ->  either '2D' | '3D'
        :string  surveyType ->  either 'SURFACE' | 'GENERAL'

        Output:
        :param UBC2D-Data file
        :return

        Last edit: February 16th, 2016

        @author: dominiquef

    """
    from SimPEG import mkvc

    assert (dim=='2D') | (dim=='3D'), "Data must be either '2D' | '3D'"
    assert (surveyType=='SURFACE') | (surveyType=='GENERAL') | (surveyType=='SIMPLE'), "Data must be either 'SURFACE' | 'GENERAL' | 'SIMPLE'"

    fid = open(fileName,'w')
    fid.write('! ' + surveyType + ' FORMAT\n')

    count = 0

    for ii in range(DCsurvey.nSrc):

        tx = np.c_[DCsurvey.srcList[ii].loc]

        rx = DCsurvey.srcList[ii].rxList[0].locs

        nD = DCsurvey.srcList[ii].nD

        M = rx[0]
        N = rx[1]

        # Adapt source-receiver location for dim and surveyType
        if dim=='2D':

            if surveyType == 'SIMPLE':

                #fid.writelines("%e " % ii for ii in mkvc(tx[0,:]))
                A = np.repeat(tx[0,0],M.shape[0],axis=0)
                B = np.repeat(tx[0,1],M.shape[0],axis=0)
                M = M[:,0]
                N = N[:,0]

                np.savetxt(fid, np.c_[A, B, M, N , DCsurvey.dobs[count:count+nD], DCsurvey.std[count:count+nD] ], fmt='%e',delimiter=' ',newline='\n')


            else:

                if surveyType == 'SURFACE':

                    fid.writelines("%e " % ii for ii in mkvc(tx[0,:]))
                    M = M[:,0]
                    N = N[:,0]

                if surveyType == 'GENERAL':

                    fid.writelines("%e " % ii for ii in mkvc(tx[::2,:]))
                    M = M[:,0::2]
                    N = N[:,0::2]

                fid.write('%i\n'% nD)
                np.savetxt(fid, np.c_[ M, N , DCsurvey.dobs[count:count+nD], DCsurvey.std[count:count+nD] ], fmt='%e',delimiter=' ',newline='\n')

        if dim=='3D':

            if surveyType == 'SURFACE':

                fid.writelines("%e " % ii for ii in mkvc(tx[0:2,:]))
                M = M[:,0:2]
                N = N[:,0:2]

            if surveyType == 'GENERAL':

                fid.writelines("%e " % ii for ii in mkvc(tx))

            fid.write('%i\n'% nD)
            np.savetxt(fid, np.c_[ M, N , DCsurvey.dobs[count:count+nD], DCsurvey.std[count:count+nD] ], fmt='%e',delimiter=' ',newline='\n')

        count += nD

    fid.close()

def convertObs_DC3D_to_2D(DCsurvey, lineID, flag='local'):
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

    def stn_id(v0,v1,r):
        """
        Compute station ID along line
        """

        dl = int(v0.dot(v1)) * r

        return dl

    srcLists = []

    srcMat = getSrc_locs(DCsurvey)

    # Find all unique line id
    uniqueID = np.unique(lineID)

    for jj in range(len(uniqueID)):

        indx = np.where(lineID==uniqueID[jj])[0]

        # Find origin of survey
        r = 1e+8 # Initialize to some large number

        Tx = srcMat[indx]

        x0 = Tx[0][0,0:2] # Define station zero along line

        vecTx, r1 = r_unit(x0,Tx[-1][1,0:2])

        for ii in range(len(indx)):

            # Get all receivers
            Rx = DCsurvey.srcList[indx[ii]].rxList[0].locs
            nrx = Rx[0].shape[0]

            if flag == 'local':
                # Find A electrode along line
                vec, r = r_unit(x0,Tx[ii][0,0:2])
                A = stn_id(vecTx,vec,r)

                # Find B electrode along line
                vec, r = r_unit(x0,Tx[ii][1,0:2])
                B = stn_id(vecTx,vec,r)

                M = np.zeros(nrx)
                N = np.zeros(nrx)
                for kk in range(nrx):

                    # Find all M electrodes along line
                    vec, r = r_unit(x0,Rx[0][kk,0:2])
                    M[kk] = stn_id(vecTx,vec,r)

                    # Find all N electrodes along line
                    vec, r = r_unit(x0,Rx[1][kk,0:2])
                    N[kk] = stn_id(vecTx,vec,r)
            elif flag == 'Yloc':
                """ Flip the XY axis locs"""
                A = Tx[ii][0,1]
                B = Tx[ii][1,1]
                M = Rx[0][:,1]
                N = Rx[1][:,1]

            elif flag == 'Xloc':
                """ Copy the rx-tx locs"""
                A = Tx[ii][0,0]
                B = Tx[ii][1,0]
                M = Rx[0][:,0]
                N = Rx[1][:,0]

            Rx = DC.RxDipole(np.c_[M,np.zeros(nrx),Rx[0][:,2]],np.c_[N,np.zeros(nrx),Rx[1][:,2]])

            srcLists.append( DC.SrcDipole( [Rx], np.asarray([A,0,Tx[ii][0,2]]),np.asarray([B,0,Tx[ii][1,2]]) ) )


    DCsurvey2D = DC.SurveyDC(srcLists)

    DCsurvey2D.dobs = np.asarray(DCsurvey.dobs)
    DCsurvey2D.std = np.asarray(DCsurvey.std)

    return DCsurvey2D

def readUBC_DC3Dobs(fileName):
    """
        Read UBC GIF DCIP 3D observation file and generate survey

        :param string fileName:, path to the UBC GIF 3D obs file
        :rtype: Survey
        :return: DCIPsurvey

    """

    # Load file
    obsfile = np.genfromtxt(fileName,delimiter=' \n',dtype=np.str,comments='!')

    # Pre-allocate
    srcLists = []
    Rx = []
    d = []
    wd = []
    zflag = True # Flag for z value provided

    # Countdown for number of obs/tx
    count = 0
    for ii in range(obsfile.shape[0]):

        if not obsfile[ii]:
            continue

        # First line is transmitter with number of receivers
        if count==0:

            temp = (np.fromstring(obsfile[ii], dtype=float,sep=' ').T)
            count = int(temp[-1])

            # Check if z value is provided, if False -> nan
            if len(temp)==5:
                tx = np.r_[temp[0:2],np.nan,temp[0:2],np.nan]
                zflag = False

            else:
                tx = temp[:-1]

            rx = []
            continue

        temp = np.fromstring(obsfile[ii], dtype=float,sep=' ')

        if zflag:

            rx.append(temp[:-2])
            # Check if there is data with the location
            if len(temp)==8:
                d.append(temp[-2])
                wd.append(temp[-1])

        else:
            rx.append(np.r_[temp[0:2],np.nan,temp[0:2],np.nan] )
            # Check if there is data with the location
            if len(temp)==6:
                d.append(temp[-2])
                wd.append(temp[-1])

        count = count -1

        # Reach the end of transmitter block
        if count == 0:
            rx = np.asarray(rx)
            Rx = DC.RxDipole(rx[:,:3],rx[:,3:])
            srcLists.append( DC.SrcDipole( [Rx], tx[:3],tx[3:]) )

    # Create survey class
    survey = DC.SurveyDC(srcLists)

    survey.dobs = np.asarray(d)
    survey.std = np.asarray(wd)

    return {'DCsurvey':survey}

def readUBC_DC2Dobs(fileName):
    """
        ------- NEEDS TO BE UPDATED ------
        Read UBC GIF 2D observation file and generate arrays for tx-rx location

        Input:
        :param fileName, path to the UBC GIF 2D model file

        Output:
        :param rx, tx
        :return

        Created on Thu Nov 12 13:14:10 2015

        @author: dominiquef

    """

    from SimPEG import np

    # Load file
    obsfile = np.genfromtxt(fileName,delimiter=' \n',dtype=np.str,comments='!')

    # Check first line and figure out if 2D or 3D file format
    line = np.array(obsfile[0].split(),dtype=float)

    tx_A  = []
    tx_B  = []
    rx_M  = []
    rx_N  = []
    d   = []
    wd  = []

    for ii in range(obsfile.shape[0]):

        # If len==3, then simple format where tx-rx is listed on each line
        if len(line) == 4:

            temp = np.fromstring(obsfile[ii], dtype=float,sep=' ')
            tx_A = np.hstack((tx_A,temp[0]))
            tx_B = np.hstack((tx_B,temp[1]))
            rx_M = np.hstack((rx_M,temp[2]))
            rx_N = np.hstack((rx_N,temp[3]))


    rx = np.transpose(np.array((rx_M,rx_N)))
    tx = np.transpose(np.array((tx_A,tx_B)))

    return tx, rx, d, wd

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


        Rx = DC.RxDipole(rx[:,:3],rx[:,3:])
        srcLists.append( DC.SrcDipole( [Rx], tx[:3],tx[3:]) )

    # Create survey class
    survey = DC.SurveyDC(srcLists)

    survey.dobs = np.asarray(d)

    return {'DCsurvey':survey}

def readUBC_DC2DMesh(fileName):
    """
        Read UBC GIF 2DTensor mesh and generate 2D Tensor mesh in simpeg

        Input:
        :param fileName, path to the UBC GIF mesh file

        Output:
        :param SimPEG TensorMesh 2D object
        :return

        Created on Thu Nov 12 13:14:10 2015

        @author: dominiquef

    """

    from SimPEG import np
    # Open file
    fopen = open(fileName,'r')

    # Read down the file and unpack dx vector
    def unpackdx(fid,nrows):
        for ii in range(nrows):

            line = fid.readline()
            var = np.array(line.split(),dtype=float)

            if ii==0:
                x0= var[0]
                xvec = np.ones(int(var[2])) * (var[1] - var[0]) / int(var[2])
                xend = var[1]

            else:
                xvec = np.hstack((xvec,np.ones(int(var[1])) * (var[0] - xend) / int(var[1])))
                xend = var[0]

        return x0, xvec

    #%% Start with dx block
    # First line specifies the number of rows for x-cells
    line = fopen.readline()
    nl = np.array(line.split(),dtype=float)

    [x0, dx] = unpackdx(fopen,nl)


    #%% Move down the file until reaching the z-block
    line = fopen.readline()
    if not line:
        line = fopen.readline()

    #%% End with dz block
    # First line specifies the number of rows for z-cells
    line = fopen.readline()
    nl = np.array(line.split(),dtype=float)

    [z0, dz] = unpackdx(fopen,nl)

    # Flip z0 to be the bottom of the mesh for SimPEG
    z0 = z0 - sum(dz)
    dz = dz[::-1]
    #%% Make the mesh using SimPEG

    from SimPEG import Mesh
    tensMsh = Mesh.TensorMesh([dx,dz],(x0, z0))
    return tensMsh

def xy_2_lineID(DCsurvey):
    """
        Read DC survey class and append line ID.
        Assumes that the locations are listed in the order
        they were collected. May need to generalize for random
        point locations, but will be more expensive

        Input:
        :param DCdict Vectors of station location

        Output:
        :param LineID Vector of integers
        :return

        Created on Thu Feb 11, 2015

        @author: dominiquef

    """

    # Compute unit vector between two points
    nstn = DCsurvey.nSrc

    # Pre-allocate space
    lineID = np.zeros(nstn)

    linenum = 0
    indx    = 0

    for ii in range(nstn):

        if ii == 0:

            A = DCsurvey.srcList[ii].loc[0]
            B = DCsurvey.srcList[ii].loc[1]

            xout = np.mean([A[0:2],B[0:2]], axis = 0)

            xy0 = A[:2]
            xym = xout

            # Deal with replicate pole location
            if np.all(xy0==xym):

                xym[0] = xym[0] + 1e-3

            continue

        A = DCsurvey.srcList[ii].loc[0]
        B = DCsurvey.srcList[ii].loc[1]

        xin = np.mean([A[0:2],B[0:2]], axis = 0)

        # Compute vector between neighbours
        vec1, r1 = r_unit(xout,xin)

        # Compute vector between current stn and mid-point
        vec2, r2 = r_unit(xym,xin)

        # Compute vector between current stn and start line
        vec3, r3 = r_unit(xy0,xin)

        # Compute vector between mid-point and start line
        vec4, r4 = r_unit(xym,xy0)

        # Compute dot product
        ang1 = np.abs(vec1.dot(vec2))
        ang2 = np.abs(vec3.dot(vec4))

        # If the angles are smaller then 45d, than next point is on a new line
        if ((ang1 < np.cos(np.pi/4.)) | (ang2 < np.cos(np.pi/4.))) & (np.all(np.r_[r1,r2,r3,r4] > 0)):

            # Re-initiate start and mid-point location
            xy0 = A[:2]
            xym = xin

            # Deal with replicate pole location
            if np.all(xy0==xym):

                xym[0] = xym[0] + 1e-3

            linenum += 1
            indx = ii

        else:
            xym = np.mean([xy0,xin], axis = 0)

        lineID[ii] = linenum
        xout = xin

    return lineID

def r_unit(p1,p2):
    """
    r_unit(x,y) : Function computes the unit vector
    between two points with coordinates p1(x1,y1) and p2(x2,y2)

    """

    assert len(p1)==len(p2), 'locs must be the same shape.'

    dx = []
    for ii in range(len(p1)):
        dx.append((p2[ii] - p1[ii]))

    # Compute length of vector
    r =  np.linalg.norm(np.asarray(dx))


    if r!=0:
        vec = dx/r

    else:
        vec = np.zeros(len(p1))

    return vec, r

def getSrc_locs(DCsurvey):
    """


    """

    srcMat = np.zeros((DCsurvey.nSrc,2,3))
    for ii in range(DCsurvey.nSrc):
        srcMat[ii,:,:] =  np.asarray(DCsurvey.srcList[ii].loc)

    return srcMat
