from __future__ import print_function
from SimPEG import np
from SimPEG.EM.Static import DC, IP

def plot_pseudoSection(DCsurvey, axs, stype='dpdp', dtype="appc", clim=None):
    """
        Read list of 2D tx-rx location and plot a speudo-section of apparent
        resistivity.

        Assumes flat topo for now...

        Input:
        :param d2D, z0
        :switch stype -> Either 'pdp' (pole-dipole) | 'dpdp' (dipole-dipole)
        :switch dtype=-> Either 'appr' (app. res) | 'appc' (app. con) | 'volt' (potential)
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
        if stype == 'pdp':
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

        elif stype == 'dpdp':
            MA = np.abs(Tx[0][0] - Rx[0][:,0])
            MB = np.abs(Tx[1][0] - Rx[0][:,0])
            NA = np.abs(Tx[0][0] - Rx[1][:,0])
            NB = np.abs(Tx[1][0] - Rx[1][:,0])

            # Create mid-point location
            Cmid = (Tx[0][0] + Tx[1][0])/2
            Pmid = (Rx[0][:,0] + Rx[1][:,0])/2
            if DCsurvey.mesh.dim == 2:
                zsrc = (Tx[0][1] + Tx[1][1])/2
            elif DCsurvey.mesh.dim ==3:
                zsrc = (Tx[0][2] + Tx[1][2])/2

        # Change output for dtype
        if dtype == 'volt':

            rho = np.hstack([rho,data])

        else:

            # Compute pant leg of apparent rho
            if stype == 'pdp':

                leg =  data * 2*np.pi  * MA * ( MA + MN ) / MN

            elif stype == 'dpdp':

                leg = data * 2*np.pi / ( 1/MA - 1/MB + 1/NB - 1/NA )
                LEG.append(1./(2*np.pi) *( 1/MA - 1/MB + 1/NB - 1/NA ))
            else:
                print("""dtype must be 'pdp'(pole-dipole) | 'dpdp' (dipole-dipole) """)
                break


            if dtype == 'appc':

                leg = np.log10(abs(1./leg))
                rho = np.hstack([rho,leg])

            elif dtype == 'appr':

                leg = np.log10(abs(leg))
                rho = np.hstack([rho,leg])

            else:
                print("""dtype must be 'appr' | 'appc' | 'volt' """)
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
    cbar = plt.colorbar(format="$10^{%.1f}$",fraction=0.04,orientation="horizontal")

    cmin,cmax = cbar.get_clim()
    ticks = np.linspace(cmin,cmax,3)
    cbar.set_ticks(ticks)
    cbar.ax.tick_params(labelsize=10)

    if dtype == 'appc':
        cbar.set_label("App.Cond",size=12)
    elif dtype == 'appr':
        cbar.set_label("App.Res.",size=12)
    elif dtype == 'volt':
        cbar.set_label("Potential (V)",size=12)

    # Plot apparent resistivity
    ax.scatter(midx,midz,s=10,c=rho.T, vmin =vmin, vmax = vmax, clim=(vmin, vmax))

    #ax.set_xticklabels([])
    #ax.set_yticklabels([])

    plt.gca().set_aspect('equal', adjustable='box')



    return ph, LEG

def gen_DCIPsurvey(endl, mesh, stype, a, b, n):
    """
        Load in endpoints and survey specifications to generate Tx, Rx location
        stations.

        Assumes flat topo for now...

        Input:
        :param endl -> input endpoints [x1, y1, z1, x2, y2, z2]
        :object mesh -> SimPEG mesh object
        :switch stype -> "dpdp" (dipole-dipole) | "pdp" (pole-dipole) | 'gradient'
        : param a, n -> pole seperation, number of rx dipoles per tx

        Output:
        :param Tx, Rx -> List objects for each tx location
            Lines: P1x, P1y, P1z, P2x, P2y, P2z

        Created on Wed December 9th, 2015

        @author: dominiquef
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

    nstn = np.floor( dl_len / a )

    # Compute discrete pole location along line
    stn_x = endl[0,0] + np.array(range(int(nstn)))*dl_x*a
    stn_y = endl[0,1] + np.array(range(int(nstn)))*dl_y*a

    if mesh.dim==2:
        ztop = mesh.vectorNy[-1]
        # Create line of P1 locations
        M = np.c_[stn_x, np.ones(nstn).T*ztop]
        # Create line of P2 locations
        N = np.c_[stn_x+a*dl_x, np.ones(nstn).T*ztop]

    elif mesh.dim==3:
        ztop = mesh.vectorNz[-1]
        # Create line of P1 locations
        M = np.c_[stn_x, stn_y, np.ones(nstn).T*ztop]
        # Create line of P2 locations
        N = np.c_[stn_x+a*dl_x, stn_y+a*dl_y, np.ones(nstn).T*ztop]


    ## Build list of Tx-Rx locations depending on survey type
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

        # Gradient survey only requires Tx at end of line and creates a square
        # grid of receivers at in the middle at a pre-set minimum distance

        # Get the edge limit of survey area
        min_x = endl[0,0] + dl_x * b
        min_y = endl[0,1] + dl_y * b

        max_x = endl[1,0] - dl_x * b
        max_y = endl[1,1] - dl_y * b

        box_l = np.sqrt( (min_x - max_x)**2 + (min_y - max_y)**2 )
        box_w = box_l/2.

        nstn = np.floor( box_l / a )

        # Compute discrete pole location along line
        stn_x = min_x + np.array(range(int(nstn)))*dl_x*a
        stn_y = min_y + np.array(range(int(nstn)))*dl_y*a

        # Define number of cross lines
        nlin = int(np.floor( box_w / a ))
        lind = list(range(-nlin,nlin+1))

        ngrad = nstn * len(lind)

        rx = np.zeros([ngrad,6])
        for ii in range( len(lind) ):

            # Move line in perpendicular direction by dipole spacing
            lxx = stn_x - lind[ii]*a*dl_y
            lyy = stn_y + lind[ii]*a*dl_x


            M = np.c_[ lxx, lyy , np.ones(nstn).T*ztop]
            N = np.c_[ lxx+a*dl_x, lyy+a*dl_y, np.ones(nstn).T*ztop]
            rx[(ii*nstn):((ii+1)*nstn),:] = np.c_[M,N]

            if mesh.dim==3:
                rxClass = DC.Rx.Dipole(rx[:,:3], rx[:,3:])
            elif mesh.dim==2:
                M = M[:,[0,2]]
                N = N[:,[0,2]]
                rxClass = DC.Rx.Dipole_ky(rx[:,[0,2]], rx[:,[3,5]])
            srcClass = DC.Src.Dipole([rxClass], M[0,:], N[-1,:])
        SrcList.append(srcClass)
    else:
        print("""stype must be either 'pole-dipole', 'dipole-dipole' or 'gradient'. """)


    return SrcList

