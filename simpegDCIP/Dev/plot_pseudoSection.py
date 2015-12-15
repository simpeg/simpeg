def plot_pseudoSection(Tx,Rx,data,z0, stype):
    
    from SimPEG import np, mkvc
    from scipy.interpolate import griddata
    from matplotlib.colors import LogNorm
    import pylab as plt
    import re
    """
        Read list of 2D tx-rx location and plot a speudo-section of apparent
        resistivity.
        
        Assumes flat topo for now...
    
        Input:
        :param d2D, z0
        :switch stype -> Either 'pdp' (pole-dipole) | 'dpdp' (dipole-dipole)
    
        Output:
        :figure scatter plot overlayed on image
        
        Created on Mon December 7th, 2015
    
        @author: dominiquef
    
    """
    #d2D = np.asarray(d2D)
    
    midl = []
    midz = []
    rho = []
    
    for ii in range(len(Tx)):
        # Get distances between each poles
        rC1P1 = np.abs(Tx[ii][0] - Rx[ii][:,0]) 
        rC2P1 = np.abs(Tx[ii][1] - Rx[ii][:,0])
        rC1P2 = np.abs(Tx[ii][1] - Rx[ii][:,1])
        rC2P2 = np.abs(Tx[ii][0] - Rx[ii][:,1])
        rP1P2 = np.abs(Rx[ii][:,1] - Rx[ii][:,0])    
    
        # Compute apparent resistivity
        if re.match(stype,'pdp'):
            rho = np.hstack([rho, data[ii] * 2*np.pi  * rC1P1 * ( rC1P1 + rP1P2 ) / rP1P2] )
            
        elif re.match(stype,'dpdp'):
            rho = np.hstack([rho, data[ii] * 2*np.pi / ( 1/rC1P1 - 1/rC2P1 - 1/rC1P2 + 1/rC2P2 ) ])
    
        Cmid = (Tx[ii][0] + Tx[ii][1])/2
        Pmid = (Rx[ii][:,0] + Rx[ii][:,1])/2
    
        midl = np.hstack([midl, ( Cmid + Pmid )/2 ])
        midz = np.hstack([midz, -np.abs(Cmid-Pmid)/2 + z0 ])
    
   
    # Grid points
    grid_x, grid_z = np.mgrid[np.min(midl):np.max(midl), np.min(midz):np.max(midz)]
    grid_rho = griddata(np.c_[midl,midz], np.log10(abs(1/rho.T)), (grid_x, grid_z), method='linear')
    
    
    #plt.subplot(2,1,2)
    plt.imshow(grid_rho.T, extent = (np.min(midl),np.max(midl),np.min(midz),np.max(midz)), origin='lower', alpha=0.8)
    cbar = plt.colorbar(format = '%.2f',fraction=0.02)
    cmin,cmax = cbar.get_clim()
    ticks = np.linspace(cmin,cmax,3)
    cbar.set_ticks(ticks)
    
    # Plot apparent resistivity
    plt.scatter(midl,midz,s=50,c=np.log10(abs(1/rho.T)))
                
