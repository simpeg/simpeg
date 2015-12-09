def plot_pseudoSection(d2D,z0):
    
    from SimPEG import np
    from scipy.interpolate import griddata
    import pylab as plt
    
    """
        Read list of 2D tx-rx location and plot a speudo-section
        Only implemented for flat topo
    
        Input:
        :param d2D, z0
    
        Output:
        :figure scatter plot overlayed on image
        
        Created on Mon December 7th, 2015
    
        @author: dominiquef
    
    """
    d2D = np.asarray(d2D)
  
    # Get distances between each poles
    rC1P1 = d2D[:,0] - d2D[:,2] 
    rC2P1 = d2D[:,0] - d2D[:,3]
    rC1P2 = d2D[:,1] - d2D[:,2]
    rC2P2 = d2D[:,1] - d2D[:,3]
    
    # Compute apparent resistivity
    rho = d2D[:,4] * 2*np.pi / ( 1/rC1P1 - 1/rC2P1 - 1/rC1P2 + 1/rC2P2 )
    
    Cmid = (d2D[:,0] + d2D[:,1])/2
    Pmid = (d2D[:,2] + d2D[:,3])/2
    
    midl = ( Cmid + Pmid )/2
    midz = -np.abs(Cmid-Pmid) + z0
    
    # Grid points
    grid_x, grid_z = np.mgrid[np.min(midl):np.max(midl), np.min(midz):np.max(midz)]
    grid_rho = griddata(np.c_[midl,midz], np.log10(abs(1/rho.T)), (grid_x, grid_z), method='linear')
    
    
    #plt.subplot(2,1,2)
    plt.imshow(grid_rho.T, extent = (np.min(midl),np.max(midl),np.min(midz),np.max(midz)), origin='lower', alpha=0.8)
    #plt.colorbar()
    
    # Plot apparent resistivity
    plt.scatter(midl,midz,s=50,c=np.log10(abs(1/rho.T)))
                
