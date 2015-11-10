def fwr_MAG_obs(xn,yn,zn,B,M,rxLoc,model):
    """ 
    Forward model magnetic data using integral equation
    
    INPUT:
    xn, yn, zn  = Mesh nodes location
    B           = Inducing field parameter [Binc, Bdecl, B0]
    M           = Magnetization matrix [Minc, Mdecl]
    rxLox       = Observation location informat [obsx, obsy, obsz]
    model       = Model associated with mesh
    
    OUTPUT:
    dobs        =Observation array in format [obsx, obsy, obsz, data]

    Created on Oct 7, 2015

    @author: dominiquef
     """ 
    
    #%%
    from SimPEG import np, Utils, sp, mkvc
    from get_T_mat import get_T_mat
    
    mcell = (len(xn)-1) * (len(yn)-1) * (len(zn)-1)
      
    ndata = rxLoc.shape[0]    
    
    # Convert declination from north to cartesian
    Md = (450.-float(M[1]))%360.
    
    # Create magnetization matrix
    mx = np.cos(np.deg2rad(M[0])) * np.cos(np.deg2rad(Md))
    my = np.cos(np.deg2rad(M[0])) * np.sin(np.deg2rad(Md))
    mz = np.sin(np.deg2rad(M[0]))
    
    Mx = Utils.sdiag(np.ones([mcell])*mx*B[2])
    My = Utils.sdiag(np.ones([mcell])*my*B[2])
    Mz = Utils.sdiag(np.ones([mcell])*mz*B[2])
    
    #matplotlib.pyplot.spy(scipy.sparse.csr_matrix(Mx))
    #plt.show()
    Mxyz = sp.vstack((Mx,My,Mz));
    
    #%% Create TMI projector
    
    # Convert Bdecination from north to cartesian 
    D = (450.-float(B[1]))%360.
    
    Ptmi = mkvc(np.r_[np.cos(np.deg2rad(B[0]))*np.cos(np.deg2rad(D)),np.cos(np.deg2rad(B[0]))*np.sin(np.deg2rad(D)),np.sin(np.deg2rad(B[0]))],2).T;
    
    d = np.zeros((ndata,1))
    
    # Loop through all observations and create forward operator (ndata-by-mcell)
    print "Begin forward modeling " +str(int(ndata)) + " data points..."
    
    # Add counter to dsiplay progress. Good for large problems
    progress = -1;
    for ii in range(ndata):
    
        tx, ty, tz = get_T_mat(xn,yn,zn,rxLoc[ii,:])  
        
        G = Ptmi.dot(np.vstack((tx,ty,tz)))*Mxyz
        #%%
        # Forward operator
        d[ii,0] = G.dot(model)
    
        d_iter = np.floor(float(ii)/float(ndata)*10.);
        
        if  d_iter > progress:
            
            arg = "Done " + str(d_iter*10) + " %" 
            print arg
            progress = d_iter;
    
    
    print "Done 100% ...forward modeling completed!!\n"
    
    return d


