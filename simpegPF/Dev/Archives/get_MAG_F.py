def fwr_MAG_F(mesh,B,M,rxLoc,flag):
    """ 
    Forward model magnetic data using integral equation
    
    INPUT:
    mesh        = Mesh in SimPEG format
    B           = Inducing field parameter [Binc, Bdecl, B0]
    M           = Magnetization information 
    [OPTIONS]
      1- [Minc, Mdecl] : Assumes uniform magnetization orientation
      2- [mx1,mx2,..., my1,...,mz1] : cell-based defined magnetization direction
      3- diag(M): Block diagonal matrix with [Mx, My, Mz] along the diagonal
        
    rxLox       = Observation location informat [obsx, obsy, obsz]
    
    flag        = 'tmi' | 'xyz' | 'full'
    [OPTIONS]
      1- tmi : Magnetization direction used and data are projected onto the
                inducing field direction F.shape([ndata, nc])
                
      2- xyz : Magnetization direction used and data are given in 3-components
                F.shape([3*ndata, nc])
                
      3- full: Full tensor matrix stored with shape([3*ndata, 3*nc])
        
    OUTPUT:
    F        = Linear forward modeling operation

    Created on Dec, 20th 2015

    @author: dominiquef
     """ 
    
    #%%
    from SimPEG import np, Utils, sp, mkvc
    from get_T_mat import get_T_mat
    
    
    xn = mesh.vectorNx;
    yn = mesh.vectorNy;
    zn = mesh.vectorNz;
    
    mcell = (len(xn)-1) * (len(yn)-1) * (len(zn)-1)
      
    ndata = rxLoc.shape[0]    
    
    #%% Create TMI projector
    
    # Convert Bdecination from north to cartesian 
    D = (450.-float(B[1]))%360.
    
    Ptmi = mkvc(np.r_[np.cos(np.deg2rad(B[0]))*np.cos(np.deg2rad(D)),
                      np.cos(np.deg2rad(B[0]))*np.sin(np.deg2rad(D)),
                        np.sin(np.deg2rad(B[0]))],2).T;
    
    # Pre-allocate space
    if flag=='tmi' | flag == 'xyz':
        
        # If assumes uniform magnetization direction
        if len(M) == 2:
            
            # Convert declination from north to cartesian
            Md = (450.-float(M[1]))%360.
            
            # Create magnetization matrix
            mx = np.cos(np.deg2rad(M[0])) * np.cos(np.deg2rad(Md))
            my = np.cos(np.deg2rad(M[0])) * np.sin(np.deg2rad(Md))
            mz = np.sin(np.deg2rad(M[0]))
            
            Mx = Utils.sdiag(np.ones([mcell])*mx*B[2])
            My = Utils.sdiag(np.ones([mcell])*my*B[2])
            Mz = Utils.sdiag(np.ones([mcell])*mz*B[2])
            
            Mxyz = sp.vstack((Mx,My,Mz));
        
        # Otherwise if given a vector 3*ncells
        elif len(M) == mesh.nC * 3:
            
            Mxyz = sp.spdiags(M,0,mesh.nC * 3,mesh.nC * 3)
        
        if flag == 'tmi':
            F = np.zeros((ndata, mesh.nC))
            
        elif flag == 'xyz':
            F = np.zeros((int(3*ndata), mesh.nC))
        
    elif flag == 'full':       
        F = np.zeros((int(3*ndata), int(3*mesh.nC)))
        
    else:
        print """Flag must be either 'tmi' | 'xyz' | 'full', please revised"""
        return
        
    
    # Loop through all observations and create forward operator (ndata-by-mcell)
    print "Begin calculation of forward operator: " + flag
    
    # Add counter to dsiplay progress. Good for large problems
    progress = -1;
    for ii in range(ndata):
    
        tx, ty, tz = get_T_mat(xn,yn,zn,rxLoc[ii,:])  
        
        if flag=='tmi':
            F[ii,:] = Ptmi.dot(np.vstack((tx,ty,tz)))*Mxyz
            
        elif flag == 'xyz':
            F[ii,:] = tx*Mxyz
            F[ii+ndata,:] = ty*Mxyz
            F[ii+2*ndata,:] = tz*Mxyz
            
        elif flag == 'full':       
            F[ii,:] = tx
            F[ii+ndata,:] = ty
            F[ii+2*ndata,:] = tz
            

    # Display progress   
    counter = np.floor(float(ii)/float(ndata)*10.);
    
    if  counter > progress:
        
        arg = "Done " + str(counter*10) + " %" 
        print arg
        progress = counter;
    
    
    print "Done 100% ...forward modeling completed!!\n"
    
    return F


