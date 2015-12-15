def gen_DCIPsurvey(endl, mesh, stype, a, b, n):
    
    from SimPEG import np
    import re
    """
        Load in endpoints and survey specifications to generate Tx, Rx location
        stations.
        
        Assumes flat topo for now...
    
        Input:
        :param endl -> input endpoints [x1, y1, z1, x2, y2, z2]
        :object mesh -> SimPEG mesh object
        :switch stype -> "dpdp" (dipole-dipole) | "pdp" (pole-dipole)
        : param a, n -> pole seperation, number of rx dipoles per tx
        
        Output:
        :param Tx, Rx -> List objects for each tx location
            Lines: P1x, P1y, P1z, P2x, P2y, P2z
        
        Created on Wed December 9th, 2015
    
        @author: dominiquef
    
    """
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
    
    # Create line of P1 locations
    M = np.c_[stn_x, stn_y, np.ones(nstn).T*mesh.vectorNz[-1]]
    
    # Create line of P2 locations
    N = np.c_[stn_x+a*dl_x, stn_y+a*dl_y, np.ones(nstn).T*mesh.vectorNz[-1]]
    
    ## Build list of Tx-Rx locations depending on survey type
    # Dipole-dipole: Moving tx with [a] spacing -> [AB a MN1 a MN2 ... a MNn]
    # Pole-dipole: Moving pole on one end -> [A a MN1 a MN2 ... MNn a B]
    Tx = []
    Rx = []
    
    if not re.match(stype,'gradient'):
        
        for ii in range(0, int(nstn)-1): 
            
            
            if re.match(stype,'dpdp'):
                tx = np.c_[M[ii,:],N[ii,:]]
            elif re.match(stype,'pdp'):
                tx = np.c_[M[ii,:],M[ii,:]]
                
            #Rx.append(np.c_[M[ii+1:indx,:],N[ii+1:indx,:]])
            
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
            # Create line of P1 locations
            P1 = np.c_[stn_x, stn_y, np.ones(nstn).T*mesh.vectorNz[-1]]
            
            # Create line of P2 locations
            P2 = np.c_[stn_x+a*dl_x, stn_y+a*dl_y, np.ones(nstn).T*mesh.vectorNz[-1]]
            
            Rx.append(np.c_[P1,P2])
            Tx.append(tx)            
            
#==============================================================================
#     elif re.match(stype,'dpdp'):
#         
#         for ii in range(0, int(nstn)-2):  
#             
#             indx = np.min([ii+n+1,nstn])
#             Tx.append(np.c_[M[ii,:],N[ii,:]])
#             Rx.append(np.c_[M[ii+2:indx,:],N[ii+2:indx,:]])
#==============================================================================
            
    elif re.match(stype,'gradient'):
        
        # Gradient survey only requires Tx at end of line and creates a square
        # grid of receivers at in the middle at a pre-set minimum distance
        Tx.append(np.c_[M[0,:],N[-1,:]])
              
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
        lind = range(-nlin,nlin+1) 
        
        ngrad = nstn * len(lind)
        
        rx = np.zeros([ngrad,6])
        for ii in range( len(lind) ):
            
            # Move line in perpendicular direction by dipole spacing
            lxx = stn_x - lind[ii]*a*dl_y
            lyy = stn_y + lind[ii]*a*dl_x
            
            
            M = np.c_[ lxx, lyy , np.ones(nstn).T*mesh.vectorNz[-1]]
            N = np.c_[ lxx+a*dl_x, lyy+a*dl_y, np.ones(nstn).T*mesh.vectorNz[-1]]
            
            rx[(ii*nstn):((ii+1)*nstn),:] = np.c_[M,N]
            
        Rx.append(rx)
        
    else:
        print """stype must be either 'pdp', 'dpdp' or 'gradient'. """

        
   
    return Tx, Rx             
