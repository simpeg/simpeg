def convertObs_DC3D_to_2D(Tx,Rx):
    
    from SimPEG import np
    import numpy.matlib as npm
    """
        Read list of 3D Tx Rx location and change coordinate system to distance
        along line assuming all data is acquired along line
        First transmitter pole is assumed to be at the origin

        Assumes flat topo for now...
    
        Input:
        :param Tx, Rx
        
        Output:
        :figure Tx2d, Rx2d
        
        Created on Mon December 7th, 2015
    
        @author: dominiquef
    
    """
    
                
    Tx2d = []
    Rx2d = []

    for ii in range(len(Tx)):
        
        if ii == 0:
            endp = Tx[0][0:2,0]
        
        nrx = Rx[ii].shape[0]
                  
        rP1 = np.sqrt( np.sum( ( endp - Tx[ii][0:2,0] )**2 , axis=0))
        rP2 = np.sqrt( np.sum( ( endp - Tx[ii][0:2,1] )**2 , axis=0))
        rC1 = np.sqrt( np.sum( ( npm.repmat(endp.T,nrx,1) - Rx[ii][:,0:2] )**2 , axis=1))
        rC2 = np.sqrt( np.sum( ( npm.repmat(endp.T,nrx,1) - Rx[ii][:,3:5] )**2 , axis=1))
        
        Tx2d.append( np.r_[rP1, rP2] )
        Rx2d.append( np.c_[rC1, rC2] )
            #np.savetxt(fid, data, fmt='%e',delimiter=' ',newline='\n')

    return Tx2d, Rx2d