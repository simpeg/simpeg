def readUBC_DC3Dobs(fileName):
    
    from SimPEG import np
    """
        Read UBC GIF DCIP 3D observation file and generate arrays for tx-rx location
    
        Input:
        :param fileName, path to the UBC GIF 3D obs file
    
        Output:
        :param rx, tx, d, wd
        :return
        
        Created on Mon December 7th, 2015
    
        @author: dominiquef
    
    """
       
    # Load file
    obsfile = np.genfromtxt(fileName,delimiter=' \n',dtype=np.str,comments='!')
    
    # Pre-allocate
    Tx = []
    Rx = []
    d = []
    wd = []
    
    # Countdown for number of obs/tx
    count = 0
    for ii in range(obsfile.shape[0]):
        
        if not obsfile[ii]:
            continue
        
        # First line is transmitter with number of receivers
        if count==0:
    
            temp = (np.fromstring(obsfile[ii], dtype=float,sep=' ').T)
            count = int(temp[-1])
            temp = np.reshape(temp[0:-1],[2,3]).T
            
            Tx.append(temp)
            rx = []
            continue
        
        temp = np.fromstring(obsfile[ii], dtype=float,sep=' ')
        
            
        rx.append(temp)          
        
        count = count -1        
        
        # Reach the end of  
        if count == 0:
            temp = np.asarray(rx)
            Rx.append(temp[:,0:6])
            
            # Check for data + uncertainties
            if temp.shape[1]==8:
                d.append(temp[:,6])
                wd.append(temp[:,7])
                
            # Check for data only    
            elif temp.shape[1]==7:
                d.append(temp[:,6])
            
    return Tx, Rx, d, wd
                
