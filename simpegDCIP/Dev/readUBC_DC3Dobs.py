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

    tx  = []
    rx  = []
    d   = []
    wd  = []
        
    count = 0
    for ii in range(1, obsfile.shape[0]):
        
        if not obsfile[ii]:
            continue
        
        # First line is transmitter with number of receivers
        if count==0:
    
            tx.append(np.fromstring(obsfile[ii], dtype=float,sep=' ').T)
            count = int(tx[-1][-1])           
    
            continue
        
        temp = np.fromstring(obsfile[ii], dtype=float,sep=' ')
        
            
        rx.append(temp[0:6])
    
        if len(temp)==8:
            d.append(temp[6])
            wd.append(temp[7])
            
        elif len(temp)==7:
            d.append(temp[6])
                
        count = count - 1
                
                
    return tx, rx, d, wd
            
