def writeUBC_DC3Dobs(fileName,Tx,Rx,d,wd):
    
    from SimPEG import np, mkvc
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
    fid = open(fileName,'w')
    fid.write('GENERAL FORMAT\n')    
    
    for ii in range(len(Tx)):
        
        tx = np.asarray(Tx[ii])
        rx = np.asarray(Rx[ii])
        nrx = rx.shape[0]
        
        fid.write('\n')
        fid.writelines("%e " % ii for ii in mkvc(tx))
        fid.write('%i\n'% nrx)
        np.savetxt(fid, np.c_[ rx ,np.asarray(d[ii]), np.asarray(wd[ii]) ], fmt='%e',delimiter=' ',newline='\n') 
        
        
    fid.close()  
            
