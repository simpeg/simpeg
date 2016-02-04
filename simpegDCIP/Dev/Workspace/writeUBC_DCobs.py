def writeUBC_DCobs(fileName,Tx,Rx,d,wd, dtype):
    
    from SimPEG import np, mkvc
    import re
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
    fid.write('! GENERAL FORMAT\n')  
    
    for ii in range(len(Tx)):
        
        tx = np.asarray(Tx[ii])
        rx = np.asarray(Rx[ii])
        nrx = rx.shape[0]
        
        fid.write('\n')
        
        if re.match(dtype,'2D'):
            
            for jj in range(nrx):
                
                fid.writelines("%e " % ii for ii in mkvc(tx))
                fid.writelines("%e " % ii for ii in mkvc(rx[jj]))
                fid.write('%e %e\n'% (d[ii][jj],wd[ii][jj]))
                #np.savetxt(fid, np.c_[ rx ,np.asarray(d[ii]), np.asarray(wd[ii]) ], fmt='%e',delimiter=' ',newline='\n') 
        
        elif re.match(dtype,'3D'):
                                
            fid.write('\n')
            fid.writelines("%e " % ii for ii in mkvc(tx))
            fid.write('%i\n'% nrx)
            np.savetxt(fid, np.c_[ rx ,np.asarray(d[ii]), np.asarray(wd[ii]) ], fmt='%e',delimiter=' ',newline='\n') 
        
        
    fid.close()  
            
