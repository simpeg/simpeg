def readUBC_DC2DLoc(fileName):

    from SimPEG import np
    """
        Read UBC GIF 2D observation file and generate arrays for tx-rx location

        Input:
        :param fileName, path to the UBC GIF 2D model file

        Output:
        :param rx, tx
        :return
        
        Created on Thu Nov 12 13:14:10 2015

        @author: dominiquef

    """
    
    # Open fileand skip header... assume that we know the mesh already
#==============================================================================
#     fopen = open(fileName,'r')
#     lines = fopen.readlines()
#     fopen.close()
#==============================================================================

    # Load file
    obsfile = np.genfromtxt(fileName,delimiter=' \n',dtype=np.str,comments='!')
    
    # Check first line and figure out if 2D or 3D file format
    line = np.array(obsfile[0].split(),dtype=float)   
    
    tx_A  = []
    tx_B  = []
    rx_M  = []
    rx_N  = []
    d   = []
    wd  = []
    
    for ii in range(obsfile.shape[0]):
        
        # If len==3, then simple format where tx-rx is listed on each line
        if len(line) == 4:
        
            temp = np.fromstring(obsfile[ii], dtype=float,sep=' ')
            tx_A = np.hstack((tx_A,temp[0]))
            tx_B = np.hstack((tx_B,temp[1]))
            rx_M = np.hstack((rx_M,temp[2]))
            rx_N = np.hstack((rx_N,temp[3]))
            
        
    rx = np.transpose(np.array((rx_M,rx_N)))    
    tx = np.transpose(np.array((tx_A,tx_B)))
    
    return tx, rx, d, wd


