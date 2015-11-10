def read_MAGfwr_inp(input_file):
    """Read input files for forward modeling MAG data with integral form
    INPUT:
    input_file: File name containing the forward parameter
    
    OUTPUT:
    mshfile
    obsfile
    modfile
    magfile
    topofile
    # All files should be in the working directory, otherwise the path must
    # be specified.

    Created on Jul 17, 2013
    
    @author: dominiquef
    """

    
    fid = open(input_file,'r')
    
    line = fid.readline()
    l_input  = line.split('!')
    mshfile = l_input[0].rstrip()
    
    line = fid.readline()
    l_input  = line.split('!')
    obsfile = l_input[0].rstrip()
    
    line = fid.readline()
    l_input = line.split('!') 
    modfile = l_input[0].rstrip()
    
    line = fid.readline()
    l_input = line.split('!') 
    if l_input=='null':
        magfile = []
        
    else:
        magfile = l_input[0].rstrip()
        
        
    line = fid.readline()
    l_input = line.split('!') 
    if l_input=='null':
        topofile = []
        
    else:
        topofile = l_input[0].rstrip()
      
    return mshfile, obsfile, modfile, magfile, topofile