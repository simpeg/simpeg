'''
Created on Jul 17, 2013

@author: dominiquef
'''
def read_MAG_obs(obs_file):
    """Read input files for the lp_norm script"""
    from numpy import zeros
    
    fid = open(obs_file,'r')
    

    # First line has the declination, inclination and amplitude of B0
    line = fid.readline()
    line = line.split()
    Incl = float(line[0]) 
    Decl = float(line[1])
    B0   = float(line[2])

    # Second line has the magnetization orientation and a flag 
    line = fid.readline()
    line = line.split()
    Minc = float(line[0])
    Mdec = float(line[1])
    FLAG = float(line[2])

    # Third line has the number of rows
    line = fid.readline()
    line = line.split()
    ndat = int(line[0])

    # Pre-allocate space for obsx, obsy, obsz, data, uncert
    obsx = zeros((ndat,1), dtype=float)
    obsy = zeros((ndat,1), dtype=float)
    obsz = zeros((ndat,1), dtype=float)
    data = zeros((ndat,1), dtype=float)
    unct = zeros((ndat,1), dtype=float)

    for ii in range(ndat):

        line = fid.readline()
        line = line.split()  

        obsx[ii] = line[0]
        obsy[ii] = line[1]
        obsz[ii] = line[2]

        if len(line)>3:

            data[ii] = line[3]

        if len(line)>4:

            unct[ii] = line[4]



    return Decl, Incl, B0, Mdec, Minc, obsx, obsy, obsz, data, unct