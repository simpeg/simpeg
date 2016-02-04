def readUBC_DC2DMesh(fileName):

    from SimPEG import np
    """
        Read UBC GIF 2DTensor mesh and generate 2D Tensor mesh in simpeg

        Input:
        :param fileName, path to the UBC GIF mesh file

        Output:
        :param SimPEG TensorMesh 2D object
        :return
        
        Created on Thu Nov 12 13:14:10 2015

        @author: dominiquef

    """
    
    # Open file 
    fopen = open(fileName,'r')
    
    # Read down the file and unpack dx vector
    def unpackdx(fid,nrows):
        for ii in range(nrows):
            
            line = fid.readline()
            var = np.array(line.split(),dtype=float)
            
            if ii==0:
                x0= var[0]
                xvec = np.ones(int(var[2])) * (var[1] - var[0]) / int(var[2])
                xend = var[1]            
                
            else:
                xvec = np.hstack((xvec,np.ones(int(var[1])) * (var[0] - xend) / int(var[1])))
                xend = var[0] 
                
        return x0, xvec
        
    #%% Start with dx block
    # First line specifies the number of rows for x-cells
    line = fopen.readline()
    nl = np.array(line.split(),dtype=float)

    [x0, dx] = unpackdx(fopen,nl) 
    

    #%% Move down the file until reaching the z-block
    line = fopen.readline()
    if not line:
        line = fopen.readline()
        
    #%% End with dz block
    # First line specifies the number of rows for z-cells
    line = fopen.readline()
    nl = np.array(line.split(),dtype=float)

    [z0, dz] = unpackdx(fopen,nl) 
    
    # Flip z0 to be the bottom of the mesh for SimPEG
    z0 = z0 - sum(dz)
    dz = dz[::-1]
    #%% Make the mesh using SimPEG
    
    from SimPEG import Mesh
    tensMsh = Mesh.TensorMesh([dx,dz],(x0, z0))
    return tensMsh


