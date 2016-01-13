def readUBC_DC2DModel(fileName):

    from SimPEG import np, mkvc
    """
        Read UBC GIF 2DTensor model and generate 2D Tensor model in simpeg

        Input:
        :param fileName, path to the UBC GIF 2D model file

        Output:
        :param SimPEG TensorMesh 2D object
        :return
        
        Created on Thu Nov 12 13:14:10 2015

        @author: dominiquef

    """
    
    # Open fileand skip header... assume that we know the mesh already

    obsfile = np.genfromtxt(fileName,delimiter=' \n',dtype=np.str,comments='!')
    
    dim = np.array(obsfile[0].split(),dtype=float)
    
    temp = np.array(obsfile[1].split(),dtype=float)
    
    if len(temp) > 1:
        model = np.zeros(dim)
        
        for ii in range(len(obsfile)-1):
            mm = np.array(obsfile[ii+1].split(),dtype=float)
            model[:,ii] = mm
            
        model = model[:,::-1]
        
    else:
        
        if len(obsfile[1:])==1:
            mm = np.array(obsfile[1:].split(),dtype=float)
            
        else:
            mm = np.array(obsfile[1:],dtype=float)
            
        # Permute the second dimension to flip the order
        model = mm.reshape(dim[1],dim[0])
    
        model = model[::-1,:]
        model = np.transpose(model, (1, 0))
        
    model = mkvc(model)


    return model


