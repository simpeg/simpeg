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
    fopen = open(fileName,'r')
    lines = fopen.readlines()[:]
    fopen.close()

    dim = np.array(lines[0].split(),dtype=float)
    
    model = np.array(lines[1:],dtype=float)
    
    # Permute the second dimension to flip the order
    model = model.reshape(dim[1],dim[0])
    
    model = model[::-1,:]
    model = np.transpose(model, (1, 0))
    model = mkvc(model)


    return model


