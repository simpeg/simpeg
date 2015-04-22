import numpy as np
from scipy import sparse as sp
from matutils import mkvc, ndgrid, sub2ind, sdiag
from codeutils import asArray_N_x_Dim
from codeutils import isScalar
import os

def exampleLrmGrid(nC, exType):
    assert type(nC) == list, "nC must be a list containing the number of nodes"
    assert len(nC) == 2 or len(nC) == 3, "nC must either two or three dimensions"
    exType = exType.lower()

    possibleTypes = ['rect', 'rotate']
    assert exType in possibleTypes, "Not a possible example type."

    if exType == 'rect':
        return list(ndgrid([np.cumsum(np.r_[0, np.ones(nx)/nx]) for nx in nC], vector=False))
    elif exType == 'rotate':
        if len(nC) == 2:
            X, Y = ndgrid([np.cumsum(np.r_[0, np.ones(nx)/nx]) for nx in nC], vector=False)
            amt = 0.5-np.sqrt((X - 0.5)**2 + (Y - 0.5)**2)
            amt[amt < 0] = 0
            return [X + (-(Y - 0.5))*amt, Y + (+(X - 0.5))*amt]
        elif len(nC) == 3:
            X, Y, Z = ndgrid([np.cumsum(np.r_[0, np.ones(nx)/nx]) for nx in nC], vector=False)
            amt = 0.5-np.sqrt((X - 0.5)**2 + (Y - 0.5)**2 + (Z - 0.5)**2)
            amt[amt < 0] = 0
            return [X + (-(Y - 0.5))*amt, Y + (-(Z - 0.5))*amt, Z + (-(X - 0.5))*amt]

def meshTensor(value):
    """
        **meshTensor** takes a list of numbers and tuples that have the form::

            mT = [ float, (cellSize, numCell), (cellSize, numCell, factor) ]

        For example, a time domain mesh code needs
        many time steps at one time::

            [(1e-5, 30), (1e-4, 30), 1e-3]

        Means take 30 steps at 1e-5 and then 30 more at 1e-4,
        and then one step of 1e-3.

        Tensor meshes can also be created by increase factors::

            [(10.0, 5, -1.3), (10.0, 50), (10.0, 5, 1.3)]

        When there is a third number in the tuple, it
        refers to the increase factor, if this number
        is negative this section of the tensor is flipped right-to-left.

        .. plot::

            from SimPEG import Mesh
            tx = [(10.0,10,-1.3),(10.0,40),(10.0,10,1.3)]
            ty = [(10.0,10,-1.3),(10.0,40)]
            M = Mesh.TensorMesh([tx, ty])
            M.plotGrid(showIt=True)

    """
    if type(value) is not list:
        raise Exception('meshTensor must be a list of scalars and tuples.')

    proposed = []
    for v in value:
        if isScalar(v):
            proposed += [float(v)]
        elif type(v) is tuple and len(v) == 2:
            proposed += [float(v[0])]*int(v[1])
        elif type(v) is tuple and len(v) == 3:
            start = float(v[0])
            num = int(v[1])
            factor = float(v[2])
            pad = ((np.ones(num)*np.abs(factor))**(np.arange(num)+1))*start
            if factor < 0: pad = pad[::-1]
            proposed += pad.tolist()
        else:
            raise Exception('meshTensor must contain only scalars and len(2) or len(3) tuples.')

    return np.array(proposed)

def closestPoints(mesh, pts, gridLoc='CC'):
    """
        Move a list of points to the closest points on a grid.

        :param simpeg.Mesh.BaseMesh mesh: The mesh
        :param numpy.ndarray pts: Points to move
        :param string gridLoc: ['CC', 'N', 'Fx', 'Fy', 'Fz', 'Ex', 'Ex', 'Ey', 'Ez']
        :rtype: numpy.ndarray
        :return: nodeInds
    """

    pts = asArray_N_x_Dim(pts, mesh.dim)
    grid = getattr(mesh, 'grid' + gridLoc)
    nodeInds = np.empty(pts.shape[0], dtype=int)

    for i, pt in enumerate(pts):
        if mesh.dim == 1:
            nodeInds[i] = ((pt - grid)**2).argmin()
        else:
            nodeInds[i] = ((np.tile(pt, (grid.shape[0],1)) - grid)**2).sum(axis=1).argmin()

    return nodeInds

def readUBCTensorMesh(fileName):
    """
        Read UBC GIF 3DTensor mesh and generate 3D Tensor mesh in simpegTD

        Input:
        :param fileName, path to the UBC GIF mesh file

        Output:
        :param SimPEG TensorMesh object
        :return
    """

    # Interal function to read cell size lines for the UBC mesh files.
    def readCellLine(line):
        for seg in line.split():
            if '*' in seg:
                st = seg
                sp = seg.split('*')
                re = np.array(sp[0],dtype=int)*(' ' + sp[1])
                line = line.replace(st,re.strip())
        return np.array(line.split(),dtype=float)

    # Read the file as line strings, remove lines with comment = !
    msh = np.genfromtxt(fileName,delimiter='\n',dtype=np.str,comments='!')

    # Fist line is the size of the model
    sizeM = np.array(msh[0].split(),dtype=float)
    # Second line is the South-West-Top corner coordinates.
    x0 = np.array(msh[1].split(),dtype=float)
    # Read the cell sizes
    h1 = readCellLine(msh[2])
    h2 = readCellLine(msh[3])
    h3temp = readCellLine(msh[4])
    h3 = h3temp[::-1] # Invert the indexing of the vector to start from the bottom.
    # Adjust the reference point to the bottom south west corner
    x0[2] = x0[2] - np.sum(h3)
    # Make the mesh
    from SimPEG import Mesh
    tensMsh = Mesh.TensorMesh([h1,h2,h3],x0)
    return tensMsh

def readUBCTensorModel(fileName, mesh):
    """
        Read UBC 3DTensor mesh model and generate 3D Tensor mesh model in simpeg

        Input:
        :param fileName, path to the UBC GIF mesh file to read
        :param mesh, TensorMesh object, mesh that coresponds to the model 

        Output:
        :return numpy array, model with TensorMesh ordered
    """
    f = open(fileName, 'r')
    model = np.array(map(float, f.readlines()))
    f.close()
    model = np.reshape(model, (mesh.nCz, mesh.nCx, mesh.nCy), order = 'F')
    model = model[::-1,:,:]
    model = np.transpose(model, (1, 2, 0))
    model = mkvc(model)

    return model

def writeUBCTensorMesh(fileName, mesh):
    """
        Writes a SimPEG TensorMesh to a UBC-GIF format mesh file.

        :param str fileName: File to write to
        :param simpeg.Mesh.TensorMesh mesh: The mesh
        
    """
    assert mesh.dim == 3
    s = ''
    s += '%i %i %i\n' %tuple(mesh.vnC)
    origin = mesh.x0 + np.array([0,0,mesh.hz.sum()]) # Have to it in the same operation or use mesh.x0.copy(), otherwise the mesh.x0 is updated.
    origin.dtype = float

    s += '%.2f %.2f %.2f\n' %tuple(origin)
    s += ('%.2f '*mesh.nCx+'\n')%tuple(mesh.hx)
    s += ('%.2f '*mesh.nCy+'\n')%tuple(mesh.hy)
    s += ('%.2f '*mesh.nCz+'\n')%tuple(mesh.hz[::-1])
    f = open(fileName, 'w')
    f.write(s)
    f.close()

def writeUBCTensorModel(fileName, mesh, model):
    """
        Writes a model associated with a SimPEG TensorMesh
        to a UBC-GIF format model file.

        :param str fileName: File to write to
        :param simpeg.Mesh.TensorMesh mesh: The mesh
        :param numpy.ndarray model: The model
    """

    # Reshape model to a matrix
    modelMat = mesh.r(model,'CC','CC','M')
    # Transpose the axes
    modelMatT = modelMat.transpose((2,0,1))
    # Flip z to positive down
    modelMatTR = mkvc(modelMatT[::-1,:,:])

    np.savetxt(fileName, modelMatTR.ravel())


def readVTRFile(fileName):
    """
        Read VTK Rectilinear (vtr xml file) and return SimPEG Tensor mesh and model

        Input:
        :param vtrFileName, path to the vtr model file to write to

        Output:
        :return SimPEG TensorMesh object
        :return SimPEG model dictionary
        
    """
    # Import
    from vtk import vtkXMLRectilinearGridReader as vtrFileReader
    from vtk.util.numpy_support import vtk_to_numpy

    # Read the file
    vtrReader = vtrFileReader()
    vtrReader.SetFileName(fileName)
    vtrReader.Update()
    vtrGrid = vtrReader.GetOutput()
    # Sort information
    hx = np.abs(np.diff(vtk_to_numpy(vtrGrid.GetXCoordinates())))
    xR = vtk_to_numpy(vtrGrid.GetXCoordinates())[0]
    hy = np.abs(np.diff(vtk_to_numpy(vtrGrid.GetYCoordinates())))
    yR = vtk_to_numpy(vtrGrid.GetYCoordinates())[0]
    zD = np.diff(vtk_to_numpy(vtrGrid.GetZCoordinates()))
    # Check the direction of hz
    if np.all(zD < 0):
        hz = np.abs(zD[::-1])
        zR = vtk_to_numpy(vtrGrid.GetZCoordinates())[-1]
    else:
        hz = np.abs(zD)
        zR = vtk_to_numpy(vtrGrid.GetZCoordinates())[0]
    x0 = np.array([xR,yR,zR])

    # Make the SimPEG object
    from SimPEG import Mesh
    tensMsh = Mesh.TensorMesh([hx,hy,hz],x0)

    # Grap the models
    modelDict = {}
    for i in np.arange(vtrGrid.GetCellData().GetNumberOfArrays()):
        modelName = vtrGrid.GetCellData().GetArrayName(i)
        if np.all(zD < 0):
            modFlip = vtk_to_numpy(vtrGrid.GetCellData().GetArray(i))
            tM = tensMsh.r(modFlip,'CC','CC','M')
            modArr = tensMsh.r(tM[:,:,::-1],'CC','CC','V')
        else:
            modArr = vtk_to_numpy(vtrGrid.GetCellData().GetArray(i))
        modelDict[modelName] = modArr

    # Return the data
    return tensMsh, modelDict

def writeVTRFile(fileName,mesh,model=None):
    """
    Makes and saves a VTK rectilinear file (vtr) for a simpeg Tensor mesh and model.

    Input:
    :param str, path to the output vtk file
    :param mesh, SimPEG TensorMesh object - mesh to be transfer to VTK
    :param model, dictionary of numpy.array - Name('s) and array('s). Match number of cells

    """
    # Import
    from vtk import vtkRectilinearGrid as rectGrid, vtkXMLRectilinearGridWriter as rectWriter
    from vtk.util.numpy_support import numpy_to_vtk

    # Deal with dimensionalities
    if mesh.dim >= 1:
        vX = mesh.vectorNx
        xD = mesh.nNx
        yD,zD = 1,1
        vY, vZ = np.array([0,0])
    if mesh.dim >= 2:
        vY = mesh.vectorNy
        yD = mesh.nNy
    if mesh.dim == 3:
        vZ = mesh.vectorNz
        zD = mesh.nNz
    # Use rectilinear VTK grid.
    # Assign the spatial information.
    vtkObj = rectGrid()
    vtkObj.SetDimensions(xD,yD,zD)
    vtkObj.SetXCoordinates(numpy_to_vtk(vX,deep=1))
    vtkObj.SetYCoordinates(numpy_to_vtk(vY,deep=1))
    vtkObj.SetZCoordinates(numpy_to_vtk(vZ,deep=1))

    # Assign the model('s) to the object
    for item in model.iteritems():
        # Convert numpy array
        vtkDoubleArr = numpy_to_vtk(item[1],deep=1)
        vtkDoubleArr.SetName(item[0])
        vtkObj.GetCellData().AddArray(vtkDoubleArr)
    # Set the active scalar
    vtkObj.GetCellData().SetActiveScalars(model.keys()[0])
    vtkObj.Update()


    # Check the extension of the fileName
    ext = os.path.splitext(fileName)[1]
    if ext is '':
        fileName = fileName + '.vtr'
    elif ext not in '.vtr':
        raise IOError('{:s} is an incorrect extension, has to be .vtr')
    # Write the file.
    vtrWriteFilter = rectWriter()
    vtrWriteFilter.SetInput(vtkObj)
    vtrWriteFilter.SetFileName(fileName)
    vtrWriteFilter.Update()


def ExtractCoreMesh(xyzlim, mesh, meshType='tensor'):
    """
        Extracts Core Mesh from Global mesh
        xyzlim: 2D array [ndim x 2]
        mesh: SimPEG mesh
        This function ouputs: 
            - actind: corresponding boolean index from global to core
            - meshcore: core SimPEG mesh  
        Warning: 1D and 2D has not been tested
    """
    from SimPEG import Mesh
    if mesh.dim ==1:
        xyzlim = xyzlim.flatten()
        xmin, xmax = xyzlim[0], xyzlim[1]
        
        xind = np.logical_and(mesh.vectorCCx>xmin, mesh.vectorCCx<xmax)        
        
        xc = mesh.vectorCCx[xind]

        hx = mesh.hx[xind]
        
        x0 = [xc[0]-hx[0]*0.5, yc[0]-hy[0]*0.5]
        
        meshCore = Mesh.TensorMesh([hx, hy] ,x0=x0)
        
        actind = (mesh.gridCC[:,0]>xmin) & (mesh.gridCC[:,0]<xmax)
        
    elif mesh.dim ==2:
        xmin, xmax = xyzlim[0,0], xyzlim[0,1]
        ymin, ymax = xyzlim[1,0], xyzlim[1,1]

        yind = np.logical_and(mesh.vectorCCy>ymin, mesh.vectorCCy<ymax)
        zind = np.logical_and(mesh.vectorCCz>zmin, mesh.vectorCCz<zmax)        

        xc = mesh.vectorCCx[xind]
        yc = mesh.vectorCCy[yind]

        hx = mesh.hx[xind]
        hy = mesh.hy[yind]
        
        x0 = [xc[0]-hx[0]*0.5, yc[0]-hy[0]*0.5]
        
        meshCore = Mesh.TensorMesh([hx, hy] ,x0=x0)
        
        actind = (mesh.gridCC[:,0]>xmin) & (mesh.gridCC[:,0]<xmax) \
               & (mesh.gridCC[:,1]>ymin) & (mesh.gridCC[:,1]<ymax) \
        
    elif mesh.dim==3:
        xmin, xmax = xyzlim[0,0], xyzlim[0,1]
        ymin, ymax = xyzlim[1,0], xyzlim[1,1]
        zmin, zmax = xyzlim[2,0], xyzlim[2,1]
        
        xind = np.logical_and(mesh.vectorCCx>xmin, mesh.vectorCCx<xmax)
        yind = np.logical_and(mesh.vectorCCy>ymin, mesh.vectorCCy<ymax)
        zind = np.logical_and(mesh.vectorCCz>zmin, mesh.vectorCCz<zmax)        

        xc = mesh.vectorCCx[xind]
        yc = mesh.vectorCCy[yind]
        zc = mesh.vectorCCz[zind]

        hx = mesh.hx[xind]
        hy = mesh.hy[yind]
        hz = mesh.hz[zind]
        
        x0 = [xc[0]-hx[0]*0.5, yc[0]-hy[0]*0.5, zc[0]-hz[0]*0.5]
        
        meshCore = Mesh.TensorMesh([hx, hy, hz] ,x0=x0)
        
        actind = (mesh.gridCC[:,0]>xmin) & (mesh.gridCC[:,0]<xmax) \
               & (mesh.gridCC[:,1]>ymin) & (mesh.gridCC[:,1]<ymax) \
               & (mesh.gridCC[:,2]>zmin) & (mesh.gridCC[:,2]<zmax)
                
    else:
        raise(Exception("Not implemented!"))
    
    
    return actind, meshCore


if __name__ == '__main__':
    from SimPEG import Mesh
    import matplotlib.pyplot as plt
    tx = [(10.0,10,-1.3),(10.0,40),(10.0,10,1.3)]
    ty = [(10.0,10,-1.3),(10.0,40)]
    M = Mesh.TensorMesh([tx, ty])
    M.plotGrid()
    plt.gca().axis('tight')
    plt.show()
