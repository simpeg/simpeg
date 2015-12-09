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

def writeUBCocTreeFiles(fileName,mesh,modelDict=None):
    '''
        Write UBC ocTree mesh and model files from a simpeg ocTree mesh and model.

        :param str fileName: File to write to
        :param simpeg.Mesh.TreeMesh mesh: The mesh
        :param dictionary modelDict: The models in a dictionary, where the keys is the name of the of the model file

    '''

    # Calculate information to write in the file.
    # Number of cells in the underlying mesh
    nCunderMesh = np.array([h.size for h in mesh.h],dtype=np.int64)
    # The top-south-west most corner of the mesh
    tswCorn = mesh.x0 + np.array([0,0,np.sum(mesh.h[2])])
    # Smallest cell size
    smallCell = np.array([h.min() for h in mesh.h])
    # Number of cells
    nrCells = mesh.nC

    ## Extract iformation about the cells.
    # cell pointers
    cellPointers = np.array([c._pointer for c in mesh])
    # cell with
    cellW = np.array([ mesh._levelWidth(i) for i in cellPointers[:,-1] ])
    # Need to shift the pointers to work with UBC indexing
    # UBC Octree indexes always the top-left-close (top-south-west) corner first and orders the cells in z(top-down),x,y vs x,y,z(bottom-up).
    # Shift index up by 1
    ubcCellPt = cellPointers[:,0:-1].copy() + np.array([1.,1.,1.])
    # Need reindex the z index to be from the top-left-close corner and to be from the global top.
    ubcCellPt[:,2] = ( nCunderMesh[-1] + 2) - (ubcCellPt[:,2] + cellW)

    # Reorder the ubcCellPt
    ubcReorder = np.argsort(ubcCellPt.view(','.join(3*['float'])),axis=0,order=['f2','f1','f0'])[:,0]
    # Make a array with the pointers and the withs, that are order in the ubc ordering
    indArr = np.concatenate((ubcCellPt[ubcReorder,:],cellW[ubcReorder].reshape((-1,1)) ),axis=1)

    ## Write the UBC octree mesh file
    with open(fileName,'w') as mshOut:
        mshOut.write('{:.0f} {:.0f} {:.0f}\n'.format(nCunderMesh[0],nCunderMesh[1],nCunderMesh[2]))
        mshOut.write('{:.4f} {:.4f} {:.4f}\n'.format(tswCorn[0],tswCorn[1],tswCorn[2]))
        mshOut.write('{:.3f} {:.3f} {:.3f}\n'.format(smallCell[0],smallCell[1],smallCell[2]))
        mshOut.write('{:.0f} \n'.format(nrCells))
        np.savetxt(mshOut,indArr,fmt='%i')

    ## Print the models
    # Assign the model('s) to the object
    if modelDict is not None:
        # indUBCvector = np.argsort(cX0[np.argsort(np.concatenate((cX0[:,0:2],cX0[:,2:3].max() - cX0[:,2:3]),axis=1).view(','.join(3*['float'])),axis=0,order=('f2','f1','f0'))[:,0]].view(','.join(3*['float'])),axis=0,order=('f2','f1','f0'))[:,0]
        for item in modelDict.iteritems():
            # Save the data
            np.savetxt(item[0],item[1][ubcReorder],fmt='%3.5e')

def readUBCocTreeFiles(meshFile,modelFiles=None):
    """
        Read UBC 3D OcTree mesh and/or modelFiles

        Input:
        :param str meshFile: path to the UBC GIF OcTree mesh file to read
        :param list of str modelFiles: list of paths modelFiles

        Output:
        :return SimPEG.Mesh.TreeMesh mesh: The octree mesh
        :return list of ndarray's: models as a list of numpy array's
    """

    ## Read the file lines
    fileLines = np.genfromtxt(meshFile,dtype=str,delimiter='\n')
    # Extract the data
    nCunderMesh = np.array(fileLines[0].split(),dtype=float)
    # I think this is the case?
    if np.unique(nCunderMesh).size >1:
        raise Exception('SimPEG TreeMeshes have the same number of cell in all directions')
    tswCorn = np.array(fileLines[1].split(),dtype=float)
    smallCell = np.array(fileLines[2].split(),dtype=float)
    nrCells = np.array(fileLines[3].split(),dtype=float)
    # Read the index array
    indArr = np.genfromtxt(fileLines[4::],dtype=np.int)

    ## Calculate simpeg parameters
    h1,h2,h3 = [np.ones(nr)*sz for nr,sz in zip(nCunderMesh,smallCell)]
    x0 = tswCorn - np.sum(h3)
    # Need to convert the index array to a points list that complies with SimPEG TreeMesh.
    # Shift to start at 0
    simpegCellPt = indArr[:,0:-1].copy() - np.array([1.,1.,1.])
    # Need reindex the z index to be from the bottom-left-close corner and to be from the global bottom.
    simpegCellPt[:,2] = ( nCunderMesh[-1] + 2) - (simpegCellPt[:,2] - indArr[:,3])
    # Figure out the reordering
    simpegReorder = np.argsort(simpegCellPt.view(','.join(3*['float'])),axis=0,order=['f2','f1','f0'])[:,0]

    # Calculate the cell level
    simpegLevel = np.log2(np.min(nCunderMesh)) - np.log2(indArr[:,3])
    # Make a pointer matrix
    simpegPointers = np.concatenate((simpegCellPt[simpegReorder,:],simpegLevel[simpegReorder].reshape((-1,1))),axis=1)
    # Make an index set

    ## Make the tree mesh
    from SimPEG.Mesh import TreeMesh
    mesh = TreeMesh([h1,h2,h3],x0)
    mesh._cells = set([mesh._index(p) for p in simpegPointers.tolist()])

    if modelFiles is None:
        return mesh
    else:
        modList = []
        for modFile in modelFiles:
            modArr = np.loadtxt(modFile)
            if len(modArr.shape) == 1:
                modList.append(modArr[simpegReorder])
            else:
                modList.append(modArr[simpegReorder,:])
        return mesh, modList

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
    if model is not None:
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

def writeVTUFile(fileName,ocTreeMesh,modelDict=None):
    '''
    Function to write a VTU file from a SimPEG TreeMesh and model.
    '''
    from vtk import vtkXMLUnstructuredGridWriter as Writer
    from vtk.util.numpy_support import numpy_to_vtk

    # Make the object
    vtuObj = simpegOcTree2vtuObj(ocTreeMesh,modelDict)

    # Make the writer
    vtuWriteFilter = Writer()
    if float(vtk.VTK_VERSION.split('.')[0]) >=6:
        vtuWriteFilter.SetInputData(vtuObj)
    else:
        vtuWriteFilter.SetInput(vtuObj)
    vtuWriteFilter.SetInput(vtuObj)
    vtuWriteFilter.SetFileName(fileName)
    # Write the file
    vtuWriteFilter.Update()

def simpegOcTree2vtuObj(simpegOcTreeMesh,modelDict=None):
    '''
    Convert simpeg OcTree mesh and model to a VTK vtu object.

    '''
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray

    if str(type(simpegOcTreeMesh)).split()[-1][1:-2] not in 'SimPEG.Mesh.TreeMesh.TreeMesh':
        raise IOError('simpegOcTreeMesh is not a SimPEG TreeMesh.')

    # Make the data parts for the vtu object
    # Points
    ptsMat = simpegOcTreeMesh._gridN + simpegOcTreeMesh.x0
    vtkPts = vtk.vtkPoints()
    vtkPts.SetData(numpy_to_vtk(ptsMat,deep=True))
    # Cells
    cellConn = np.array([c.nodes for c in simpegOcTreeMesh],dtype=np.int64)

    cellsMat = np.concatenate((np.ones((cellConn.shape[0],1),dtype=np.int64)*cellConn.shape[1],cellConn),axis=1).ravel()
    cellsArr = vtk.vtkCellArray()
    cellsArr.SetNumberOfCells(cellConn.shape[0])
    cellsArr.SetCells(cellConn.shape[0],numpy_to_vtkIdTypeArray(cellsMat,deep=True))

    # Make the object
    vtuObj = vtk.vtkUnstructuredGrid()
    vtuObj.SetPoints(vtkPts)
    vtuObj.SetCells(vtk.VTK_VOXEL,cellsArr)
    # Add the level of refinement as a cell array
    cellSides = np.array([np.array(vtuObj.GetCell(i).GetBounds()).reshape((3,2)).dot(np.array([-1, 1])) for i in np.arange(vtuObj.GetNumberOfCells())])
    uniqueLevel, indLevel = np.unique(np.prod(cellSides,axis=1),return_inverse=True)
    refineLevelArr = numpy_to_vtk(indLevel.max() - indLevel,deep=1)
    refineLevelArr.SetName('octreeLevel')
    vtuObj.GetCellData().AddArray(refineLevelArr)
    # Assign the model('s) to the object
    if modelDict is not None:
        for item in modelDict.iteritems():
            # Convert numpy array
            vtkDoubleArr = numpy_to_vtk(item[1],deep=1)
            vtkDoubleArr.SetName(item[0])
            vtuObj.GetCellData().AddArray(vtkDoubleArr)

    return vtuObj

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
