import numpy as np, os
from SimPEG import Utils
import six

class TensorMeshIO(object):

    @classmethod
    def readUBC(TensorMesh, fileName):
        """
            Read UBC GIF 3D tensor mesh and generate 3D TensorMesh in SimPEG.

            :param string fileName: path to the UBC GIF mesh file
            :rtype: TensorMesh
            :return: The tensor mesh for the fileName.
        """

        # Interal function to read cell size lines for the UBC mesh files.
        def readCellLine(line):
            for seg in line.split():
                if '*' in seg:
                    st = seg
                    sp = seg.split('*')
                    re = int(sp[0])*(' ' + sp[1])
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
        tensMsh = TensorMesh([h1,h2,h3],x0)
        return tensMsh

    @classmethod
    def readVTK(TensorMesh, fileName):
        """
            Read VTK Rectilinear (vtr xml file) and return SimPEG Tensor mesh and model

            Input:
            :param string fileName: path to the vtr model file to read
            :rtype: tuple
            :return: (TensorMesh, modelDictionary)

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
        tensMsh = TensorMesh([hx,hy,hz],x0)

        # Grap the models
        models = {}
        for i in np.arange(vtrGrid.GetCellData().GetNumberOfArrays()):
            modelName = vtrGrid.GetCellData().GetArrayName(i)
            if np.all(zD < 0):
                modFlip = vtk_to_numpy(vtrGrid.GetCellData().GetArray(i))
                tM = tensMsh.r(modFlip,'CC','CC','M')
                modArr = tensMsh.r(tM[:,:,::-1],'CC','CC','V')
            else:
                modArr = vtk_to_numpy(vtrGrid.GetCellData().GetArray(i))
            models[modelName] = modArr

        # Return the data
        return tensMsh, models

    def writeVTK(mesh, fileName, models=None):
        """
        Makes and saves a VTK rectilinear file (vtr) for a simpeg Tensor mesh and model.

        Input:
        :param string fileName: path to the output vtk file
        :param dict models: dictionary of numpy.array - Name('s) and array('s). Match number of cells

        """
        # Import
        from vtk import vtkRectilinearGrid as rectGrid, vtkXMLRectilinearGridWriter as rectWriter, VTK_VERSION
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
        if models is not None:
            for item in six.iteritems(models):
                # Convert numpy array
                vtkDoubleArr = numpy_to_vtk(item[1],deep=1)
                vtkDoubleArr.SetName(item[0])
                vtkObj.GetCellData().AddArray(vtkDoubleArr)
            # Set the active scalar
            vtkObj.GetCellData().SetActiveScalars(models.keys()[0])

        # Check the extension of the fileName
        ext = os.path.splitext(fileName)[1]
        if ext is '':
            fileName = fileName + '.vtr'
        elif ext not in '.vtr':
            raise IOError('{:s} is an incorrect extension, has to be .vtr')
        # Write the file.
        vtrWriteFilter = rectWriter()
        if float(VTK_VERSION.split('.')[0]) >=6:
            vtrWriteFilter.SetInputData(vtkObj)
        else:
            vtuWriteFilter.SetInput(vtuObj)
        vtrWriteFilter.SetFileName(fileName)
        vtrWriteFilter.Update()

    def _toVTRObj(mesh,models=None):
        """
        Makes and saves a VTK rectilinear file (vtr) for a simpeg Tensor mesh and model.

        Input:
        :param str, path to the output vtk file
        :param mesh, SimPEG TensorMesh object - mesh to be transfer to VTK
        :param models, dictionary of numpy.array - Name('s) and array('s). Match number of cells

        """
        # Import
        from vtk import vtkRectilinearGrid as rectGrid, VTK_VERSION
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
        if models is not None:
            for item in models.iteritems():
                # Convert numpy array
                vtkDoubleArr = numpy_to_vtk(item[1],deep=1)
                vtkDoubleArr.SetName(item[0])
                vtkObj.GetCellData().AddArray(vtkDoubleArr)
            # Set the active scalar
            vtkObj.GetCellData().SetActiveScalars(models.keys()[0])
        return vtkObj

    def readModelUBC(mesh, fileName):
        """
            Read UBC 3DTensor mesh model and generate 3D Tensor mesh model in simpeg

            :param string fileName: path to the UBC GIF mesh file to read
            :rtype: numpy.ndarray
            :return: model with TensorMesh ordered
        """
        f = open(fileName, 'r')
        model = np.array(list(map(float, f.readlines())))
        f.close()
        model = np.reshape(model, (mesh.nCz, mesh.nCx, mesh.nCy), order = 'F')
        model = model[::-1,:,:]
        model = np.transpose(model, (1, 2, 0))
        model = Utils.mkvc(model)
        return model

    def writeModelUBC(mesh, fileName, model):
        """
            Writes a model associated with a SimPEG TensorMesh
            to a UBC-GIF format model file.

            :param string fileName: File to write to
            :param numpy.ndarray model: The model
        """

        # Reshape model to a matrix
        modelMat = mesh.r(model,'CC','CC','M')
        # Transpose the axes
        modelMatT = modelMat.transpose((2,0,1))
        # Flip z to positive down
        modelMatTR = Utils.mkvc(modelMatT[::-1,:,:])

        np.savetxt(fileName, modelMatTR.ravel())

    def writeVectorUBC(mesh, fileName, model):
        """
            Writes a vector model associated with a SimPEG TensorMesh
            to a UBC-GIF format model file.

            :param string fileName: File to write to
            :param numpy.ndarray model: The model
        """

        modelMatTR = np.zeros_like(model)

        for ii in range(3):
            # Reshape model to a matrix
            modelMat = mesh.r(model[:, ii], 'CC', 'CC', 'M')
            # Transpose the axes
            modelMatT = modelMat.transpose((2, 0, 1))
            # Flip z to positive down
            modelMatTR[:, ii] = Utils.mkvc(modelMatT[::-1, :, :])

        np.savetxt(fileName, modelMatTR)

    def writeUBC(mesh, fileName, models=None):
        """
            Writes a SimPEG TensorMesh to a UBC-GIF format mesh file.

            :param string fileName: File to write to
            :param dict models: A dictionary of the models

        """
        assert mesh.dim == 3
        s = ''
        s += '{0:d} {1:d} {2:d}\n'.format(*tuple(mesh.vnC))
        origin = mesh.x0 + np.array([0,0,mesh.hz.sum()]) # Have to it in the same operation or use mesh.x0.copy(), otherwise the mesh.x0 is updated.
        origin.dtype = float

        s += '{0:.2f} {1:.2f} {2:.2f}\n'.format(*tuple(origin))
        s += ('%.2f '*mesh.nCx+'\n')%tuple(mesh.hx)
        s += ('%.2f '*mesh.nCy+'\n')%tuple(mesh.hy)
        s += ('%.2f '*mesh.nCz+'\n')%tuple(mesh.hz[::-1])
        f = open(fileName, 'w')
        f.write(s)
        f.close()

        if models is None: return
        assert type(models) is dict, 'models must be a dict'
        for key in models:
            assert type(key) is str, 'The dict key is a file name'
            mesh.writeModelUBC(key, models[key])

class TreeMeshIO(object):

    def writeUBC(mesh, fileName, models=None):
        """
            Write UBC ocTree mesh and model files from a simpeg ocTree mesh and model.

            :param string fileName: File to write to
            :param dict models: The models in a dictionary, where the keys is the name of the of the model file
        """

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
        head = ('{:.0f} {:.0f} {:.0f}\n'.format(nCunderMesh[0],nCunderMesh[1],nCunderMesh[2])+
            '{:.4f} {:.4f} {:.4f}\n'.format(tswCorn[0],tswCorn[1],tswCorn[2])+
            '{:.3f} {:.3f} {:.3f}\n'.format(smallCell[0],smallCell[1],smallCell[2])+
            '{:.0f} \n'.format(nrCells))
        np.savetxt(fileName,indArr,fmt='%i',header=head,comments='')

        ## Print the models
        # Assign the model('s) to the object
        if models is not None:
            # indUBCvector = np.argsort(cX0[np.argsort(np.concatenate((cX0[:,0:2],cX0[:,2:3].max() - cX0[:,2:3]),axis=1).view(','.join(3*['float'])),axis=0,order=('f2','f1','f0'))[:,0]].view(','.join(3*['float'])),axis=0,order=('f2','f1','f0'))[:,0]
            for item in six.iteritems(models):
                # Save the data
                np.savetxt(item[0],item[1][ubcReorder],fmt='%3.5e')

    @classmethod
    def readUBC(TreeMesh, meshFile):
        from io import StringIO
        """
            Read UBC 3D OcTree mesh and/or modelFiles

            Input:
            :param str meshFile: path to the UBC GIF OcTree mesh file to read
            :rtype: SimPEG.Mesh.TreeMesh
            :return: The octree mesh

        """

        # Read the file lines
        fileLines = np.genfromtxt(meshFile, dtype=str,
            delimiter='\n', comments='!')
        # Extract the data
        nCunderMesh = np.array(fileLines[0].
            split('!')[0].split(), dtype=float)
        # I think this is the case?
        # Format of file changed... First 3 values are the # of cells in the
        # underlying mesh and remaining 6 values are padding for the core region.
        nCunderMesh  = nCunderMesh[0:3]

        if np.unique(nCunderMesh).size >1:
            raise Exception('SimPEG TreeMeshes have the same number of cell in all directions')
        tswCorn = np.array(fileLines[1].
            split('!')[0].split(), dtype=float)
        smallCell = np.array(fileLines[2].
            split('!')[0].split(), dtype=float)
        nrCells = np.array(fileLines[3].
            split('!')[0].split(), dtype=float)
        # Read the index array
        indArr = np.genfromtxt((line.encode('utf8') for line in fileLines[4::]),dtype=np.int)

        ## Calculate simpeg parameters
        h1,h2,h3 = [np.ones(nr)*sz for nr,sz in zip(nCunderMesh,smallCell)]
        x0 = tswCorn - np.array([0,0,np.sum(h3)])
        # Need to convert the index array to a points list that complies with SimPEG TreeMesh.
        # Shift to start at 0
        simpegCellPt = indArr[:,0:-1].copy()
        simpegCellPt[:,2] = ( nCunderMesh[-1] + 2) - (simpegCellPt[:,2] + indArr[:,3])
        # Need reindex the z index to be from the bottom-left-close corner and to be from the global bottom.
        simpegCellPt = simpegCellPt - np.array([1.,1.,1.])

        # Calculate the cell level
        simpegLevel = np.log2(np.min(nCunderMesh)) - np.log2(indArr[:,3])
        # Make a pointer matrix
        simpegPointers = np.concatenate((simpegCellPt,simpegLevel.reshape((-1,1))),axis=1)

        ## Make the tree mesh
        mesh = TreeMesh([h1,h2,h3],x0)
        mesh._cells = set([mesh._index(p) for p in simpegPointers.tolist()])

        # Figure out the reordering
        mesh._simpegReorderUBC = np.argsort(np.array([mesh._index(i) for i in simpegPointers.tolist()]))
        # mesh._simpegReorderUBC = np.argsort((np.array([[1,1,1,-1]])*simpegPointers).view(','.join(4*['float'])),axis=0,order=['f3','f2','f1','f0'])[:,0]

        return mesh


    def readModelUBC(mesh, fileName):
        """
            Read UBC OcTree model and get vector

            :param string fileName: path to the UBC GIF model file to read
            :rtype: numpy.ndarray
            :return: OcTree model
        """

        if type(fileName) is list:
            out = {}
            for f in fileName:
                out[f] = mesh.readModelUBC(f)
            return out

        assert hasattr(mesh, '_simpegReorderUBC'), 'The file must have been loaded from a UBC format.'
        assert mesh.dim == 3

        modList = []
        modArr = np.loadtxt(fileName)
        if len(modArr.shape) == 1:
            modList.append(modArr[mesh._simpegReorderUBC])
        else:
            modList.append(modArr[mesh._simpegReorderUBC,:])
        return modList

    def writeVTK(mesh, fileName, models=None):
        """
        Function to write a VTU file from a SimPEG TreeMesh and model.
        """
        import vtk
        from vtk import vtkXMLUnstructuredGridWriter as Writer, VTK_VERSION
        from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray

        if str(type(mesh)).split()[-1][1:-2] not in 'SimPEG.Mesh.TreeMesh.TreeMesh':
            raise IOError('mesh is not a SimPEG TreeMesh.')

        # Make the data parts for the vtu object
        # Points
        mesh.number()
        ptsMat = mesh._gridN + mesh.x0

        vtkPts = vtk.vtkPoints()
        vtkPts.SetData(numpy_to_vtk(ptsMat,deep=True))
        # Cells
        cellConn = np.array([c.nodes for c in mesh],dtype=np.int64)

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
        if models is not None:
            for item in six.iteritems(models):
                # Convert numpy array
                vtkDoubleArr = numpy_to_vtk(item[1],deep=1)
                vtkDoubleArr.SetName(item[0])
                vtuObj.GetCellData().AddArray(vtkDoubleArr)

        # Make the writer
        vtuWriteFilter = Writer()
        if float(VTK_VERSION.split('.')[0]) >=6:
            vtuWriteFilter.SetInputData(vtuObj)
        else:
            vtuWriteFilter.SetInput(vtuObj)
        vtuWriteFilter.SetFileName(fileName)
        # Write the file
        vtuWriteFilter.Update()

