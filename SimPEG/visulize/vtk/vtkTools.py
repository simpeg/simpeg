import numpy as np, vtk, vtk.util.numpy_support as npsup, pdb
from SimPEG.utils import mkvc


class vtkTools(object):
	""" 
	Class that interacts with VTK visulization toolkit.

	"""

	def __init__(self):
		""" Initializes the VTK vtkTools.
		
		"""

		pass

	@staticmethod
	def makeCellVTKObject(mesh,model):
		"""
		Make and return a cell based VTK object for a simpeg mesh and model.

		Input: 
		:param mesh, SimPEG TensorMesh object - mesh to be transfer to VTK
		:param model, dictionary of numpy.array - Name('s) and array('s). Match number of cells

		Output:
        :rtype: vtkRecilinearGrid object
        :return: vtkObj
		"""

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
		# Asaign the spatial information.
		vtkObj = vtk.vtkRectilinearGrid()
		vtkObj.SetDimensions(xD,yD,zD)
		vtkObj.SetXCoordinates(npsup.numpy_to_vtk(vX,deep=1))
		vtkObj.SetYCoordinates(npsup.numpy_to_vtk(vY,deep=1))
		vtkObj.SetZCoordinates(npsup.numpy_to_vtk(vZ,deep=1))

		# Assign the model('s) to the object
		for item in model.iteritems():
			# Convert numpy array
			vtkDoubleArr = npsup.numpy_to_vtk(item[1],deep=1)
			vtkDoubleArr.SetName(item[0])
			vtkObj.GetCellData().AddArray(vtkDoubleArr)

		return vtkObj

	@staticmethod
	def makeFaceVTKObject(mesh,model):
		"""
		Make and return a face based VTK object for a simpeg mesh and model.

		Input: 
		:param mesh, SimPEG TensorMesh object - mesh to be transfer to VTK
		:param model, dictionary of numpy.array - Name('s) and array('s). 
			Property array must be order hstack(Fx,Fy,Fz)

		Output:
        :rtype: vtkUnstructuredGrid object
        :return: vtkObj
		"""

		## Convert simpeg mesh to VTK properties
		# Convert mesh nodes to vtkPoints
		vtkPts = vtk.vtkPoints()
		vtkPts.SetData(npsup.numpy_to_vtk(mesh.gridN,deep=1))
		
		# Define the face "cells"
		# Using VTK_QUAD cell for faces (see VTK file format)
		nodeMat = mesh.r(np.arange(mesh.nN,dtype='int64'),'N','N','M')
		def faceR(mat,length):
			return mat.T.reshape((length,1))
		# First direction
		nTFx = np.prod(mesh.nFx)
		FxCellBlock = np.hstack([ 4*np.ones((nTFx,1),dtype='int64'),faceR(nodeMat[:,:-1,:-1],nTFx),faceR(nodeMat[:,1: ,:-1],nTFx),faceR(nodeMat[:,1: ,1: ],nTFx),faceR(nodeMat[:,:-1,1: ],nTFx)] )
		FyCellBlock = np.array([],dtype='int64')
		FzCellBlock = np.array([],dtype='int64')
		# Second direction
		if mesh.dim >= 2:
			nTFy = np.prod(mesh.nFy)
			FyCellBlock = np.hstack([ 4*np.ones((nTFy,1),dtype='int64'),faceR(nodeMat[:-1,:,:-1],nTFy),faceR(nodeMat[1: ,:,:-1],nTFy),faceR(nodeMat[1: ,:,1: ],nTFy),faceR(nodeMat[:-1,:,1: ],nTFy)] )
		# Third direction
		if mesh.dim == 3:
			nTFz = np.prod(mesh.nFz)
			FzCellBlock = np.hstack([ 4*np.ones((nTFz,1),dtype='int64'),faceR(nodeMat[:-1,:-1,:],nTFz),faceR(nodeMat[1: ,:-1,:],nTFz),faceR(nodeMat[1: ,1: ,:],nTFz),faceR(nodeMat[:-1,1: ,:],nTFz)] )	
		# Cells -cell array
		FCellArr = vtk.vtkCellArray()
		FCellArr.SetNumberOfCells(np.sum(mesh.nF))
		FCellArr.SetCells(np.sum(mesh.nF)*5,npsup.numpy_to_vtkIdTypeArray(np.vstack([FxCellBlock,FyCellBlock,FzCellBlock]),deep=1))
		# Cell type
		FCellType = npsup.numpy_to_vtk(vtk.VTK_QUAD*np.ones(np.sum(mesh.nF),dtype='uint8'),deep=1)
		# Cell location
		FCellLoc = npsup.numpy_to_vtkIdTypeArray(np.arange(0,np.sum(mesh.nF)*5,5,dtype='int64'),deep=1)
		
		## Make the object
		vtkObj = vtk.vtkUnstructuredGrid()
		# Set the objects properties
		vtkObj.SetPoints(vtkPts)
		vtkObj.SetCells(FCellType,FCellLoc,FCellArr)		

		# Assign the model('s) to the object
		for item in model.iteritems():
			# Convert numpy array
			vtkDoubleArr = npsup.numpy_to_vtk(item[1],deep=1)
			vtkDoubleArr.SetName(item[0])
			vtkObj.GetCellData().AddArray(vtkDoubleArr)
		vtkObj.Update()
		return vtkObj

	@staticmethod
	def makeEdgeVTKObject(mesh,model):
		"""
		Make and return a edge based VTK object for a simpeg mesh and model.

		Input: 
		:param mesh, SimPEG TensorMesh object - mesh to be transfer to VTK
		:param model, dictionary of numpy.array - Name('s) and array('s). 
			Property array must be order hstack(Ex,Ey,Ez)

		Output:
        :rtype: vtkUnstructuredGrid object
        :return: vtkObj
		"""

		## Convert simpeg mesh to VTK properties
		# Convert mesh nodes to vtkPoints
		vtkPts = vtk.vtkPoints()
		vtkPts.SetData(npsup.numpy_to_vtk(mesh.gridN,deep=1))
		
		# Define the face "cells"
		# Using VTK_QUAD cell for faces (see VTK file format)
		nodeMat = mesh.r(np.arange(mesh.nN,dtype='int64'),'N','N','M')
		def edgeR(mat,length):
			return mat.T.reshape((length,1))
		# First direction
		nTEx = np.prod(mesh.nEx)
		ExCellBlock = np.hstack([ 2*np.ones((nTEx,1),dtype='int64'),edgeR(nodeMat[:-1,:,:],nTEx),edgeR(nodeMat[1:,:,:],nTEx)])
		# Second direction
		if mesh.dim >= 2:
			nTEy = np.prod(mesh.nEy)
			EyCellBlock = np.hstack([ 2*np.ones((nTEy,1),dtype='int64'),edgeR(nodeMat[:,:-1,:],nTEy),edgeR(nodeMat[:,1:,:],nTEy)])
		# Third direction
		if mesh.dim == 3:
			nTEz = np.prod(mesh.nEz)
			EzCellBlock = np.hstack([ 2*np.ones((nTEz,1),dtype='int64'),edgeR(nodeMat[:,:,:-1],nTEz),edgeR(nodeMat[:,:,1:],nTEz)])	
		# Cells -cell array
		ECellArr = vtk.vtkCellArray()
		ECellArr.SetNumberOfCells(np.sum(mesh.nE))
		ECellArr.SetCells(np.sum(mesh.nE)*3,npsup.numpy_to_vtkIdTypeArray(np.vstack([ExCellBlock,EyCellBlock,EzCellBlock]),deep=1))
		# Cell type
		ECellType = npsup.numpy_to_vtk(vtk.VTK_LINE*np.ones(np.sum(mesh.nE),dtype='uint8'),deep=1)
		# Cell location
		ECellLoc = npsup.numpy_to_vtkIdTypeArray(np.arange(0,np.sum(mesh.nE)*3,3,dtype='int64'),deep=1)

		## Make the object
		vtkObj = vtk.vtkUnstructuredGrid()
		# Set the objects properties
		vtkObj.SetPoints(vtkPts)
		vtkObj.SetCells(ECellType,ECellLoc,ECellArr)		

		# Assign the model('s) to the object
		for item in model.iteritems():
			# Convert numpy array
			vtkDoubleArr = npsup.numpy_to_vtk(item[1],deep=1)
			vtkDoubleArr.SetName(item[0])
			vtkObj.GetCellData().AddArray(vtkDoubleArr)

		return vtkObj

	@staticmethod
	def makeRenderWindow(ren):
		renWin = vtk.vtkRenderWindow()
		renWin.AddRenderer(ren)
		iren = vtk.vtkRenderWindowInteractor()
		iren.SetRenderWindow(renWin)

		return iren


	@staticmethod
	def closeRenderWindow(iren):
		renwin = iren.GetRenderWindow()
		renwin.Finalize()
		iren.TerminateApp()
		del iren, renwin

	@staticmethod
	def makeVTKActor(vtkObj):
		""" Makes a vtk mapper and Actor"""
		mapper = vtk.vtkDataSetMapper()
		mapper.SetInput(vtkObj)
		actor = vtk.vtkActor()
		actor.SetMapper(mapper)
		actor.GetProperty().SetColor(0,0,0)
		actor.GetProperty().SetRepresentationToWireframe()

		return actor

	@staticmethod
	def startRenderWindow(iren):
		""" Start a vtk rendering window"""
		iren.Initialize()
		renwin = iren.GetRenderWindow()
		renwin.Render()
		iren.Start()


	# Simple write/read VTK xml model functions.
	@staticmethod
	def writeVTPFile(fileName,vtkPolyObject):
	    '''Function to write vtk polydata file (vtp).'''
	    polyWriter = vtk.vtkXMLPolyDataWriter()
	    polyWriter.SetInput(vtkPolyObject)
	    polyWriter.SetFileName(fileName)
	    polyWriter.Update()

	@staticmethod
	def writeVTUFile(fileName,vtkUnstructuredGrid):
	    '''Function to write vtk unstructured grid (vtu).'''
	    Writer = vtk.vtkXMLUnstructuredGridWriter()
	    Writer.SetInput(vtkUnstructuredGrid)
	    Writer.SetFileName(fileName)
	    Writer.Update()

	@staticmethod
	def writeVTRFile(fileName,vtkRectilinearGrid):
	    '''Function to write vtk rectilinear grid (vtr).'''
	    Writer = vtk.vtkXMLRectilinearGridWriter()
	    Writer.SetInput(vtkRectilinearGrid)
	    Writer.SetFileName(fileName)
	    Writer.Update()

	@staticmethod
	def writeVTSFile(fileName,vtkStructuredGrid):
	    '''Function to write vtk structured grid (vts).'''
	    Writer = vtk.vtkXMLStructuredGridWriter()
	    Writer.SetInput(vtkStructuredGrid)
	    Writer.SetFileName(fileName)
	    Writer.Update()

	@staticmethod
	def readVTSFile(fileName):
	    '''Function to read vtk structured grid (vts) and return a grid object.'''
	    Reader = vtk.vtkXMLStructuredGridReader()
	    Reader.SetFileName(fileName)
	    Reader.Update()
	    return Reader.GetOutput()

	@staticmethod
	def readVTUFile(fileName):
	    '''Function to read vtk structured grid (vtu) and return a grid object.'''
	    Reader = vtk.vtkXMLUnstructuredGridReader()
	    Reader.SetFileName(fileName)
	    Reader.Update()
	    return Reader.GetOutput()

	@staticmethod
	def readVTRFile(fileName):
	    '''Function to read vtk structured grid (vtr) and return a grid object.'''
	    Reader = vtk.vtkXMLRectilinearGridReader()
	    Reader.SetFileName(fileName)
	    Reader.Update()
	    return Reader.GetOutput()

	@staticmethod
	def readVTPFile(fileName):
	    '''Function to read vtk structured grid (vtp) and return a grid object.'''
	    Reader = vtk.vtkXMLPolyDataReader()
	    Reader.SetFileName(fileName)
	    Reader.Update()
	    return Reader.GetOutput()

