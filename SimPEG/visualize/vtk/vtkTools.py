import numpy as np
try:
	import vtk, vtk.util.numpy_support as npsup, pdb
except Exception, e:
	print 'VTK import error. Please ensure you have VTK installed to use this visualization package.'
from SimPEG.Utils import mkvc


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
		# Assign the spatial information.
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

		vtkObj.GetCellData().SetActiveScalars(model.keys()[0])
		vtkObj.Update()
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
		FCellArr.SetNumberOfCells(mesh.nF)
		FCellArr.SetCells(mesh.nF,npsup.numpy_to_vtkIdTypeArray(np.vstack([FxCellBlock,FyCellBlock,FzCellBlock]),deep=1))
		# Cell type
		FCellType = npsup.numpy_to_vtk(vtk.VTK_QUAD*np.ones(mesh.nF,dtype='uint8'),deep=1)
		# Cell location
		FCellLoc = npsup.numpy_to_vtkIdTypeArray(np.arange(0,mesh.nF*5,5,dtype='int64'),deep=1)

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

		vtkObj.GetCellData().SetActiveScalars(model.keys()[0])
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
		ECellArr.SetNumberOfCells(mesh.nE)
		ECellArr.SetCells(mesh.nE,npsup.numpy_to_vtkIdTypeArray(np.vstack([ExCellBlock,EyCellBlock,EzCellBlock]),deep=1))
		# Cell type
		ECellType = npsup.numpy_to_vtk(vtk.VTK_LINE*np.ones(mesh.nE,dtype='uint8'),deep=1)
		# Cell location
		ECellLoc = npsup.numpy_to_vtkIdTypeArray(np.arange(0,mesh.nE*3,3,dtype='int64'),deep=1)

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

		vtkObj.GetCellData().SetActiveScalars(model.keys()[0])
		vtkObj.Update()
		return vtkObj

	@staticmethod
	def makeRenderWindow(ren):
		renwin = vtk.vtkRenderWindow()
		renwin.AddRenderer(ren)
		iren = vtk.vtkRenderWindowInteractor()
		iren.GetInteractorStyle().SetCurrentStyleToTrackballCamera()
		iren.SetRenderWindow(renwin)

		return iren, renwin


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
	def makeVTKLODActor(vtkObj,clipper):
		"""Make LOD vtk Actor"""
		selectMapper = vtk.vtkDataSetMapper()
		selectMapper.SetInputConnection(clipper.GetOutputPort())
		selectMapper.SetScalarVisibility(1)
		selectMapper.SetColorModeToMapScalars()
		selectMapper.SetScalarModeToUseCellData()
		selectMapper.SetScalarRange(clipper.GetInputDataObject(0,0).GetCellData().GetArray(0).GetRange())

		selectActor = vtk.vtkLODActor()
		selectActor.SetMapper(selectMapper)
		selectActor.GetProperty().SetEdgeColor(1,0.5,0)
		selectActor.GetProperty().SetEdgeVisibility(0)
		selectActor.VisibilityOn()
		selectActor.SetScale(1.01, 1.01, 1.01)
		return selectActor

	@staticmethod
	def setScalar2View(vtkObj,scalarName):
		""" Sets the sclar to view """
		useArr = vtkObj.GetCellData().GetArray(scalarName)
		if useArr == None:
			raise IOError('Nerty array {:s} in the vtkObject'.format(scalarName))
		vtkObj.GetCellData().SetActiveScalars(scalarName)

	@staticmethod
	def makeRectiVTKVOIThres(vtkObj,VOI,limits):
		"""Make volume of interest and threshold for rectilinear grid."""
		# Check for the input
		cellCore = vtk.vtkExtractRectilinearGrid()
		cellCore.SetVOI(VOI)
		cellCore.SetInput(vtkObj)

		cellThres = vtk.vtkThreshold()
		cellThres.AllScalarsOn()
		cellThres.SetInputConnection(cellCore.GetOutputPort())
		cellThres.ThresholdBetween(limits[0],limits[1])
		cellThres.Update()
		return cellThres.GetOutput(), cellCore.GetOutput()

	@staticmethod
	def makeUnstructVTKVOIThres(vtkObj,extent,limits):
		"""Make volume of interest and threshold for rectilinear grid."""
		# Check for the input
		cellCore = vtk.vtkExtractUnstructuredGrid()
		cellCore.SetExtent(extent)
		cellCore.SetInput(vtkObj)

		cellThres = vtk.vtkThreshold()
		cellThres.AllScalarsOn()
		cellThres.SetInputConnection(cellCore.GetOutputPort())
		cellThres.ThresholdBetween(limits[0],limits[1])
		cellThres.Update()
		return cellThres.GetOutput(), cellCore.GetOutput()

	@staticmethod
	def makePlaneClipper(vtkObj):
		"""Makes a plane and clipper """
		plane = vtk.vtkPlane()
		clipper = vtk.vtkClipDataSet()
		clipper.SetInputConnection(vtkObj.GetProducerPort())
		clipper.SetClipFunction(plane)
		clipper.InsideOutOff()
		return clipper, plane

	@staticmethod
	def makePlaneWidget(vtkObj,iren,plane,actor):
		"""Make an interactive planeWidget"""

		# Callback function
		def movePlane(obj, events):
		    obj.GetPlane(intPlane)
		    intActor.VisibilityOn()

		# Associate the line widget with the interactor
		planeWidget = vtk.vtkImplicitPlaneWidget()
		planeWidget.SetInteractor(iren)
		planeWidget.SetPlaceFactor(1.25)
		planeWidget.SetInput(vtkObj)
		planeWidget.PlaceWidget()
		#planeWidget.AddObserver("InteractionEvent", movePlane)
		planeWidget.SetScaleEnabled(0)
		planeWidget.SetEnabled(1)
		planeWidget.SetOutlineTranslation(0)
		planeWidget.GetPlaneProperty().SetOpacity(0.1)
		return planeWidget


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

