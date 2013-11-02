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

		# Use rectilinear VTK grid.
		# Asign the spatial information.
		vtkObj = vtk.vtkRectilinearGrid()
		vtkObj.SetDimensions(mesh.nNx,mesh.nNy,mesh.nNz)
		vtkObj.SetXCoordinates(npsup.numpy_to_vtk(mesh.vectorNx,deep=1))
		vtkObj.SetYCoordinates(npsup.numpy_to_vtk(mesh.vectorNy,deep=1))
		vtkObj.SetZCoordinates(npsup.numpy_to_vtk(mesh.vectorNz,deep=1))

		# Assign the model('s) to the object
		for item in model.iteritems():
			# Convert numpy array
			vtkDoubleArr = npsup.numpy_to_vtk(item[1],deep=1)
			vtkDoubleArr.SetName(item[0])
			vtkObj.GetCellData().AddArray(vtkDoubleArr)

		return vtkObj

	@staticmethod
	def makeFaceVTKobject(mesh,model):
		"""
		Make and return a cell based VTK object for a simpeg mesh and model.

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
		# Second direction
		nTFy = np.prod(mesh.nFy)
		FyCellBlock = np.hstack([ 4*np.ones((nTFy,1),dtype='int64'),faceR(nodeMat[:-1,:,:-1],nTFy),faceR(nodeMat[1: ,:,:-1],nTFy),faceR(nodeMat[1: ,:,1: ],nTFy),faceR(nodeMat[:-1,:,1: ],nTFy)] )
		# Third direction
		nTFz = np.prod(mesh.nFz)
		FzCellBlock = np.hstack([ 4*np.ones((nTFz,1),dtype='int64'),faceR(nodeMat[:-1,:-1,:],nTFz),faceR(nodeMat[1: ,:-1,:],nTFz),faceR(nodeMat[1: ,1: ,:],nTFz),faceR(nodeMat[:-1,1: ,:],nTFz)] )	
		# Cells -cell array
		FCellArr = vtk.vtkCellArray()
		FCellArr.SetCells(np.prod(mesh.nF),npsup.numpy_to_vtkIdTypeArray(np.vstack([FxCellBlock,FyCellBlock,FzCellBlock]),deep=1))
		# Cell type
		FCellType = npsup.numpy_to_vtk(vtk.VTK_QUAD*np.ones(np.sum(mesh.nF),dtype='uint8'),deep=1)
		# Cell location
		FCellLoc = npsup.numpy_to_vtkIdTypeArray(np.arange(0,np.sum(mesh.nF),5,dtype='int64'),deep=1)

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

		return vtkObj


	# Simple write model functions.
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

