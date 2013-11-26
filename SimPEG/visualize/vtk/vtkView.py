import numpy as np
try:
	import vtk
	#import SimPEG.visualize.vtk.vtkTools as vtkSP # Always get an error for this import
except Exception, e:
	print 'VTK import error. Please ensure you have VTK installed to use this visualization package.'
import SimPEG as simpeg

class vtkView(object):
	"""
	Class for storing and view of SimPEG models in VTK (visulization toolkit).

	Inputs:
	:param mesh, SimPEG mesh.
	:param propdict, dictionary of property models.
		Can have these dictionary names:
		'C' - cell model; 'F' - face model; 'E' - edge model
		The dictionary properties are given as dictionaries with:
		{'NameOfThePropertyModel': np.array of the properties}.
		The property array has to be ordered in compliance with SimPEG standards.

	::
	Example of usages.

		ToDo

	"""

	def __init__(self,mesh,propdict):
		"""
		"""

		# Error check inputs
		if type(mesh).__name__ != 'TensorMesh':
			raise Exception('The input {:s} to vtkView has to be a TensorMesh object'.format(mesh))


		# Set default values
		self.name = 'VTK figure of SimPEG model'
		self._extent = [0,mesh.nCx-1,0,mesh.nCy-1,0,mesh.nCz-1]
		self._limits = [0, 1e12]
		self._viewprop = {'C':0} # Name of the tyep and Int order of the array or name of the vector.
		self._mesh = mesh


		# Set vtk object containers
		self._cell = None
		self._faces = None
		self._edges = None

		self._readPropertyDictionary(propdict)

		# Setup hidden properties
		self._ren = None
		self._iren = None
		self._renwin = None
		self._core = None
		self._viewobj = None
		self._plane = None
		self._clipper = None
		self._widget = None
		self._actor = None
		self._lut = None

	# Set/Get properties
	@property 
	def extent(self):
		return getattr(self,_extent)

	@extent.setter
	def extent(self,value):

		# Error check
		valnp = np.array(value)
		if valnp.dtype != int or len(valnp) != 6:
			raise Exception('.extent has to be list or nparray of 6 integers.')

		self._extent = valnp

	@property
	def limits(self):
		return getattr(self._limits)

	@limits.setter
	def limits(self,value):
		valnp = np.array(value)
		if valnp.dtype != float or len(valnp) != 2:
			raise Exception('.limits has to be list or nparray of 2 floats.')

	def _readPropertyDictionary(self,propdict):
		"""
		Reads the property and assigns to the object
		"""
		import SimPEG.visualize.vtk.vtkTools as vtkSP

		# Test the property dictionary
		if len(propdict) > 3:
			raise(Exception,'Too many input items in the property dictionary')
		for propitem in propdict.iteritems():
			if propitem[0] in ['C','F','E']:
				if propitem[0] == 'C':
					self._cell = vtkSP.makeCellVTKObject(self._mesh,propitem[1])
				if propitem[0] == 'F':
					self._face = vtkSP.makeFaceVTKObject(self._mesh,propitem[1])
				if propitem[0] == 'E':
					self._edge = vtkSP.makeEdgeVTKObject(self._mesh,propitem[1])
			else:
				raise(Exception,'{:s} is not allowed as a dictonary key. Can be \'C\',\'F\',\'E\'.'.format(propitem[0]))

	def Show(self):
		"""
		Open the VTK figure window and show the mesh.
		"""
		#vtkSP = simpeg.visualize.vtk.vtkTools
		import SimPEG.visualize.vtk.vtkTools as vtkSP

		# Make a renderer
		self._ren = vtk.vtkRenderer()
		# Make renderwindow. Returns the interactor.
		self._iren, self._renwin = vtkSP.makeRenderWindow(self._ren)

		imageType = self.viewprop.keys()[0]
		# Sort out the actor
		if imageType == 'C':
			self._vtkobj, self._core = vtkSP.makeRectiVTKVOIThres(self._cell,self.extent,self.limits)
		elif imageType == 'F':
			extent = [self._mesh.vectorNx[self.extent[0]], self._mesh.vectorNx[self.extent[1]], self._mesh.vectorNy[self.extent[2]], self._mesh.vectorNy[self.extent[3]], self._mesh.vectorNz[self.extent[4]], self._mesh.vectorNz[self.extent[5]] ]
			self._vtkobj, self._core = vtkSP.makeUnstructVTKVOIThres(self._face,extent,self.limits)
		elif imageType == 'E':
			extent = [self._mesh.vectorNx[self.extent[0]], self._mesh.vectorNx[self.extent[1]], self._mesh.vectorNy[self.extent[2]], self._mesh.vectorNy[self.extent[3]], self._mesh.vectorNz[self.extent[4]], self._mesh.vectorNz[self.extent[5]] ]
			self._vtkobj, self._core = vtkSP.makeUnstructVTKVOIThres(self._edge,extent,self.limits)
		else:
			raise Exception("{:s} is not a valid viewprop. Has to be 'C':'F':'E'".format(imageType))

		# Set the active scalar.
		if type(self.viewprop.values()[0]) == int:
			actScalar = self._vtkobj.GetCellData().GetArrayName(self.viewprop.values()[0])
		elif type(self.viewprop.values()[0]) == str:
			actScalar = self.viewprop.values()[0]
		else :
			raise Exception('The vtkView.viewprop.values()[0] has the wrong format. Has to be interger or a string.')
		self._vtkobj.GetCellData().SetActiveScalars(actScalar)
		# Set up the plane, clipper and the user interaction.
		global intPlane, intActor
		self._clipper, intPlane = vtkSP.makePlaneClipper(self._vtkobj)
		intActor = vtkSP.makeVTKLODActor(self._vtkobj,self._clipper)
		self._widget = vtkSP.makePlaneWidget(self._vtkobj,self._iren,self._clipper.GetClipFunction(),self._actor)
		# Callback function
		self._plane = intPlane
		self._actor = intActor
		def movePlane(obj, events):
			global intPlane, intActor
			obj.GetPlane(intPlane)
			intActor.VisibilityOn()

		self._widget.AddObserver("InteractionEvent",movePlane)
		lut = vtk.vtkLookupTable()
		lut.SetNumberOfColors(256)
		lut.SetHueRange(0,0.66667)
		lut.Build()
		self._lut = lut
		self._actor.GetMapper().SetLookupTable(lut)

		# Set renderer options
		self._ren.SetBackground(.5,.5,.5)
		self._ren.AddActor(self._actor)

		# Start the render Window
		vtkSP.startRenderWindow(self._iren)
		# Close the window when exited
		vtkSP.closeRenderWindow(self._iren)
		del self._iren, self._renwin



if __name__ == '__main__':
	#Make a mesh and model
	x0 = np.zeros(3)
	h1 = np.ones(20)*50
	h2 = np.ones(10)*100
	h3 = np.ones(5)*200

	mesh = simpeg.mesh.TensorMesh([h1,h2,h3],x0)

	# Make a models that correspond to the cells, faces and edges.
	models = {'C':{'Test':np.arange(0,mesh.nC),'AllOnce':np.ones(mesh.nC)},'F':{'Test':np.arange(0,np.sum(mesh.nF)),'AllOnce':np.ones(np.sum(mesh.nF))},'E':{'Test':np.arange(0,np.sum(mesh.nE)),'AllOnce':np.ones(np.sum(mesh.nE))}}
	# Make the vtk viewer object.
	vtkViewer = simpeg.visualize.vtk.vtkView(mesh,models)
	# Show the image
	vtkViewer.Show()
