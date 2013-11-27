import numpy as np, matplotlib as mpl
try:
	import vtk, vtk.util.numpy_support as npsup
	#import SimPEG.visualize.vtk.vtkTools as vtkSP # Always get an error for this import
except Exception, e:
	print 'VTK import error. Please ensure you have VTK installed to use this visualization package.'
import SimPEG as simpeg

class vtkView(object):
	"""
	Class for storing and view of SimPEG models in VTK (visualization toolkit).

	Inputs:
	:param mesh, SimPEG mesh.
	:param propdict, dictionary of property models.
		Can have these dictionary names:
		'C' - cell model; 'F' - face model; 'E' - edge model; ('V' - vector field : NOT SUPPORTED)
		The dictionary values are given as dictionaries with:
		{'NameOfThePropertyModel': np.array of the properties}.
		The property np.array has to be ordered in compliance with SimPEG standards.

	::
	Example of usages.

		ToDo

	"""

	def __init__(self,mesh,propdict):
		"""
		"""

		# Setup hidden properties, used for the visualization
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
		# Set vtk object containers
		self._cells = None
		self._faces = None
		self._edges = None
		self._vectors = None # Not implemented
		# Set default values
		self.name = 'VTK figure of SimPEG model'
		


		# Error check the input mesh
		if type(mesh).__name__ != 'TensorMesh':
			raise Exception('The input {:s} to vtkView has to be a TensorMesh object'.format(mesh))
		# Set the mesh
		self._mesh = mesh

		# Read the property dictionary 
		self._readPropertyDictionary(propdict)
		

		
		
	# Set/Get properties
	@property 
	def cmap(self):
		''' Colormap to use in vtkView. Colormap is a matplotlib cmap(cm) array, has to be uint8(use flag bytes=True during cmap generation).'''
		if getattr(self,'_cmap',None) is None:
			# Set default
			self._cmap = mpl.cm.hsv(np.arange(0.,1.,0.05),bytes=True)
		return self._cmap
	@cmap.setter
	def cmap(self,value):
		if value.min() > 0 or value.max() < 255 or value.shape[1] != 4 or value.dtype != np.uint8:
			raise Exception('Input not an allowed array.\n Use matplotlib.cm to generate an array of size [nrColors,4] and dtype = uint8(flag bytes=True).')
		self._cmap = value
	
	@property 
	def range(self):
		''' Range of the colors in vtkView.'''
		if getattr(self,'_range',None) is None:
			self._range = np.array(self._getActiveVTKobj().GetArray(self.viewprop.values()[0]).GetRange())
		return self._range
	@range.setter
	def range(self,value):
		if type(value) not in [tuple, list, np.ndarray] or len(value) != 2 or np.array(value).dtype is not np.dtype('float'):
			raise Exception('Input not in correct format. \n Has to be a list, tuple or np.arry of 2 floats.')
		self._range = np.array(value)

	@property 
	def extent(self):
		''' Extent of the sub-domain of the model to view'''
		if getattr(self,'_extent',None) is None:
			self._extent = [0,self._mesh.nCx-1,0,self._mesh.nCy-1,0,self._mesh.nCz-1]
		return self._extent
	@extent.setter
	def extent(self,value):

		import warnings
		# Error check
		valnp = np.array(value,dtype=int)
		if valnp.dtype != int or len(valnp) != 6:
			raise Exception('.extent has to be list or nparray of 6 integers.')
		# Test the range of the values
		loB = np.zeros(3,dtype=int)
		upB = np.array(self._mesh.nCv - np.ones(3),dtype=int)
		# Test the bounds
		change = 0
		# Test for lower bounds, can't be smaller the 0
		tlb = valnp[::2] < loB
		if tlb.any(): 
			valnp[::2][tlb] = loB[tlb] 
			change = 1
			warnings.warn('Lower bounds smaller then 0')
		# Test for lower bounds, can't be larger then upB
		tlub = valnp[::2] > upB
		if tlub.any(): 
			valnp[::2][tlub] = upB[tlub] - 1  
			change = 1
			warnings.warn('Lower bounds larger then uppermost bounds')
		# Test for upper bounds, can't be larger the extent of the mesh
		tub = valnp[1::2] > upB
		if tub.any(): 
			valnp[1::2][tub] = upB[tub] 
			change = 1
			warnings.warn('Upper bounds greater then number of cells')
		# Test if lower is smaller the upper
		tgt = valnp[::2] > valnp[1::2]
		if tgt.any():
			valnp[1::2][tgt] = valnp[::2][tgt] + 1
			change = 1
			warnings.warn('Lower bounds greater the Upper bounds')
		# Print a warning
		if change:
			warnings.warn('Changed given extent from {:s} to {:s}'.format(value,valnp.tolist()))
		
		# Set extent
		self._extent = valnp

	@property
	def limits(self):
		''' Lower and upper limits (cutoffs) of the values to view. '''
		return getattr(self,'_limits',None)
	@limits.setter
	def limits(self,value):
		if value is None:
			self._limits = None
		else:
			valnp = np.array(value)
			if valnp.dtype != float or len(valnp) != 2:
				raise Exception('.limits has to be list or numpy array of 2 floats.')
			self._limits = valnp


	@property 
	def viewprop(self):
		''' Controls the property that will be viewed.'''

		if getattr(self,'_viewprop',None) is None:
			self._viewprop = {'C':0} # Name of the type and Int order of the array or name of the vector.
		return self._viewprop
	@viewprop.setter
	def viewprop(self,value):
		if type(value) != dict:
			raise Exception('{:s} has to be a python dictionary containing property type and name index. ')
		if len(value) > 1:
			raise Exception('Too many input items in the viewprop dictionary')
		if value.keys()[0] not in ['C','F','E']:
			raise Exception('\"{:s}\" is not allowed as a dictionary key. Can be \'C\',\'F\',\'E\'.'.format(propitem[0]))
		if not(type(self.viewprop.values()[0]) is int or type(self.viewprop.values()[0]) is str):
			raise Exception('The vtkView.viewprop.values()[0] has the wrong format. Has to be integer or a string with the index.')
		

		self._viewprop = value

	def _getActiveVTKobj(self):
		"""
		Finds the active VTK object.
		"""

		if self.viewprop.keys()[0] is 'C':
			vtkCellData = self._cells.GetCellData()
		elif self.viewprop.keys()[0] is 'F':
			vtkCellData = self._faces.GetCellData()
		elif self.viewprop.keys()[0] is 'E':
			vtkCellData = self._edges.GetCellData()

		return vtkCellData

	def _getActiveArrayName(self):
		"""
		Finds the name of the active array.
		"""
		actArr = self.viewprop.values()[0]
		if type(actArr) is str:
			activeName = actArr
		elif type(actArr) is int:
			activeName = self._getActiveVTKobj().GetArrayName(actArr)
		return activeName

	def _readPropertyDictionary(self,propdict):
		"""
		Reads the property and assigns to the object
		"""
		import SimPEG.visualize.vtk.vtkTools as vtkSP

		# Test the property dictionary
		if type(propdict) != dict:
			raise Exception('{:s} has to be a python dictionary containing property models. ')
		if len(propdict) > 4:
			raise Exception('Too many input items in the property dictionary')
		for propitem in propdict.iteritems():
			if propitem[0] in ['C','F','E']:
				if propitem[0] == 'C':
					self._cells = vtkSP.makeCellVTKObject(self._mesh,propitem[1])
				if propitem[0] == 'F':
					self._faces = vtkSP.makeFaceVTKObject(self._mesh,propitem[1])
				if propitem[0] == 'E':
					self._edges = vtkSP.makeEdgeVTKObject(self._mesh,propitem[1])
			else:
				raise Exception('\"{:s}\" is not allowed as a dictionary key. Can be \'C\',\'F\',\'E\'.'.format(propitem[0]))

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

		
		# Set the active scalar.
		if type(self.viewprop.values()[0]) == int:
			actScalar = self._getActiveVTKobj().GetArrayName(self.viewprop.values()[0])
		elif type(self.viewprop.values()[0]) == str:
			actScalar = self.viewprop.values()[0]
		else :
			raise Exception('The vtkView.viewprop.values()[0] has the wrong format. Has to be interger or a string.')
		self._getActiveVTKobj().SetActiveScalars(actScalar)
		# Sort out the actor
		imageType = self.viewprop.keys()[0]
		if imageType == 'C':
			if self.limits is None:
				self.limits = self._cells.GetCellData().GetArray(self.viewprop.values()[0]).GetRange()
			self._vtkobj, self._core = vtkSP.makeRectiVTKVOIThres(self._cells,self.extent,self.limits)
		elif imageType == 'F':
			if self.limits is None:
				self.limits = self._faces.GetCellData().GetArray(self.viewprop.values()[0]).GetRange()
			extent = [self._mesh.vectorNx[self.extent[0]], self._mesh.vectorNx[self.extent[1]], self._mesh.vectorNy[self.extent[2]], self._mesh.vectorNy[self.extent[3]], self._mesh.vectorNz[self.extent[4]], self._mesh.vectorNz[self.extent[5]] ]
			self._vtkobj, self._core = vtkSP.makeUnstructVTKVOIThres(self._faces,extent,self.limits)
		elif imageType == 'E':
			if self.limits is None:
				self.limits = self._edges.GetCellData().GetArray(self.viewprop.values()[0]).GetRange()
			extent = [self._mesh.vectorNx[self.extent[0]], self._mesh.vectorNx[self.extent[1]], self._mesh.vectorNy[self.extent[2]], self._mesh.vectorNy[self.extent[3]], self._mesh.vectorNz[self.extent[4]], self._mesh.vectorNz[self.extent[5]] ]
			self._vtkobj, self._core = vtkSP.makeUnstructVTKVOIThres(self._edges,extent,self.limits)
		else:
			raise Exception("{:s} is not a valid viewprop. Has to be 'C':'F':'E'".format(imageType))
		#self._vtkobj.GetCellData().SetActiveScalars(actScalar)
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
		lut.SetNumberOfColors(len(self.cmap))
		lut.SetTable(npsup.numpy_to_vtk(self.cmap))
		lut.Build()
		self._lut = lut
		scalarBar = vtk.vtkScalarBarActor()
  		scalarBar.SetLookupTable(lut)
 		scalarBar.SetTitle(self._getActiveArrayName())
  		scalarBar.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
  		scalarBar.GetPositionCoordinate().SetValue(0.1,0.01)
  		scalarBar.SetOrientationToHorizontal()
  		scalarBar.SetWidth(0.8)
  		scalarBar.SetHeight(0.17)

		self._actor.GetMapper().SetScalarRange(self.range)
		self._actor.GetMapper().SetLookupTable(lut)

		# Set renderer options
		self._ren.SetBackground(.5,.5,.5)
		self._ren.AddActor(self._actor)
		self._ren.AddActor2D(scalarBar)
  		self._renwin.SetSize(450,450)

		# Start the render Window
		vtkSP.startRenderWindow(self._iren)
		# Close the window when exited
		vtkSP.closeRenderWindow(self._iren)
		del self._iren, self._renwin



if __name__ == '__main__':
	

	#Make a mesh and model
	x0 = np.zeros(3)
	h1 = np.ones(60)*50
	h2 = np.ones(60)*100
	h3 = np.ones(50)*200

	mesh = simpeg.mesh.TensorMesh([h1,h2,h3],x0)

	# Make a models that correspond to the cells, faces and edges.
	t = np.ones(mesh.nC)
	t[10000:50000] = 100
	t[100000:120000] = 100
	t[100000:120000] = 50
	models = {'C':{'Test':np.arange(0,mesh.nC),'Model':t, 'AllOnce':np.ones(mesh.nC)},'F':{'Test':np.arange(0,mesh.nF),'AllOnce':np.ones(mesh.nF)},'E':{'Test':np.arange(0,mesh.nE),'AllOnce':np.ones(mesh.nE)}}
	# Make the vtk viewer object.
	vtkViewer = simpeg.visualize.vtk.vtkView(mesh,models)
	# Set the .viewprop for which model to view
	vtkViewer.viewprop = {'F':'Test'}
	# Show the image
	vtkViewer.Show()

	# Set subset of the mesh to view (remove padding)
	vtkViewer.extent = [4,14,0,7,0,3]
	vtkViewer.Show()

	# Change viewing property 
	vtkViewer.viewprop = {'C':'Model'}
	# Set the color range
	# Reset extent.
	vtkViewer.extent = [-1,1000,-1,1000,-1,1000]
	vtkViewer.range = [0.,100.]
	vtkViewer.Show()
	# Change color scale, has to be set to bytes=True.
	vtkViewer.cmap = mpl.cm.copper(np.arange(0.,1.,0.01),bytes=True)
	vtkViewer.Show()
	# Set limits of values to view 
	vtkViewer.limits = [5.0,100.0]
	vtkViewer.Show()