import numpy as np, vtk
import SimPEG as simpeg
#import SimPEG.visulize.vtk.vtkTools as vtkSP # Always get an error for this import

class vtkView(object):
	"""
	Class for storing and view of SimPEG models in VTK (visulization toolkit).

	Inputs:
	:param mesh, SimPEG mesh.
	:param propdict, dictionary of property models. 
		Can have these dictionary names:
		'cell' - cell model; 'face' - face model; 'edge' - edge model
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

		self.name = 'VTK figure of SimPEG model'
		self._mesh = mesh
		self._cell = None
		self._faces = None
		self._edges = None



		self._readPropertyDictionary(propdict)

	def _readPropertyDictionary(self,propdict):
		""" 
		Reads the property and assigns to the object
		"""
		import SimPEG.visulize.vtk.vtkTools as vtkSP

		# Test the property dictionary
		if len(propdict) > 3:
			raise(Exception,'Too many input items in the property dictionary')
		for propitem in propdict.iteritems():
			if propitem[0] in ['cell','face','edge']:
				if propitem[0] == 'cell':
					self._cell = vtkSP.makeCellVTKObject(self._mesh,propitem[1])
				if propitem[0] == 'face':
					self._face = vtkSP.makeFaceVTKObject(self._mesh,propitem[1])
				if propitem[0] == 'edge':
					self._edge = vtkSP.makeEdgeVTKObject(self._mesh,propitem[1])
			else:
				raise(Exception,'{:s} is not allowed as a dictonary key. Can be \'cell\',\'face\',\'edge\'.'.format(propitem[0]))

	def Show(self,imageType='cell'):
		"""
		Open the VTK figure window
		"""
		#vtkSP = simpeg.visulize.vtk.vtkTools
		import SimPEG.visulize.vtk.vtkTools as vtkSP

		# Make a renderer
		ren = vtk.vtkRenderer()
		# Make renderwindow. Returns the interactor.
		iren = vtkSP.makeRenderWindow(ren)

		# Sort out the actor
		if imageType == 'cell':
			actor = vtkSP.makeVTKActor(self._cell)
		elif imageType == 'face':
			actor = vtkSP.makeVTKActor(self._face)
		elif imageType == 'edge':
			actor = vtkSP.makeVTKActor(self._edge)
			actor.GetProperty().SetRepresentationToSurface()
		else:
			raise(Exception,"{:s} is not a vailid imageType. Has to be 'cell':'face':'edge'".format(imageType))

		ren.AddActor(actor)
		ren.SetBackground(.5,.5,.5)

		vtkSP.startRenderWindow(iren)

		vtkSP.closeRenderWindow(iren)








