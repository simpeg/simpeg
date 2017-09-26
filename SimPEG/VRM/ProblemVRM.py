from SimPEG import Problem
from SimPEG.VRM.SurveyVRM import SurveyVRM
# from SimPEG.VRM.FieldsVRM import Fields_LinearFWD
import numpy as np







############################################
# BASE VRM PROBLEM CLASS
############################################

class BaseProblemVRM(Problem.BaseProblem):

	def __init__(self, mesh, **kwargs):

		assert len(mesh.h) == 3, 'Problem requires 3D tensor or OcTree mesh'

		refFact = 3
		refRadius = 1.25*np.mean(np.r_[np.min(mesh.h[0]),np.min(mesh.h[1]),np.min(mesh.h[2])])*np.r_[1.,2.,3.]

		assert len(refRadius) == refFact, 'Number of refinement radii must equal refinement factor'

		super(BaseProblemVRM,self).__init__(mesh, **kwargs)
		self.surveyPair = SurveyVRM
		self.refFact = refFact
		self.refRadius = refRadius
		self.A = None
		self.Tb = None
		self.Tdbdt = None

	# def getH0(self, mesh, topoInd, pp):

	# 	# INPUTS
	# 	# self: Problem instance
	# 	# mesh: 3D mesh
	# 	# topoInd: True/False for cells that will be modeled (so air cells aren't modeled)
	# 	# pp: Source index

	# 	xyz = meshObj.gridCC[topoInd,:]
	# 	hx = meshObj.hx[topoInd]
	# 	hy = meshObj.hy[topoInd]
	# 	hz = meshObj.hz[topoInd]

	# 	xn = xyz[:,0] - hx/2
	# 	yn = xyz[:,1] - hy/2
	# 	zn = xyz[:,2] - hz/2





#######################################################################################
# VRM CHARACTERISTIC DECAY FORMULATION (SINGLE MODEL PARAMETER AND ALLOWS INVERSION)
#######################################################################################


class LinearFWD(BaseProblemVRM):

	def __init__(self, mesh, **kwargs):
		super(LinearFWD,self).__init__(mesh, **kwargs)



	def fields(self, mod, **kwargs):

		topoInd = mod != 0 # Only predict data from non-zero model values

		
		# Get A matrix
		A = np.array([])
		
		srcList = self.survey.srcList




		# Get T matrix






























