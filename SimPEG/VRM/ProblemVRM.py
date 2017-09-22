from SimPEG import Problem
from SimPEG.VRM.SurveyVRM import SurveyVRM
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




#######################################################################################
# VRM CHARACTERISTIC DECAY FORMULATION (SINGLE MODEL PARAMETER AND ALLOWS INVERSION)
#######################################################################################


class ProblemLinear(BaseProblemVRM):

	def __init__(self, mesh, **kwargs):
		super(ProblemLinear,self).__init__(mesh, **kwargs)
































