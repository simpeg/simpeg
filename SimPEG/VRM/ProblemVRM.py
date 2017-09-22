from SimPEG import Problem
from SimPEG.VRM.SurveyVRM import SurveyVRM







############################################
# BASE VRM PROBLEM CLASS
############################################

class BaseProblemVRM(Problem.BaseProblem):

	def __init__(self, mesh, **kwargs):

		refFact = 3



		super(ProblemVRM,self).__init__(mesh, **kwargs)
		self.surveyPair = SurveyVRM




#######################################################################################
# VRM CHARACTERISTIC DECAY FORMULATION (SINGLE MODEL PARAMETER AND ALLOWS INVERSION)
#######################################################################################





