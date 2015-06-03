from simpegEM.FDEM import BaseFDEMProblem
from SurveyMT import SurveyMT
from DataMT import DataMT
from FieldsMT import FieldsMT
from SimPEG import SolverLU as SimpegSolver

class BaseMTProblem(BaseFDEMProblem):

    def __init__(self, mesh, **kwargs):
        BaseFDEMProblem.__init__(self, mesh, **kwargs)

    # Set the default pairs of the problem
    surveyPair = SurveyMT
    dataPair = DataMT
    fieldsPair = FieldsMT

    # Set the solver
    Solver = SimpegSolver
    solverOpts = {}

    verbose = False
    # Notes:
    # Use the forward and devs from BaseFDEMProblem
    # Might need to add more stuff here.

