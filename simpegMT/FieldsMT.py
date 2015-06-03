from SimPEG import Survey, Utils, Problem, np, sp, mkvc
from scipy.constants import mu_0
import sys
from numpy.lib import recfunctions as recFunc

##############
### Fields ###
##############
class FieldsMT(Problem.Fields):
    """Fancy Field Storage for a MT survey."""
    knownFields = {'b_px': 'F','b_py': 'F', 'e_px': 'E','e_py': 'E','b_1d':'E','e_1d':'F'}
    dtype = complex
