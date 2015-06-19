from SimPEG import Survey, Utils, Problem, np, sp, mkvc
from scipy.constants import mu_0
import sys
from numpy.lib import recfunctions as recFunc
from simpegEM.Utils.EMUtils import omega

##############
### Fields ###
##############
class FieldsMT(Problem.Fields):
    """Field Storage for a MT survey."""
    knownFields = {'b_px': 'F','b_py': 'F', 'e_px': 'E','e_py': 'E','b_1d':'E','e_1d':'F'}
    dtype = complex


    def _b_1dDeriv_u(self,src,v,adjoint=False):
    	"""
    	The derivative of b_1d wrt u
    	"""
    	return -( self.mesh.nodalGrad * v)/( 1j*omega(src.freq) )