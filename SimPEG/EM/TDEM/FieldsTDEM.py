import numpy as np
import scipy.sparse as sp
import SimPEG
from SimPEG import Utils
from SimPEG.EM.Utils import omega
from SimPEG.Utils import Zero, Identity

class Fields(SimPEG.Problem.Fields):
    """
    
    Fancy Field Storage for a TDEM survey. Only one field type is stored for
    each problem, the rest are computed. The fields obejct acts like an array and is indexed by

    .. code-block:: python

        f = problem.fields(m)
        e = f[srcList,'e']
        b = f[srcList,'b']

    If accessing all sources for a given field, use the :code:`:`

    .. code-block:: python

        f = problem.fields(m)
        e = f[:,'e']
        b = f[:,'b']

    The array returned will be size (nE or nF, nSrcs :math:`\\times` nFrequencies)
    """

    knownFields = {}
    dtype = float 

    


class Fields_b(Fields):
    """Fancy Field Storage for a TDEM survey."""
    knownFields = {'bSolution': 'F'}
    aliasFields = {
                    'b': ['bSolution', 'F', '_b'],
                    'e': ['bSolution', 'E', '_e'],
                  }

    def startup(self):
        self.MeSigmaI = self.survey.prob.MeSigmaI
        self.edgeCurl = self.survey.prob.mesh.edgeCurl
        self.MfMui    = self.survey.prob.MfMui

    def _b(self, bSolution, srcList, tInd):
        return bSolution

    def _bDeriv_u(self, src, du_dm_v, adjoint = False):
        return Identity()*v

    def _bDeriv_m(self, src, v, adjoint = False):
        return Zero()

    def _e(self, bSolution, srcList, tInd):
        return self.MeSigmaI * ( self.edgeCurl.T * ( self.MfMui * bSolution ) )

    def _eDeriv_u(self, src, du_dm_v, adjoint = False):
        raise NotImplementedError

    def _eDeriv_m(self, src, v, adjoint = False):
        raise NotImplementedError


