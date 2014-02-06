from BaseTDEM import ProblemBaseTDEM
from scipy.constants import mu_0
from SimPEG.Utils import sdiag

class ProblemTDEM_b(ProblemBaseTDEM):
    """
        docstring for ProblemTDEM_b
    """
    def __init__(self, mesh, model, **kwargs):
        ProblemBaseTDEM.__init__(self, mesh, model, **kwargs)

    solType = 'b'

    ####################################################
    # Physical Properties
    ####################################################

    @property
    def sigma(self):
        return self._sigma
    @sigma.setter
    def sigma(self, value):
        self._sigma = value
    _sigma = None

    ####################################################
    # Mass Matrices
    ####################################################

    @property
    def MfMui(self):
        if self._MfMui is None:
            self._MfMui = self.mesh.getMass(1/mu_0, loc='f')
        return self._MfMui
    @MfMui.setter
    def MfMui(self, value):
        self._MfMui = value
    _MfMui = None

    @property
    def MeSigmaI(self):
        if self._MeSigmaI is None:
            MeSigma = self.mesh.getMass(self.sigma, loc='e')
            self._MeSigmaI = sdiag(1/MeSigma.diagonal())
        return self._MeSigmaI
    @MeSigmaI.setter
    def MeSigmaI(self, value):
        self._MeSigmaI = value
    _MeSigmaI = None
    
    ####################################################
    # Internal Methods
    ####################################################

    def getA(self, tInd):
        dt = self.getDt(tInd)
        return self.MfMui*self.mesh.edgeCurl*self.MeSigmaI*self.mesh.edgeCurl.T*self.MfMui + (1/dt)*self.MfMui

    def getRHS(self, tInd, F):
        dt = self.getDt(tInd)
        return (1/dt)*self.MfMui*F.get_b(tInd-1)        
