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

if __name__ == '__main__':
    from SimPEG import *
    import simpegEM as EM

    cs = 5.
    ncx = 20
    ncy = 6
    npad = 15
    hx = Utils.meshTensors(((0,cs), (ncx,cs), (npad,cs)))
    hy = Utils.meshTensors(((npad,cs), (ncy,cs), (npad,cs)))
    mesh = Mesh.Cyl1DMesh([hx,hy], -hy.sum()/2)
    model = Model.Vertical1DModel(mesh)

    txLoc = 0.
    txType = 'VMD_MVP'
    rxLoc = np.r_[150., 0.]
    rxType = 'bz'
    timeCh = np.logspace(-4,-2,20)
    dat = EM.TDEM.DataTDEM1D(txLoc=txLoc, txType=txType, rxLoc=rxLoc, rxType=rxType, timeCh=timeCh)

    prb = EM.TDEM.ProblemTDEM_b(mesh, model)
    prb.setTimes([1e-5, 5e-5, 2.5e-4], [50, 50, 50])
    prb.sigma = np.ones(mesh.nCz)
    prb.pair(dat)

    F = prb.field(prb.sigma)