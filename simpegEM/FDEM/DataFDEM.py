from SimPEG import Utils, np
from SimPEG.Data import BaseData
from FieldsFDEM import FieldsFDEM

class DataFDEM(BaseData):
    """
        docstring for DataFDEM
    """

    txLoc = None #: txLoc
    txType = None #: txType
    nTx    = 1 #: Number of transmitters
    rxLoc = None #: rxLoc
    rxType = None #: rxType
    freq = None #: freq


    @property
    def omega(self):
        return 2*np.pi*self.freq

    @property
    def nFreq(self):
        """Number of frequencies"""
        return self.freq.size

    def __init__(self, **kwargs):
        BaseData.__init__(self, **kwargs)
        Utils.setKwargs(self, **kwargs)

    def projectFields(self, u):
        #TODO: this is hardcoded to 1Tx
        return self.Qrx.dot(u.b[:,:,0].T).T

    def projectFieldsAdjoint(self, d):
        # TODO: fix this
        pass

    ####################################################
    # Interpolation Matrices
    ####################################################

    @property
    def Qrx(self):
        if self._Qrx is None:
            if self.rxType == 'bz':
                locType = 'Fz'
            self._Qrx = self.prob.mesh.getInterpolationMat(self.rxLoc, locType=locType)
        return self._Qrx
    _Qrx = None
