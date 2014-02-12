from SimPEG import Utils
from SimPEG.Data import BaseData

class DataTDEM1D(BaseData):
    """
        docstring for DataTDEM1D
    """

    txLoc = None #: txLoc
    txType = None #: txType
    rxLoc = None #: rxLoc
    rxType = None #: rxType
    timeCh = None #: timeCh

    def __init__(self, **kwargs):
        BaseData.__init__(self, **kwargs)
        Utils.setKwargs(self, **kwargs)

    def projectFields(self, u):
        return self.Qrx.dot(u.b[:,:,0].T)

    ####################################################
    # Interpolation Matrices
    ####################################################

    @property
    def Qrx(self):
        if self._Qrx is None:
            if self.rxType == 'bz':
                locType = 'fz'
            self._Qrx = self.prob.mesh.getInterpolationMat(self.rxLoc, locType=locType)
        return self._Qrx
    _Qrx = None
