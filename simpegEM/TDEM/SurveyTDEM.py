from SimPEG import Utils, np
from SimPEG.Survey import BaseSurvey
from FieldsTDEM import FieldsTDEM

class SurveyTDEM1D(BaseSurvey):
    """
        docstring for SurveyTDEM1D
    """

    txLoc = None #: txLoc
    txType = None #: txType
    rxLoc = None #: rxLoc
    rxType = None #: rxType
    timeCh = None #: timeCh
    nTx    = 1 #: Number of transmitters

    @property
    def nTimeCh(self):
        """Number of time channels"""
        return self.timeCh.size

    def __init__(self, **kwargs):
        BaseSurvey.__init__(self, **kwargs)
        Utils.setKwargs(self, **kwargs)

    def projectFields(self, u):
        #TODO: this is hardcoded to 1Tx
        return self.Qrx.dot(u.b[:,:,0].T).T

    def projectFieldsAdjoint(self, d):
        # TODO: make the following self.nTimeCh
        d = d.reshape((self.prob.nTimes, self.nTx), order='F')
        #TODO: *Qtime.T need to multiply by a time projection. (outside for loop??)
        ii = 0
        F = FieldsTDEM(self.prob.mesh, self.nTx, self.prob.nTimes, 'b')
        for ii in range(self.prob.nTimes):
            b = self.Qrx.T*d[ii,:]
            F.set_b(b, ii)
            F.set_e(np.zeros((self.prob.mesh.nE,self.nTx)), ii)
        return F

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
