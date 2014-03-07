from SimPEG import Survey, Utils, np, sp
from FieldsFDEM import FieldsFDEM

class SurveyFDEM(Survey.BaseSurvey):
    """
        docstring for SurveyFDEM
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
        Survey.BaseSurvey.__init__(self, **kwargs)
        Utils.setKwargs(self, **kwargs)

    @property
    def nRx(self):
        return self.rxLoc.shape[0]

    def projectFields(self, u):
        P = sp.identity(self.prob.mesh.nE)
        Pes = range(self.nFreq)
        #TODO: this is hardcoded to 1Tx
        for i, freqInd in enumerate(range(self.nFreq)):
            e = u.get_e(freqInd)
            Pes[i] = P*e
        Pe = np.concatenate(Pes)
        return Pe

    def projectFieldsDeriv(self, u):
        # TODO : more general
        return sp.identity(self.prob.mesh.nE)

    ####################################################
    # Interpolation Matrices
    ####################################################


