from SimPEG import Survey, Problem, Utils, np, sp
from simpegEM import Sources
from simpegEM.Utils.EMUtils import omega

def omega(freq):
    """Change frequency to angular frequency, omega"""
    return 2.*np.pi*freq

class RxFDEM(Survey.BaseRx):

    knownRxTypes = {
                    'exr':['e', 'Ex', 'real'],
                    'eyr':['e', 'Ey', 'real'],
                    'ezr':['e', 'Ez', 'real'],
                    'exi':['e', 'Ex', 'imag'],
                    'eyi':['e', 'Ey', 'imag'],
                    'ezi':['e', 'Ez', 'imag'],

                    'bxr':['b', 'Fx', 'real'],
                    'byr':['b', 'Fy', 'real'],
                    'bzr':['b', 'Fz', 'real'],
                    'bxi':['b', 'Fx', 'imag'],
                    'byi':['b', 'Fy', 'imag'],
                    'bzi':['b', 'Fz', 'imag'],

                    'jxr':['j', 'Fx', 'real'],
                    'jyr':['j', 'Fy', 'real'],
                    'jzr':['j', 'Fz', 'real'],
                    'jxi':['j', 'Fx', 'imag'],
                    'jyi':['j', 'Fy', 'imag'],
                    'jzi':['j', 'Fz', 'imag'],

                    'hxr':['h', 'Ex', 'real'],
                    'hyr':['h', 'Ey', 'real'],
                    'hzr':['h', 'Ez', 'real'],
                    'hxi':['h', 'Ex', 'imag'],
                    'hyi':['h', 'Ey', 'imag'],
                    'hzi':['h', 'Ez', 'imag'],
                   }
    radius = None

    def __init__(self, locs, rxType):
        Survey.BaseRx.__init__(self, locs, rxType)

    @property
    def projField(self):
        """Field Type projection (e.g. e b ...)"""
        return self.knownRxTypes[self.rxType][0]

    @property
    def projGLoc(self):
        """Grid Location projection (e.g. Ex Fy ...)"""
        return self.knownRxTypes[self.rxType][1]

    @property
    def projComp(self):
        """Component projection (real/imag)"""
        return self.knownRxTypes[self.rxType][2]

    def projectFields(self, tx, mesh, u):
        P = self.getP(mesh)
        u_part_complex = u[tx, self.projField]
        # get the real or imag component
        real_or_imag = self.projComp
        u_part = getattr(u_part_complex, real_or_imag)
        return P*u_part

    def projectFieldsDeriv(self, tx, mesh, u, v, adjoint=False):
        P = self.getP(mesh)

        if not adjoint:
            Pv_complex = P * v
            real_or_imag = self.projComp
            Pv = getattr(Pv_complex, real_or_imag)
        elif adjoint:
            Pv_real = P.T * v

            real_or_imag = self.projComp
            if real_or_imag == 'imag':
                Pv = 1j*Pv_real
            elif real_or_imag == 'real':
                Pv = Pv_real.astype(complex)
            else:
                raise NotImplementedError('must be real or imag')

        return Pv


class TxFDEM(Survey.BaseTx):
    #TODO: Break these out into Classes of Sources.

    freq = None #: Frequency (float)

    rxPair = RxFDEM

    knownTxTypes = ['VMD', 'VMD_B', 'CircularLoop', 'Simple']

    radius = None

    def __init__(self, loc, txType, freq, rxList):
        self.freq = float(freq)
        Survey.BaseTx.__init__(self, loc, txType, rxList)

    def getSource(self, prob):

        tx = self
        freq = tx.freq
        solType = prob.solType

        if solType == 'e' or solType == 'b':
            gridEJx = prob.mesh.gridEx
            gridEJy = prob.mesh.gridEy
            gridEJz = prob.mesh.gridEz
            nEJ = prob.mesh.nE

            gridBHx = prob.mesh.gridFx
            gridBHy = prob.mesh.gridFy
            gridBHz = prob.mesh.gridFz
            nBH = prob.mesh.nF


            C = prob.mesh.edgeCurl
            mui = prob.MfMui

        elif solType == 'h' or solType == 'j':
            gridEJx = prob.mesh.gridFx
            gridEJy = prob.mesh.gridFy
            gridEJz = prob.mesh.gridFz
            nEJ = prob.mesh.nF

            gridBHx = prob.mesh.gridEx
            gridBHy = prob.mesh.gridEy
            gridBHz = prob.mesh.gridEz
            nBH = prob.mesh.nE

            C = prob.mesh.edgeCurl.T
            mui = prob.MeMuI

        else:
            NotImplementedError('Only E or F sources')


        if prob.mesh._meshType is 'CYL':
            if not prob.mesh.isSymmetric:
                raise NotImplementedError('Non-symmetric cyl mesh not implemented yet!')

            if tx.txType == 'VMD':
                SRC = Sources.MagneticDipoleVectorPotential(tx.loc, gridEJy, 'y')
            elif tx.txType == 'CircularLoop':
                SRC = Sources.MagneticLoopVectorPotential(tx.loc, gridEJy, 'y', tx.radius)
            else:
                raise NotImplementedError('Only VMD and CircularLoop')

        elif prob.mesh._meshType is 'TENSOR':

            if tx.txType == 'VMD':
                src = Sources.MagneticDipoleVectorPotential
                SRCx = src(tx.loc, gridEJx, 'x')
                SRCy = src(tx.loc, gridEJy, 'y')
                SRCz = src(tx.loc, gridEJz, 'z')

            elif tx.txType == 'VMD_B':
                src = Sources.MagneticDipoleFields
                SRCx = src(tx.loc, gridBHx, 'x')
                SRCy = src(tx.loc, gridBHy, 'y')
                SRCz = src(tx.loc, gridBHz, 'z')

            elif tx.txType == 'CircularLoop':
                src = Sources.MagneticLoopVectorPotential
                SRCx = src(tx.loc, gridEJx, 'x', tx.radius)
                SRCy = src(tx.loc, gridEJy, 'y', tx.radius)
                SRCz = src(tx.loc, gridEJz, 'z', tx.radius)
            else:

                raise NotImplemented('%s txType is not implemented' % tx.txType)
            SRC = np.concatenate((SRCx, SRCy, SRCz))

        else:
            raise Exception('Unknown mesh for VMD')

        # b-forumlation
        if tx.txType == 'VMD_B':
            b_0 = SRC
        else:
            a = SRC
            b_0 = C*a

        # if solType == 'b' or solType == 'h':
        return -1j*omega(freq)*b_0, None
        # elif solType == 'e' or solType == 'j':
        #     return -1j*omega(freq)*C.T*mui*b_0, None

class SimpleTxFDEM_g(TxFDEM):

    def __init__(self, vec, freq, rxList):
        self.vec = vec
        self.freq = float(freq)
        TxFDEM.__init__(self, None, 'Simple', freq, rxList)

    def getSource(self, prob):
        return None, self.vec


class SimpleTxFDEM_m(TxFDEM):

    def __init__(self, vec, freq, rxList):
        self.vec = vec
        self.freq = float(freq)
        TxFDEM.__init__(self, None, 'Simple', freq, rxList)

    def getSource(self, prob):
        return self.vec, None


class FieldsFDEM(Problem.Fields):
    """Fancy Field Storage for a FDEM survey."""
    knownFields = {'b': 'F', 'e': 'E', 'j': 'F', 'h': 'E'}
    dtype = complex


class SurveyFDEM(Survey.BaseSurvey):
    """
        docstring for SurveyFDEM
    """

    txPair = TxFDEM

    def __init__(self, txList, **kwargs):
        # Sort these by frequency
        self.txList = txList
        Survey.BaseSurvey.__init__(self, **kwargs)

        _freqDict = {}
        for tx in txList:
            if tx.freq not in _freqDict:
                _freqDict[tx.freq] = []
            _freqDict[tx.freq] += [tx]

        self._freqDict = _freqDict
        self._freqs = sorted([f for f in self._freqDict])

    @property
    def freqs(self):
        """Frequencies"""
        return self._freqs

    @property
    def nFreq(self):
        """Number of frequencies"""
        return len(self._freqDict)

    @property
    def nTxByFreq(self):
        if getattr(self, '_nTxByFreq', None) is None:
            self._nTxByFreq = {}
            for freq in self.freqs:
                self._nTxByFreq[freq] = len(self.getTransmitters(freq))
        return self._nTxByFreq

    def getTransmitters(self, freq):
        """Returns the transmitters associated with a specific frequency."""
        assert freq in self._freqDict, "The requested frequency is not in this survey."
        return self._freqDict[freq]

    def projectFields(self, u):
        data = Survey.Data(self)
        for tx in self.txList:
            for rx in tx.rxList:
                data[tx, rx] = rx.projectFields(tx, self.mesh, u)
        return data

    def projectFieldsDeriv(self, u):
        raise Exception('Use Transmitters to project fields deriv.')
