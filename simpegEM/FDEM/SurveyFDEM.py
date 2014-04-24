from SimPEG import Survey, Utils, np, sp

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
                   }

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

    freq = None #: Frequency (float)

    rxListPair = RxFDEM

    knownTxTypes = ['VMD']

    def __init__(self, loc, txType, freq, rxList):
        self.freq = float(freq)
        Survey.BaseTx.__init__(self, loc, txType, rxList)



class FieldsFDEM(object):
    """Fancy Field Storage for a FDEM survey."""

    knownFields = {'b': 'F', 'e': 'E'}

    def __init__(self, mesh, survey):
        self.survey = survey
        self.mesh = mesh
        self._fields = {}

    def _initStore(self, name):
        if name in self._fields:
            return self._fields[name]

        assert name in self.knownFields, 'field name is not known.'

        loc = self.knownFields[name]

        nP = {'CC': self.mesh.nC,
              'F':  self.mesh.nF,
              'E':  self.mesh.nE}[loc]

        field = {}
        for freq in self.survey.freqs:
            nTx_f = len(self.survey.getTransmitters(freq))
            field[freq] = np.empty((nP, nTx_f))

        self._fields[name] = field

        return field

    def _ensureCorrectKey(self, key):
        if type(key) is tuple:
            assert len(key) == 2, 'must be [freq, fieldName]'
            freqTest, name = key

            if name not in self.knownFields:
                raise KeyError('Invalid field name')

            if type(freqTest) is float:
                freq = freqTest
            elif isinstance(freqTest, TxFDEM):
                freq = freqTest.freq
                if freqTest not in self.survey.txList:
                    raise KeyError('Invalid Transmitter')
            else:
                raise KeyError('Invalid Frequency Key')

        elif type(key) is float:
            freq = key
        elif isinstance(key, TxFDEM):
            freq = key.freq
            if key not in self.survey.txList:
                raise KeyError('Invalid Transmitter')
        else:
            raise KeyError('Unexpected key use [freq, fieldName]')
        if freq not in self.survey.freqs:
            raise KeyError('Invalid frequency')

    def __setitem__(self, key, value):
        self._ensureCorrectKey(key)
        if type(key) is tuple:
            freq, name = key
            assert type(freq) is float, 'Frequency must be a float for setter.'
            assert type(value) is np.ndarray, 'Must be set to a numpy array'
            newFields = {name: value}
        elif type(key) is float:
            freq = key
            assert type(value) is dict, 'New fields must be a dictionary'
            newFields = value
        elif isinstance(key, TxFDEM):
            raise Exception('Cannot set one transmitter at a time.')

        for name in newFields:
            field = self._initStore(name)
            if field[freq].shape[1] == 1:
                newFields[name] = Utils.mkvc(newFields[name],2)
            assert field[freq].shape == newFields[name].shape, 'Must be correct shape (n%s x nTx[freq])' % self.knownFields[name]
            field[freq] = newFields[name]

    def __getitem__(self, key):
        self._ensureCorrectKey(key)
        if type(key) is tuple:
            freqTest, name = key
            if type(freqTest) is float:
                return self._fields[name][freqTest]
            elif isinstance(freqTest, TxFDEM):
                key = freqTest
                ind = np.array([tx is key for tx in self.survey.getTransmitters(key.freq)])
                return Utils.mkvc(self._fields[name][key.freq][:,ind])
        elif type(key) is float:
            freq = key
            out = {}
            for name in self._fields:
                out[name] = self._fields[name][freq]
            return out
        elif isinstance(key, TxFDEM):
            freq = key.freq
            ind = np.array([tx is key for tx in self.survey.getTransmitters(freq)])
            out = {}
            for name in self._fields:
                out[name] = Utils.mkvc(self._fields[name][freq][:,ind])
            return out


    def __contains__(self, key):
        return key in self.children


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
    def nTx(self):
        if getattr(self, '_nTx', None) is None:
            self._nTx = {}
            for freq in self.freqs:
                self._nTx[freq] = len(self.getTransmitters(freq))
        return self._nTx

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
