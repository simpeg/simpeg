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

        self._Ps = {}
        self._Ps[self.projGLoc] = {}

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

    def getP(self, mesh):
        """
            Returns the projection matrices as a
            list for all components collected by
            the receivers.

            .. note::

                Projection matrices are stored as a nested dict,
                First gridLocation, then mesh.
        """
        gloc = self.projGLoc
        if mesh not in self._Ps[gloc]:
            self._Ps[gloc][mesh] = mesh.getInterpolationMat(self.locs, gloc)
        P = self._Ps[gloc][mesh]
        return P


class TxFDEM(Survey.BaseTx):

    freq = None #: Frequency (float)

    rxListPair = RxFDEM

    knownTxTypes = ['VMD']

    def __init__(self, loc, txType, freq, rxList):
        self.freq = float(freq)
        Survey.BaseTx.__init__(self, loc, txType, rxList)

    @property
    def nD(self):
        """Number of data"""
        return self.vnD.sum()

    @property
    def vnD(self):
        """Vector number of data"""
        return np.array([rx.locs.shape[0] for rx in self.rxList])

    def projectFields(self, mesh, u):

        nRt = len(self.rxList)
        Pu = range(nRt)

        for ii, rx in enumerate(self.rxList):
            P = rx.getP(mesh)
            u_part_complex = u[self, rx.projField]
            # get the real or imag component
            real_or_imag = rx.projComp
            u_part = getattr(u_part_complex, real_or_imag)
            Pu[ii] = P*u_part
        return np.concatenate(Pu)

    def projectFieldsDeriv(self, mesh, u, v, adjoint=False):
        V = v.reshape((-1, len(Ps)), order='F')
        Pvs = range(len(Ps))
        for ii, rx in enumerate(self.rxList):
            P = rx.getP(mesh)

            if not adjoint:
                Pv = Ps[ii] * V[:, ii]
            elif adjoint:
                Pv = Ps[ii].T * V[:, ii]

            real_or_imag = rx.projComp
            if real_or_imag == 'imag':
                Pvs[ii] = 1j*Pv
            elif real_or_imag == 'real':
                Pvs[ii] = Pv.astype(complex)
            else:
                raise NotImplementedError('must be real or imag')

        return np.concatenate(Pvs)


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


class DataFDEM(object):
    """docstring for DataFDEM"""
    def __init__(self, survey, v=None):
        self.survey = survey
        self._dataDict = {}
        if v is not None:
            self.fromvec(v)

    def _ensureCorrectKey(self, key):
        if key not in self.survey.txList:
            raise KeyError('Key must be a transmitter in the survey.')

    def __setitem__(self, key, value):
        self._ensureCorrectKey(key)
        assert type(value) == np.ndarray, 'value must by ndarray'
        assert value.size == key.nD, "value must have the same number of data as the transmitter."
        self._dataDict[key] = Utils.mkvc(value)

    def __getitem__(self, key):
        self._ensureCorrectKey(key)
        if key not in self._dataDict:
            raise Exception('Data for transmitter has not yet been set.')
        return self._dataDict[key]

    def tovec(self):
        return np.concatenate([self[tx] for tx in self.survey.txList])

    def fromvec(self, v):
        v = Utils.mkvc(v)
        assert v.size == self.survey.nD, 'v must have the correct number of data.'
        indBot, indTop = 0, 0
        for tx in self.survey.txList:
            indTop += tx.nD
            self[tx] = v[indBot:indTop]
            indBot += tx.nD




class SurveyFDEM(Survey.BaseSurvey):
    """
        docstring for SurveyFDEM
    """

    txPair = TxFDEM

    def __init__(self, txList, **kwargs):
        assert type(txList) is list, 'txList must be a list'
        for tx in txList:
            assert isinstance(tx, self.txPair), 'txList must be a %s'%self.txPair.__name__

        assert len(set(txList)) == len(txList), 'The txList must be unique'
        # Sort these by frequency
        _freqDict = {}
        for tx in txList:
            if tx.freq not in _freqDict:
                _freqDict[tx.freq] = []
            _freqDict[tx.freq] += [tx]

        self._txList = txList
        self._freqDict = _freqDict
        self._freqs = sorted([f for f in self._freqDict])

        Survey.BaseSurvey.__init__(self, **kwargs)

    @property
    def nD(self):
        """Number of data"""
        return np.array([tx.nD for tx in self.txList]).sum()

    @property
    def freqs(self):
        """Frequencies"""
        return self._freqs

    @property
    def nFreq(self):
        """Number of frequencies"""
        return len(self._freqDict)

    @property
    def txList(self):
        """Transmitter List"""
        return self._txList

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
        data = DataFDEM(self)
        for tx in self.txList:
            data[tx] = tx.projectFields(self.mesh, u)
        return data

    def projectFieldsDeriv(self, u):
        raise Exception('Use Transmitters to project fields deriv.')
