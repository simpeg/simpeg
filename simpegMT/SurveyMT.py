from SimPEG import Survey, Utils, Problem, np, sp

class RxMT(Survey.BaseRx):

    knownRxTypes = {
                    'zxxr':[['e', 'Ex'],['b','Fx'], 'real'],
                    'zxyr':[['e', 'Ex'],['b','Fy'], 'real'],
                    'zyxr':[['e', 'Ey'],['b','Fx'], 'real'],
                    'zyyr':[['e', 'Ey'],['b','Fy'], 'real'],
                    'zxxi':[['e', 'Ex'],['b','Fx'], 'imag'],
                    'zxyi':[['e', 'Ex'],['b','Fy'], 'imag'],
                    'zyxi':[['e', 'Ey'],['b','Fx'], 'imag'],
                    'zyyi':[['e', 'Ey'],['b','Fy'], 'imag'],

                    #TODO: Add tipper fractions as well. Bz/B(x|y)
                    # 'exi':['e', 'Ex', 'imag'],
                    # 'eyi':['e', 'Ey', 'imag'],
                    # 'ezi':['e', 'Ez', 'imag'],

                    # 'bxr':['b', 'Fx', 'real'],
                    # 'byr':['b', 'Fy', 'real'],
                    # 'bzr':['b', 'Fz', 'real'],
                    # 'bxi':['b', 'Fx', 'imag'],
                    # 'byi':['b', 'Fy', 'imag'],
                    # 'bzi':['b', 'Fz', 'imag'],
                   }
    # TODO: Have locs as single or double coordinates for both or numerator and denominator separately, respectively.
    def __init__(self, locs, rxType):
        Survey.BaseRx.__init__(self, locs, rxType)

    @property
    def projField(self,fracPos):
        """
        Field Type projection (e.g. e b ...)
        :param str fracPos: Position of the field in the data ratio
        
        """
        if 'numerator' in fracPos:
            return self.knownRxTypes[self.rxType][0][0]
        elif 'denominator' in fracPos:
            return self.knownRxTypes[self.rxType][1][0]
        else:
            raise Exception('{s} is an unknown option. Use numerator or denominator.')

    @property
    def projGLoc(self,fracPos):
        """
        Grid Location projection (e.g. Ex Fy ...)
        :param str fracPos: Position of the field in the data ratio
        
        """
        if 'numerator' in fracPos:
            return self.knownRxTypes[self.rxType][0][1]
        elif 'denominator' in fracPos:
            return self.knownRxTypes[self.rxType][0][1]
        else:
            raise Exception('{s} is an unknown option. Use numerator or denominator.')

    @property
    def projComp(self):
        """Component projection (real/imag)"""
        return self.knownRxTypes[self.rxType][2]

    def projectFields(self, src, mesh, u):
        '''
        Project the fields and return the 
        '''
        # Get the numerator information
        P_num = self.getP(mesh,self.projGLoc('numerator'))
        u_num_complex = u[src, self.projField('numerator')]
        # Get the denominator information
        P_den = self.getP(mesh,self.projGLoc('denominator'))
        u_den_complex = u[src, self.projField('denominator')]
        # Calculate the fraction
        f_part_complex = (P_num*u_num_complex)/(P_den*u_den_complex)
        # get the real or imag component
        real_or_imag = self.projComp
        f_part = getattr(f_part_complex, real_or_imag)
        return f_part

    def projectFieldsDeriv(self, src, mesh, u, v, adjoint=False):
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


# Call this Source or polarization or something...?
class srcMT(Survey.BaseTx):
    '''
    Sources for the MT problem. 
    Use the SimPEG BaseTx, since the source fields share properties with the transmitters.

    :param float freq: The frequency of the source
    :param list rxList: A list of receivers associated with the source
    '''

    freq = None #: Frequency (float)

    rxPair = RxMT

    knownTxTypes = ['ORTPOL'] # ORThogonal POLarization

    def __init__(self, freq, rxList): # remove txType? hardcode to one thing. always polarizations
        self.freq = float(freq)
        Survey.BaseTx.__init__(self, None, 'ORTPOL', rxList)
        # Survey.BaseTx.__init__(self, loc, 'polarization', rxList)



class FieldsMT(Problem.Fields):
    """Fancy Field Storage for a MT survey."""
    knownFields = {'b': 'F', 'e': 'E'}
    dtype = complex


class SurveyMT(Survey.BaseSurvey):
    """
        Survey class for MT. Contains all the sources associated with the survey.

        :param list srcList: List of sources associated with the survey

    """

    srcPair = srcMT

    def __init__(self, srcList, **kwargs):
        # Sort these by frequency
        self.srcList = srcList
        Survey.BaseSurvey.__init__(self, **kwargs)

        _freqDict = {}
        for src in srcList:
            if src.freq not in _freqDict:
                _freqDict[src.freq] = []
            _freqDict[src.freq] += [src]

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
    # Don't need this
    # @property
    # def nTxByFreq(self):
    #     if getattr(self, '_nTxByFreq', None) is None:
    #         self._nTxByFreq = {}
    #         for freq in self.freqs:
    #             self._nTxByFreq[freq] = len(self.getTransmitters(freq))
    #     return self._nTxByFreq

    # TODO: Rename to getSources
    def getSources(self, freq):
        """Returns the transmitters associated with a specific frequency."""
        assert freq in self._freqDict, "The requested frequency is not in this survey."
        return self._freqDict[freq]

    def projectFields(self, u):
        data = Survey.Data(self)
        for src in self.srcList:
            for rx in src.rxList:
                data[src, rx] = rx.projectFields(src, self.mesh, u)
        return data

    def projectFieldsDeriv(self, u):
        raise Exception('Use Transmitters to project fields deriv.')
