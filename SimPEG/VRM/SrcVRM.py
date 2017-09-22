import numpy as np
from SimPEG import Props, Utils, Survey
from RxVRM import BaseRxVRM






#########################################
# VRM WAVEFORM CLASS
#########################################

# class waveformVRM():











#########################################
# BASE VRM SOURCE CLASS
#########################################

class BaseSrcVRM(Survey.BaseSrc):
    """SimPEG Source Object"""

    def __init__(self, rxList, **kwargs):
        super(BaseSrcVRM, self).__init__(rxList, **kwargs)
        self.rxPair = BaseRxVRM # Links base Src class to acceptable Rx class?

    @property
    def nRx(self):
        """Number of data"""
        return len(self.rxList)

    @property
    def nD(self):
        """Vector number of receivers"""
        return np.array([rx.nD for rx in self.rxList])

    


#########################################
# MAGNETIC DIPOLE VRM SOURCE CLASS
#########################################

class MagDipole(BaseSrcVRM):

    def __init__(self, rxList, loc, moment, **kwargs):

        waveform = 'StepOff'

        assert len(loc) is 3, 'Tx location must be given as a column vector np.r[x,y,z]'
        assert len(moment) is 3, 'Dipole moment given as column vector np.r_[mx,my,mz]'
        super(MagDipole, self).__init__(rxList, **kwargs)
        
        self.waveform = waveform
        self.moment = moment


#########################################
# LINE CURRENT VRM SOURCE CLASS
#########################################

class LineCurrent(BaseSrcVRM):

    def __init__(self, rxList, locs, **kwargs):

        Imax = 1.
        waveform = 'StepOff'

        assert np.shape(locs)[1] == 3 and np.shape(locs)[0] > 1, 'locs is a N+1 by 3 array where N is the number of transmitter segments'
        if waveform is not 'StepOff':
            assert np.shape(waveform)[1] is 2, 'For custom waveforms, must have times and current (N X 2 array)'
        self.Imax = Imax
        self.waveform = waveform
        super(LineCurrent, self).__init__(rxList, **kwargs)











