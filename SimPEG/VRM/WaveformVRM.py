import numpy as np

###################################################
#           STEP OFF WAVEFORM
###################################################


class StepOff():

    """
Step-Off waveform for predicting VRM response.

INPUTS:

KWARGS:

t0: The start of the off-time
"""


def __init__(self, t0=0.):

    self.t0 = t0

    def getCharDecay(self, fieldType, times):

        """
Characteristic decay function for step-off waveform. This function describes
the decay of the VRM response for the Linear problem type. Note that the
current will be normalized by its maximum value. The maximum current in the
transmitter is specified in the source object.

INPUTS:

fieldType: must be 'dhdt' or 'dbdt'. Characteristic decay for 'h' or 'b' cannot
be computed for step-off times: Observation times. These times must be during
the off-time.
        """

        assert fieldType in ["dhdt", "dbdt"], "For step-off, fieldType must be one of 'dhdt' or 'dbdt' and cannot be 'h' or 'b'"
        assert self.t0 < np.min(times), "Earliest time channel must be after beginning of off-time"

        t0 = self.t0

        if fieldType is "dbdt":
            mu0 = 4*np.pi*1e-7
            eta = -mu0/(times-t0)
        elif fieldType is "dhdt":
            eta = -1/(times-t0)

        return eta

    # def getTrueDecay(self, fieldType, times, chi0, dchi, tau1, tau2):

###################################################
#           SQUARE PULSE WAVEFORM
###################################################


class SquarePulse():

    """
Square-pulse waveform for predicting VRM response.

INPUTS:

delt: During of the on-time

KWARGS:

t0: The start of the off-time
    """

    def __init__(self, delt, t0=0.):

        self.delt = delt
        self.t0 = t0

    def getCharDecay(self, fieldType, times):

        """
Characteristic decay function for square-pulse waveform. This function
describes the decay of the VRM response for the Linear problem type. Note that
the current will be normalized by its maximum value. The maximum current in the
transmitter is specified in the source object.

INPUTS:

fieldType: must be 'h', 'b', 'dhdt' or 'dbdt'.
times: Observation times. These times must be during the off-time.
        """

        assert fieldType in ["h", "dhdt", "b", "dbdt"], "For square-pulse, fieldType must be one of 'h', 'dhdt', 'b' or 'dbdt'"
        assert self.t0 < np.min(times), "Earliest time channel must be after beginning of off-time"

        t0 = self.t0
        delt = self.delt
        mu0 = 4*np.pi*1e-7

        if fieldType is "h":
            eta = np.log(1 + delt/(times-t0))
        elif fieldType is "b":
            eta = mu0*np.log(1 + delt/(times-t0))
        elif fieldType is "dhdt":
            eta = -(1/(times-t0) - 1/(times-t0+delt))
        elif fieldType is "dbdt":
            eta = -mu0*(1/(times-t0) - 1/(times-t0+delt))

        return eta

###################################################
#           ARBITRARY WAVEFORM
###################################################


class Arbitrary():

    """
Arbitrary waveform for predicting VRM response. Note that the current will be
normalized by its maximum value. The maximum current in the transmitter is
specified in the source object.

INPUTS:

t: Times for the waveform
I: Current for the waveform
    """

    def __init__(self, t, I):

        assert np.abs(I[0]) < 1e-10 and np.abs(I[-1]) < 1e-10, "Current at t0 and tmax should be 0"
        assert len(t) == len(I), "Time values and current values must have same length"

        self.t = t
        self.I = I

    def getCharDecay(self, fieldType, times):

        """
Characteristic decay function for arbitrary waveform. This function describes
the decay of the VRM response for the Linear problem type. Note that the
current will be normalized by its maximum value. The maximum current in the
transmitter is specified in the source object.

INPUTS:

fieldType: must be 'h', 'b', 'dhdt' or 'dbdt'.
times: Observation times. These times must be during the off-time.
        """

        assert fieldType in ["h", "dhdt", "b", "dbdt"], "fieldType must be one of 'h', 'dhdt', 'b' or 'dbdt'"
        assert np.max(self.t) < np.min(times), "Earliest time channel must be after beginning of off-time"

        k = np.nonzero(self.I)
        j = k[0][0]
        k = k[0][-1]

        twave = self.t[j:k+1]
        Iwave = self.I[j:k+1]/np.max(np.abs(self.I[j:k+1]))

        N = int(np.ceil(25*(np.max(twave)-np.min(twave))/np.min(times)))

        if N > 25000:
            N = 25000

        dt = (np.max(twave) - np.min(twave))/np.float64(N)
        tvec = np.linspace(np.min(twave), np.max(twave)-dt, N)
        # g evaluated at middle of pulses
        g = np.interp(tvec+dt/2, twave, Iwave)

        eta = np.zeros(len(times))

        if fieldType in ["h", "b"]:
            for tt in range(0, len(eta)):
                eta[tt] = np.sum(g*np.log(1 + dt/(times[tt] - tvec + dt)))
        elif fieldType in ["dhdt", "dbdt"]:
            for tt in range(0, len(eta)):
                eta[tt] = np.sum(g*(1/(times[tt] - tvec + dt) - 1/(times[tt] - tvec)))

        if fieldType in ["b", "dbdt"]:
            mu0 = 4*np.pi*1e-7
            eta = mu0*eta

        return eta











