import numpy as np
import scipy.special as spec
import properties

###################################################
#           STEP OFF WAVEFORM
###################################################


class StepOff(properties.HasProperties):

    """

    """

    t0 = properties.Float('Start of off-time', default=0.)

    def getCharDecay(self, fieldType, times):

        """
        Characteristic decay function for step-off waveform. This function
        describes the decay of the VRM response for the linear problem type.
        Note that the current will be normalized by its maximum value. The
        maximum current in the transmitter is specified in the source object.

        REQUIRED ARGUMENTS:

        fieldType -- must be 'dhdt' or 'dbdt'. Characteristic decay for 'h'
        or 'b' CANNOT be computed for step-off

        times -- Observation times. These times MUST be during the off-time.

        OUTPUTS:

        eta -- characteristic decay function evaluated at all specified times.

        """

        assert fieldType in ["dhdt", "dbdt"], (
            "For step-off, fieldType must be one of 'dhdt' or 'dbdt' and cannot be 'h' or 'b'")

        assert self.t0 < np.min(times), "Earliest time channel must be after beginning of off-time"

        t0 = self.t0

        if fieldType is "dbdt":
            mu0 = 4*np.pi*1e-7
            eta = -mu0/(times-t0)
        elif fieldType is "dhdt":
            eta = -1/(times-t0)

        return eta

    def getLogUniformDecay(self, fieldType, times, chi0, dchi, tau1, tau2):

        """
        Decay function for a step-off waveform for log-uniform distribution of
        time-relaxation constants. The output of this function is the
        magnetization at each time for each cell, normalized by the inducing
        field.

        REQUIRED ARGUMENTS:

        fieldType -- must be 'h', 'b', 'dhdt' or 'dbdt'.

        times -- Observation times

        chi0 -- DC (zero-frequency) magnetic susceptibility for all cells

        dchi -- DC (zero-frequency) magnetic susceptibility attributed to VRM
        for all cells

        tau1 -- Lower-bound for log-uniform distribution of time-relaxation
        constants for all cells

        tau2 -- Upper-bound for log-uniform distribution of time-relaxation
        constants for all cells

        OUTPUTS:

        eta -- characteristic decay function evaluated at all specified times.

        """

        assert fieldType in ["h", "dhdt", "b", "dbdt"], (
            "For step-off, fieldType must be one of 'h', dhdt', 'b' or 'dbdt' ")

        nT = len(times)
        nC = len(dchi)
        t0 = self.t0

        times = np.kron(np.ones((nC, 1)), times)
        chi0 = np.kron(np.reshape(chi0, newshape=(nC, 1)), np.ones((1, nT)))
        dchi = np.kron(np.reshape(dchi, newshape=(nC, 1)), np.ones((1, nT)))
        tau1 = np.kron(np.reshape(tau1, newshape=(nC, 1)), np.ones((1, nT)))
        tau2 = np.kron(np.reshape(tau2, newshape=(nC, 1)), np.ones((1, nT)))

        if fieldType is "h":
            eta = (
                0.5*(1-np.sign(times-t0))*chi0 +
                0.5*(1+np.sign(times-t0))*(dchi/np.log(tau2/tau1)) *
                (spec.expi(-(times-t0)/tau2) - spec.expi(-(times-t0)/tau1))
                )
        elif fieldType is "b":
            mu0 = 4*np.pi*1e-7
            eta = (
                0.5*(1-np.sign(times-t0))*chi0 +
                0.5*(1+np.sign(times-t0))*(dchi/np.log(tau2/tau1)) *
                (spec.expi(-(times-t0)/tau2) - spec.expi(-(times-t0)/tau1))
                )
            eta = mu0*eta
        elif fieldType is "dhdt":
            eta = (
                0. + 0.5*(1+np.sign(times-t0))*(dchi/np.log(tau2/tau1)) *
                (np.exp(-(times-t0)/tau1)-np.exp(-(times-t0)/tau2))/(times-t0)
                )
        elif fieldType is "dbdt":
            mu0 = 4*np.pi*1e-7
            eta = (
                0. + 0.5*(1+np.sign(times-t0))*(dchi/np.log(tau2/tau1)) *
                (np.exp(-(times-t0)/tau1)-np.exp(-(times-t0)/tau2))/(times-t0)
                )
            eta = mu0*eta

        return eta


###################################################
#           SQUARE PULSE WAVEFORM
###################################################


class SquarePulse(properties.HasProperties):

    """

    """

    t0 = properties.Float('Start of off-time', default=0.)
    delt = properties.Float('Pulse width')

    def getCharDecay(self, fieldType, times):

        """
        Characteristic decay function for a square-pulse waveform. This
        function describes the decay of the VRM response for the linear
        problem type. Note that the current will be normalized by its maximum
        value. The maximum current in the transmitter is specified in the
        source object.

        REQUIRED ARGUMENTS:

        fieldType -- must be 'h', 'b', 'dhdt' or 'dbdt'.

        times -- Observation times. These times MUST be during the off-time.

        OUTPUTS:

        eta -- characteristic decay function evaluated at all specified times.

        """

        assert self.delt is not None, "Pulse width 'delt' must be set"
        assert fieldType in ["h", "dhdt", "b", "dbdt"], (
            "For square-pulse, fieldType must be one of 'h', 'dhdt', 'b' or 'dbdt'")
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

    def getLogUniformDecay(self, fieldType, times, chi0, dchi, tau1, tau2):

        """
        Decay function for a square-pulse waveform for log-uniform distribution
        of time-relaxation constants. The output of this function is the
        magnetization at each time for each cell, normalized by the inducing
        field.

        REQUIRED ARGUMENTS:

        fieldType -- must be 'h', 'b', 'dhdt' or 'dbdt'.

        times -- Observation times.

        chi0 -- DC (zero-frequency) magnetic susceptibility for all cells

        dchi -- DC (zero-frequency) magnetic susceptibility attributed to VRM
        for all cells

        tau1 -- Lower-bound for log-uniform distribution of time-relaxation
        constants for all cells

        tau2 -- Upper-bound for log-uniform distribution of time-relaxation
        constants for all cells

        OUTPUTS:

        eta -- characteristic decay function evaluated at all specified times.

        """

        assert self.delt is not None, "Pulse width 'delt' must be set"
        assert fieldType in ["h", "dhdt", "b", "dbdt"], (
            "For step-off, fieldType must be one of 'h', dhdt', 'b' or 'dbdt'")

        nT = len(times)
        nC = len(dchi)
        t0 = self.t0
        delt = self.delt

        times = np.kron(np.ones((nC, 1)), times)
        chi0 = np.kron(np.reshape(chi0, newshape=(nC, 1)), np.ones((1, nT)))
        dchi = np.kron(np.reshape(dchi, newshape=(nC, 1)), np.ones((1, nT)))
        tau1 = np.kron(np.reshape(tau1, newshape=(nC, 1)), np.ones((1, nT)))
        tau2 = np.kron(np.reshape(tau2, newshape=(nC, 1)), np.ones((1, nT)))

        if fieldType is "h":
            eta = (
                (np.sign(times-t0+delt) - np.sign(times-t0))*(chi0 - dchi) -
                0.5*(1+np.sign(times-t0))*(dchi/np.log(tau2/tau1)) *
                (spec.expi(-(times-t0)/tau2) -
                    spec.expi(-(times-t0)/tau1) -
                    spec.expi(-(times-t0+delt)/tau2) +
                    spec.expi(-(times-t0+delt)/tau1))
                )
        elif fieldType is "b":
            mu0 = 4*np.pi*1e-7
            eta = (
                (np.sign(times-t0+delt) - np.sign(times-t0))*(chi0 - dchi) -
                0.5*(1+np.sign(times-t0))*(dchi/np.log(tau2/tau1)) *
                (spec.expi(-(times-t0)/tau2) -
                    spec.expi(-(times-t0)/tau1) -
                    spec.expi(-(times-t0+delt)/tau2) +
                    spec.expi(-(times-t0+delt)/tau1))
                )
            eta = mu0*eta
        elif fieldType is "dhdt":
            eta = (
                0. + 0.5*(1+np.sign(times-t0))*(dchi/np.log(tau2/tau1)) *
                (np.exp(-(times-t0)/tau1) - np.exp(-(times-t0)/tau2))/(times-t0) -
                0.5*(1+np.sign(times-t0+delt))*(dchi/np.log(tau2/tau1)) *
                (np.exp(-(times-t0+delt)/tau1) - np.exp(-(times-t0+delt)/tau2))/(times-t0+delt)
                )
        elif fieldType is "dbdt":
            mu0 = 4*np.pi*1e-7
            eta = (
                0. + 0.5*(1+np.sign(times-t0))*(dchi/np.log(tau2/tau1)) *
                (np.exp(-(times-t0)/tau1) - np.exp(-(times-t0)/tau2))/(times-t0) -
                0.5*(1+np.sign(times-t0+delt))*(dchi/np.log(tau2/tau1)) *
                (np.exp(-(times-t0+delt)/tau1) - np.exp(-(times-t0+delt)/tau2))/(times-t0+delt)
                )
            eta = mu0*eta

        return eta

###################################################
#    ARBITRARY WAVEFORM UNIFORM DISCRETIZATION
###################################################


class ArbitraryDiscrete(properties.HasProperties):

    """

    """

    t_wave = properties.Array('Waveform times', dtype=float)
    I_wave = properties.Array('Waveform current', dtype=float)

    @properties.validator('t_wave')
    def _t_wave_validator(self, change):
        assert len(change['value']) > 2, "Waveform must be defined by at least 3 points"

    @properties.observer('t_wave')
    def _t_wave_observer(self, change):
        if self.I_wave is not None:
            if len(change['value']) != len(self.I_wave):
                print('Length of time vector no longer matches length of current vector')

    @properties.validator('I_wave')
    def _I_wave_validator(self, change):
        assert len(change['value']) > 2, "Waveform must be defined by at least 3 points"
        assert np.abs(change['value'][0]) < 1e-10 and np.abs(change['value'][-1]) < 1e-10, (
            "Current waveform should begin and end at 0")

    @properties.observer('I_wave')
    def _I_wave_observer(self, change):
        if self.t_wave is not None:
            if len(change['value']) != len(self.t_wave):
                print('Length of time vector no longer matches length of current vector')

    def getCharDecay(self, fieldType, times):

        """
        Characteristic decay function for arbitrary waveform. This function
        describes the decay of the VRM response for the Linear problem type.
        Note that the current will be normalized by its maximum value. The
        maximum current in the transmitter is specified in the source object.

        REQUIRD ARGUMENTS:

        fieldType -- must be 'h', 'b', 'dhdt' or 'dbdt'.

        times -- Observation times. These times MUST be during the off-time.

        OUTPUTS:

        eta -- characteristic decay function evaluated at all specified times.

        """

        assert self.t_wave is not None and self.I_wave is not None, (
            "Times and current for waveform must be set")
        assert fieldType in ["h", "dhdt", "b", "dbdt"], (
            "fieldType must be one of 'h', 'dhdt', 'b' or 'dbdt'")
        assert np.max(self.t_wave) < np.min(times), (
            "Earliest time channel must be after beginning of off-time")

        k = np.where(self.I_wave > 1e-10)
        j = k[0][0]-1
        k = k[0][-1]+1

        twave = self.t_wave[j:k+1]
        Iwave = self.I_wave[j:k+1]/np.max(np.abs(self.I_wave[j:k+1]))

        N = int(np.ceil(25*(np.max(twave)-np.min(twave))/np.min(times)))

        if N > 25000:
            N = 25000

        dt = (np.max(twave) - np.min(twave))/np.float64(N)
        tvec = np.linspace(np.min(twave), np.max(twave), N+1)

        g = np.r_[Iwave[0], np.interp(tvec[1:-1], twave, Iwave), Iwave[-1]]
        tvec = tvec[1:]

        eta = np.zeros(len(times))

        if fieldType in ["h", "b"]:
            for tt in range(0, len(eta)):
                eta[tt] = np.sum(
                    (g[1:] + (g[1:]-g[0:-1])*(times[tt]-tvec)/dt) *
                    np.log(1 + dt/(times[tt] - tvec)) - g[1:] + g[0:-1]
                    )
        elif fieldType in ["dhdt", "dbdt"]:
            for tt in range(0, len(eta)):
                eta[tt] = np.sum(
                    ((g[1:]-g[0:-1])/dt)*np.log(1 + dt/(times[tt] - tvec)) -
                    (g[1:] + (g[1:]-g[0:-1])*(times[tt]-tvec)/dt) *
                    (1/(times[tt] - tvec + dt) - 1/(times[tt] - tvec))
                    )

        if fieldType in ["b", "dbdt"]:
            mu0 = 4*np.pi*1e-7
            eta = mu0*eta

        return eta

###################################################
#    ARBITRARY WAVEFORM PIECEWISE DISCRETIZATION
###################################################


class ArbitraryPiecewise(properties.HasProperties):

    """

    """

    t_wave = properties.Array('Waveform times', dtype=float)
    I_wave = properties.Array('Waveform current', dtype=float)

    @properties.validator('t_wave')
    def _t_wave_validator(self, change):
        assert len(change['value']) > 2, "Waveform must be defined by at least 3 points"

    @properties.observer('t_wave')
    def _t_wave_observer(self, change):
        if self.I_wave is not None:
            if len(change['value']) != len(self.I_wave):
                print('Length of time vector no longer matches length of current vector')

    @properties.validator('I_wave')
    def _I_wave_validator(self, change):
        assert len(change['value']) > 2, "Waveform must be defined by at least 3 points"
        assert np.abs(change['value'][0]) < 1e-10 and np.abs(change['value'][-1]) < 1e-10, (
            "Current waveform should begin and end at 0")

    @properties.observer('I_wave')
    def _I_wave_observer(self, change):
        if self.t_wave is not None:
            if len(change['value']) != len(self.t_wave):
                print('Length of time vector no longer matches length of current vector')

    def getCharDecay(self, fieldType, times):

        """
        Characteristic decay function for arbitrary waveform. This function
        describes the decay of the VRM response for the Linear problem type.
        Note that the current will be LogUniformized by its maximum value. The
        maximum current in the transmitter is specified in the source object.

        INPUTS:

        fieldType -- must be 'h', 'b', 'dhdt' or 'dbdt'.

        times -- Observation times. These times must be during the off-time.

        OUTPUTS:

        eta -- characteristic decay function evaluated at all specified times.

        """

        assert self.t_wave is not None and self.I_wave is not None, (
            "Times and current for waveform must be set")
        assert fieldType in ["h", "dhdt", "b", "dbdt"], (
            "fieldType must be one of 'h', 'dhdt', 'b' or 'dbdt'")
        assert np.max(self.t_wave) < np.min(times), (
            "Earliest time channel must be after beginning of off-time")

        k = np.where(self.I_wave > 1e-10)
        j = k[0][0]-1
        k = k[0][-1]+1

        tvec = self.t_wave[j:k+1]
        dt = tvec[1:] - tvec[0:-1]
        g = self.I_wave[j:k+1]/np.max(np.abs(self.I_wave[j:k+1]))
        tvec = tvec[1:]

        eta = np.zeros(len(times))

        if fieldType in ["h", "b"]:
            for tt in range(0, len(eta)):
                eta[tt] = np.sum(
                    (g[1:] + (g[1:]-g[0:-1])*(times[tt]-tvec)/dt) *
                    np.log(1 + dt/(times[tt] - tvec)) - g[1:] + g[0:-1]
                    )
        elif fieldType in ["dhdt", "dbdt"]:
            for tt in range(0, len(eta)):
                eta[tt] = np.sum(
                    ((g[1:]-g[0:-1])/dt)*np.log(1 + dt/(times[tt] - tvec)) -
                    (g[1:] + (g[1:]-g[0:-1])*(times[tt]-tvec)/dt) *
                    (1/(times[tt] - tvec + dt) - 1/(times[tt] - tvec))
                    )

        if fieldType in ["b", "dbdt"]:
            mu0 = 4*np.pi*1e-7
            eta = mu0*eta

        return eta

###################################################
#               CUSTOM DECAY
###################################################


class Custom(properties.HasProperties):

    """

    """

    times = properties.Array(
        'Times at which characteristic decay function is evaluated', dtype=float)
    eta = properties.Array(
        'Characteristic decay function at evaluation times', dtype=float)

    @properties.observer('times')
    def _times_observer(self, change):
        if self.eta is not None:
            if len(change['value']) != len(self.eta):
                print('Length of time vector no longer matches length of eta vector')

    @properties.observer('eta')
    def _eta_observer(self, change):
        if self.times is not None:
            if len(change['value']) != len(self.times):
                print('Length of eta vector no longer matches length of time vector')

    def getCharDecay(self):
        """Returns characteristic decay function at specified times"""

        assert self.eta is not None, "Characteristic decay must be set"
        return self.eta
