import numpy as np
import scipy.special as spec
# import properties

###################################################
#           STEP OFF WAVEFORM
###################################################


class StepOff:
    """Characteristic decay class for step-off waveform

    Parameters
    ----------
    t0 : float
        Beginning of the off-time
    """

    # t0 = properties.Float("Start of off-time", default=0.0)

    def __init__(self, t0=0.0):

        self.t0 = t0

    @property
    def t0(self):
        """Beginning of the off-time

        Returns
        -------
        float
            Beginning of the off-time
        """
        return self._t0

    @t0.setter
    def t0(self, value):
        if isinstance(value, int):
            value = float(value)
        if not isinstance(value, float):
            raise TypeError(
                f"t0 must be a float, the value provided, {value} is "
                f"{type(value)}"
            )
        self._t0 = value
    

    def getCharDecay(self, fieldType, times):
        """Return characteristic decay for step-off waveform.

        This function defines the decay of the VRM response for the linear problem type.
        Note that the current will be normalized by its maximum value. The
        maximum current in the transmitter is specified in the source object.

        Parameters
        ----------
        fieldType : str
            Field type. Must be one of 'dhdt' or 'dbdt'. Characteristic decay for 'h'
            or 'b' CANNOT be computed for step-off
        times : numpy.ndarray
            Observation times. These times MUST be during the off-time.

        Returns
        -------
        eta : (n_times) numpy.ndarray
            Characteristic decay evaluated at all specified times.
        """

        if fieldType not in ["dhdt", "dbdt"]:
            raise NameError('For step-off, fieldType must be one of "dhdt" or "dbdt"')

        if self.t0 >= np.min(times):
            raise ValueError(
                "Earliest time channel must be after beginning of off-time (t0 = %.2e s)"
                % self.t0
            )

        t0 = self.t0

        if fieldType == "dbdt":
            mu0 = 4 * np.pi * 1e-7
            eta = -mu0 / (times - t0)
        elif fieldType == "dhdt":
            eta = -1 / (times - t0)

        return eta

    def getLogUniformDecay(self, fieldType, times, chi0, dchi, tau1, tau2):
        """Return characteristic decay for a step-off waveform for a log-uniform distribution of time-relaxation constants.

        The output of this function is the magnetization at each time for each cell, normalized by the inducing
        field.

        Parameters
        ----------
        fieldType : str
            Field type. Must be one of {'h', 'b', 'dhdt', 'dbdt'}.
        times : numpy.ndarray
            Observation times. These times MUST be during the off-time.
        chi0 : float
            DC (zero-frequency) magnetic susceptibility for all cells
        dchi : float
            DC (zero-frequency) magnetic susceptibility attributed to VRM for all cells
        tau1 : float
            Lower-bound for log-uniform distribution of time-relaxation
            constants for all cells
        tau2 : float
            Upper-bound for log-uniform distribution of time-relaxation
            constants for all cells

        Returns
        -------
        eta : (n_times) numpy.ndarray
            Characteristic decay evaluated at all specified times.
        """

        if fieldType not in ["dhdt", "dbdt"]:
            raise NameError(
                'For step-off, fieldType must be one of "dhdt" or "dbdt". Cannot be "h" or "dbdt".'
            )

        nT = len(times)
        nC = len(dchi)
        t0 = self.t0

        times = np.kron(np.ones((nC, 1)), times)
        chi0 = np.kron(np.reshape(chi0, newshape=(nC, 1)), np.ones((1, nT)))
        dchi = np.kron(np.reshape(dchi, newshape=(nC, 1)), np.ones((1, nT)))
        tau1 = np.kron(np.reshape(tau1, newshape=(nC, 1)), np.ones((1, nT)))
        tau2 = np.kron(np.reshape(tau2, newshape=(nC, 1)), np.ones((1, nT)))

        if fieldType == "h":
            eta = 0.5 * (1 - np.sign(times - t0)) * chi0 + 0.5 * (
                1 + np.sign(times - t0)
            ) * (dchi / np.log(tau2 / tau1)) * (
                spec.expi(-(times - t0) / tau2) - spec.expi(-(times - t0) / tau1)
            )
        elif fieldType == "b":
            mu0 = 4 * np.pi * 1e-7
            eta = 0.5 * (1 - np.sign(times - t0)) * chi0 + 0.5 * (
                1 + np.sign(times - t0)
            ) * (dchi / np.log(tau2 / tau1)) * (
                spec.expi(-(times - t0) / tau2) - spec.expi(-(times - t0) / tau1)
            )
            eta = mu0 * eta
        elif fieldType == "dhdt":
            eta = 0.0 + 0.5 * (1 + np.sign(times - t0)) * (
                dchi / np.log(tau2 / tau1)
            ) * (np.exp(-(times - t0) / tau1) - np.exp(-(times - t0) / tau2)) / (
                times - t0
            )
        elif fieldType == "dbdt":
            mu0 = 4 * np.pi * 1e-7
            eta = 0.0 + 0.5 * (1 + np.sign(times - t0)) * (
                dchi / np.log(tau2 / tau1)
            ) * (np.exp(-(times - t0) / tau1) - np.exp(-(times - t0) / tau2)) / (
                times - t0
            )
            eta = mu0 * eta

        return eta


###################################################
#           SQUARE PULSE WAVEFORM
###################################################


class SquarePulse(StepOff):
    """Characteristic decay class for square-pulse waveform

    Parameters
    ----------
    t0 : float
        Beginning of the off-time
    delt : float
        Pulse width
    """

    # t0 = properties.Float("Start of off-time", default=0.0)
    # delt = properties.Float("Pulse width")

    def __init__(self, delt=None, t0=0.0):
        if delt is None:
            raise AttributeError("Pulse width must be defined using 'delt'. Cannot be 'None'")

        super(SquarePulse, self).__init__(t0=t0)

        self.delt = delt

    @property
    def delt(self):
        """Square pulse on-time length

        Returns
        -------
        float
            Square pulse on-time length
        """
        return self._delt

    @delt.setter
    def delt(self, value):
        if isinstance(value, int):
            value = float(value)
        if not isinstance(value, float):
            raise TypeError(
                f"delt must be a float, the value provided, {value} is "
                f"{type(value)}"
            )
        if value <= 0.:
            raise ValueError("'delt' must be positive")
        self._delt = value

    def getCharDecay(self, fieldType, times):
        """Compute characteristic decay for a square-pulse waveform.

        This function describes the decay of the VRM response for the linear
        problem type. Note that the current will be normalized by its maximum
        value. The maximum current in the transmitter is specified in the
        source object.

        Parameters
        ----------
        fieldType : str
            Field type. Must be one of {'h', 'b', 'dhdt', 'dbdt'}.
        times : numpy.ndarray
            Observation times. These times MUST be during the off-time.

        Returns
        -------
        eta : (n_times) numpy.ndarray
            Characteristic decay evaluated at all specified times.
        """

        if self.delt is None:
            raise AssertionError("Pulse width property delt must be set.")

        if fieldType not in ["h", "b", "dhdt", "dbdt"]:
            raise NameError(
                'For square pulse, fieldType must be one of "h", "b", "dhdt" or "dbdt".'
            )

        if self.t0 >= np.min(times):
            raise ValueError(
                "Earliest time channel must be after beginning of off-time (t0 = %.2e s)"
                % self.t0
            )

        t0 = self.t0
        delt = self.delt
        mu0 = 4 * np.pi * 1e-7

        if fieldType == "h":
            eta = np.log(1 + delt / (times - t0))
        elif fieldType == "b":
            eta = mu0 * np.log(1 + delt / (times - t0))
        elif fieldType == "dhdt":
            eta = -(1 / (times - t0) - 1 / (times - t0 + delt))
        elif fieldType == "dbdt":
            eta = -mu0 * (1 / (times - t0) - 1 / (times - t0 + delt))

        return eta

    def getLogUniformDecay(self, fieldType, times, chi0, dchi, tau1, tau2):
        """Characteristic decay for a square-pulse waveform for log-uniform distribution of time-relaxation constants.

        The output of this function is the magnetization at each time for each cell, normalized by the inducing field.

        The output of this function is the magnetization at each time for each cell, normalized by the inducing
        field.

        Parameters
        ----------
        fieldType : str
            Field type. Must be one of {'h', 'b', 'dhdt', 'dbdt'}.
        times : numpy.ndarray
            Observation times. These times MUST be during the off-time.
        chi0 : float
            DC (zero-frequency) magnetic susceptibility for all cells
        dchi : float
            DC (zero-frequency) magnetic susceptibility attributed to VRM for all cells
        tau1 : float
            Lower-bound for log-uniform distribution of time-relaxation
            constants for all cells
        tau2 : float
            Upper-bound for log-uniform distribution of time-relaxation
            constants for all cells

        Returns
        -------
        eta : (n_times) numpy.ndarray
            Characteristic decay evaluated at all specified times.
        """

        if self.delt is None:
            raise AssertionError("Pulse width property delt must be set.")

        if fieldType not in ["h", "b", "dhdt", "dbdt"]:
            raise NameError(
                'For square pulse, fieldType must be one of "h", "b", "dhdt" or "dbdt".'
            )

        nT = len(times)
        nC = len(dchi)
        t0 = self.t0
        delt = self.delt

        times = np.kron(np.ones((nC, 1)), times)
        chi0 = np.kron(np.reshape(chi0, newshape=(nC, 1)), np.ones((1, nT)))
        dchi = np.kron(np.reshape(dchi, newshape=(nC, 1)), np.ones((1, nT)))
        tau1 = np.kron(np.reshape(tau1, newshape=(nC, 1)), np.ones((1, nT)))
        tau2 = np.kron(np.reshape(tau2, newshape=(nC, 1)), np.ones((1, nT)))

        if fieldType == "h":
            eta = (np.sign(times - t0 + delt) - np.sign(times - t0)) * (
                chi0 - dchi
            ) - 0.5 * (1 + np.sign(times - t0)) * (dchi / np.log(tau2 / tau1)) * (
                spec.expi(-(times - t0) / tau2)
                - spec.expi(-(times - t0) / tau1)
                - spec.expi(-(times - t0 + delt) / tau2)
                + spec.expi(-(times - t0 + delt) / tau1)
            )
        elif fieldType == "b":
            mu0 = 4 * np.pi * 1e-7
            eta = (np.sign(times - t0 + delt) - np.sign(times - t0)) * (
                chi0 - dchi
            ) - 0.5 * (1 + np.sign(times - t0)) * (dchi / np.log(tau2 / tau1)) * (
                spec.expi(-(times - t0) / tau2)
                - spec.expi(-(times - t0) / tau1)
                - spec.expi(-(times - t0 + delt) / tau2)
                + spec.expi(-(times - t0 + delt) / tau1)
            )
            eta = mu0 * eta
        elif fieldType == "dhdt":
            eta = (
                0.0
                + 0.5
                * (1 + np.sign(times - t0))
                * (dchi / np.log(tau2 / tau1))
                * (np.exp(-(times - t0) / tau1) - np.exp(-(times - t0) / tau2))
                / (times - t0)
                - 0.5
                * (1 + np.sign(times - t0 + delt))
                * (dchi / np.log(tau2 / tau1))
                * (
                    np.exp(-(times - t0 + delt) / tau1)
                    - np.exp(-(times - t0 + delt) / tau2)
                )
                / (times - t0 + delt)
            )
        elif fieldType == "dbdt":
            mu0 = 4 * np.pi * 1e-7
            eta = (
                0.0
                + 0.5
                * (1 + np.sign(times - t0))
                * (dchi / np.log(tau2 / tau1))
                * (np.exp(-(times - t0) / tau1) - np.exp(-(times - t0) / tau2))
                / (times - t0)
                - 0.5
                * (1 + np.sign(times - t0 + delt))
                * (dchi / np.log(tau2 / tau1))
                * (
                    np.exp(-(times - t0 + delt) / tau1)
                    - np.exp(-(times - t0 + delt) / tau2)
                )
                / (times - t0 + delt)
            )
            eta = mu0 * eta

        return eta


###################################################
#    ARBITRARY WAVEFORM UNIFORM DISCRETIZATION
###################################################


class ArbitraryDiscrete:
    """Characteristic decay for arbitrary discrete waveform

    This class is used to approximate an arbitrary waveform as a set of square-pulse waveforms;
    for which we have a solution to the characteristic decay.

    Parameters
    ----------
    t_wave : numpy.ndarray
        Waveform on-times
    I_wave : numpy.ndarray
        Waveform on-time currents
    """

    def __init__(self, t_wave=None, I_wave=None):
        if (t_wave is None) | (I_wave is None):
            raise AttributeError("Must instantiate with 't_wave' and 'I_wave'. Cannot be 'None'")

        try:
            if len(t_wave) != len(I_wave):
                raise ValueError("'t_wave' and 'I_wave' must have the same length.")
        except:
            raise TypeError("'t_wave' and 'I_wave' must be 1D array-like")

        self.t_wave = t_wave
        self.I_wave = I_wave

    # t_wave = properties.Array("Waveform times", dtype=float)
    # I_wave = properties.Array("Waveform current", dtype=float)

    # @properties.validator("t_wave")
    # def _t_wave_validator(self, change):

    #     if len(change["value"]) < 3:
    #         ValueError("Waveform must be defined by at least 3 points.")

    #     if self.I_wave is not None:
    #         if len(change["value"]) != len(self.I_wave):
    #             print(
    #                 "Length of time vector no longer matches length of current vector"
    #             )

    # @properties.validator("I_wave")
    # def _I_wave_validator(self, change):

    #     if len(change["value"]) < 3:
    #         ValueError("Waveform must be defined by at least 3 points.")

    #     if (np.abs(change["value"][0]) > 1e-10) | (np.abs(change["value"][-1]) > 1e-10):
    #         raise ValueError(
    #             "Current waveform should begin and end with amplitude of 0. Right now I_1 = {0:.2e} and I_end = {1:.2e}".format(
    #                 change["value"][0], change["value"][-1]
    #             )
    #         )

    #     if self.t_wave is not None:
    #         if len(change["value"]) != len(self.t_wave):
    #             print(
    #                 "Length of time vector no longer matches length of current vector"
    #             )

    @property
    def t_wave(self):
        """Waveform on-times

        Returns
        -------
        numpy.ndarray
            Waveform on-times
        """
        return self._t_wave

    @t_wave.setter
    def t_wave(self, value):
        try:
            value = np.atleast_1d(value).astype(float)
        except:
            raise TypeError(f"t_wave is not a valid type. Got {type(value)}")
        
        if value.ndim > 1:
            raise TypeError("t_wave must be ('*') array")

        if getattr(self, 'I_wave', None) is not None:
            if len(value) == len(self._I_wave):
                self._t_wave = value
            else:
                raise ValueError("'t_wave' and 'I_wave' must be the same length")
        else:
            self._t_wave = value

    @property
    def I_wave(self):
        """Waveform on-time currents

        Returns
        -------
        numpy.ndarray
            Waveform on-time currents
        """
        return self._I_wave

    @I_wave.setter
    def I_wave(self, value):
        try:
            value = np.atleast_1d(value).astype(float)
        except:
            raise TypeError("I_wave is not a valid type. Got {type(value)}")
        
        if value.ndim > 1:
            raise TypeError("I_wave must be ('*') array")

        if getattr(self, 't_wave', None) is not None:
            if len(value) == len(self._t_wave):
                self._I_wave = value
            else:
                raise ValueError("'t_wave' and 'I_wave' must be the same length")
        else:
            self._I_wave = value

    def getCharDecay(self, fieldType, times):
        """Compute characteristic decay for arbitrary waveform.

        This function describes the decay of the VRM response for the Linear problem type.
        Note that the current will be normalized by its maximum value. The
        maximum current in the transmitter is specified in the source object.

        Parameters
        ----------
        fieldType : str
            Field type. Must be one of {'h', 'b', 'dhdt', 'dbdt'}.
        times : numpy.ndarray
            Observation times. These times MUST be during the off-time.

        Returns
        -------
        eta : (n_times) numpy.ndarray
            Characteristic decay evaluated at all specified times.
        """

        if self.t_wave is None:
            raise AssertionError("Waveform times (Property: t_wave) are not set.")

        if self.I_wave is None:
            raise AssertionError("Waveform current (Property: I_wave) is not set.")

        if fieldType not in ["h", "b", "dhdt", "dbdt"]:
            raise NameError(
                'For square pulse, fieldType must be one of "h", "b", "dhdt" or "dbdt".'
            )

        if len(self.t_wave) != len(self.I_wave):
            raise ValueError(
                "Length of t_wave and I_wave properties must be the same. Currently len(t_wave) = {0: i} and len(I_wave) = {1: i}".format(
                    self.t_wave, self.I_wave
                )
            )

        k = np.where(self.I_wave > 1e-10)
        j = k[0][0] - 1
        k = k[0][-1] + 1

        twave = self.t_wave[j : k + 1]
        Iwave = self.I_wave[j : k + 1] / np.max(np.abs(self.I_wave[j : k + 1]))

        n_pts = int(np.ceil(25 * (np.max(twave) - np.min(twave)) / np.min(times)))

        if n_pts > 25000:
            n_pts = 25000

        dt = (np.max(twave) - np.min(twave)) / np.float64(n_pts)
        tvec = np.linspace(np.min(twave), np.max(twave), n_pts + 1)

        g = np.r_[Iwave[0], np.interp(tvec[1:-1], twave, Iwave), Iwave[-1]]
        tvec = tvec[1:]

        eta = np.zeros(len(times))

        if fieldType in ["h", "b"]:
            for tt in range(0, len(eta)):
                eta[tt] = np.sum(
                    (g[1:] + (g[1:] - g[0:-1]) * (times[tt] - tvec) / dt)
                    * np.log(1 + dt / (times[tt] - tvec))
                    - g[1:]
                    + g[0:-1]
                )
        elif fieldType in ["dhdt", "dbdt"]:
            for tt in range(0, len(eta)):
                eta[tt] = np.sum(
                    ((g[1:] - g[0:-1]) / dt) * np.log(1 + dt / (times[tt] - tvec))
                    - (g[1:] + (g[1:] - g[0:-1]) * (times[tt] - tvec) / dt)
                    * (1 / (times[tt] - tvec + dt) - 1 / (times[tt] - tvec))
                )

        if fieldType in ["b", "dbdt"]:
            mu0 = 4 * np.pi * 1e-7
            eta = mu0 * eta

        return eta


###################################################
#    ARBITRARY WAVEFORM PIECEWISE DISCRETIZATION
###################################################


class ArbitraryPiecewise(ArbitraryDiscrete):
    """Characteristic decay for arbitrary piecewise waveform

    This class is used to approximate an arbitrary waveform using a piecewise linear approximation;
    for which we have a solution to the characteristic decay.

    Parameters
    ----------
    t_wave : numpy.ndarray
        Waveform on-times
    I_wave : numpy.ndarray
        Waveform on-time currents
    """

    def __init__(self, t_wave=None, I_wave=None):
        super(ArbitraryPiecewise, self).__init__(t_wave=t_wave, I_wave=I_wave)

    # t_wave = properties.Array("Waveform times", dtype=float)
    # I_wave = properties.Array("Waveform current", dtype=float)

    # @properties.validator("t_wave")
    # def _t_wave_validator(self, change):
    #     if len(change["value"]) < 3:
    #         ValueError("Waveform must be defined by at least 3 points.")

    # @properties.observer("t_wave")
    # def _t_wave_observer(self, change):
    #     if self.I_wave is not None:
    #         if len(change["value"]) != len(self.I_wave):
    #             print(
    #                 "Length of time vector no longer matches length of current vector"
    #             )

    # @properties.validator("I_wave")
    # def _I_wave_validator(self, change):
    #     if len(change["value"]) < 3:
    #         ValueError("Waveform must be defined by at least 3 points.")

    #     if (np.abs(change["value"][0]) > 1e-10) | (np.abs(change["value"][-1]) > 1e-10):
    #         raise ValueError(
    #             "Current waveform should begin and end with amplitude of 0. Right now I_1 = {0:.2e} and I_end = {1:.2e}".format(
    #                 change["value"][0], change["value"][-1]
    #             )
    #         )

    # @properties.observer("I_wave")
    # def _I_wave_observer(self, change):
    #     if self.t_wave is not None:
    #         if len(change["value"]) != len(self.t_wave):
    #             print(
    #                 "Length of time vector no longer matches length of current vector"
    #             )

    def getCharDecay(self, fieldType, times):
        """Compute characteristic decay function for arbitrary waveform.

        This function describes the decay of the VRM response for the Linear problem type.
        Note that the current will be LogUniformized by its maximum value. The
        maximum current in the transmitter is specified in the source object.

        Parameters
        ----------
        fieldType : str
            Field type. Must be one of {'h', 'b', 'dhdt', 'dbdt'}.
        times : numpy.ndarray
            Observation times. These times MUST be during the off-time.

        Returns
        -------
        eta : (n_times) numpy.ndarray
            Characteristic decay evaluated at all specified times.
        """

        if self.t_wave is None:
            raise AssertionError("Waveform times (Property: t_wave) are not set.")

        if self.I_wave is None:
            raise AssertionError("Waveform current (Property: I_wave) is not set.")

        if fieldType not in ["h", "b", "dhdt", "dbdt"]:
            raise NameError(
                'For square pulse, fieldType must be one of "h", "b", "dhdt" or "dbdt".'
            )

        if np.max(self.t_wave) >= np.min(times):
            raise ValueError(
                "Earliest time channel must be after beginning of off-time (t0 = %.2e s)"
                % np.max(self.t_wave)
            )

        k = np.where(self.I_wave > 1e-10)
        j = k[0][0] - 1
        k = k[0][-1] + 1

        tvec = self.t_wave[j : k + 1]
        dt = tvec[1:] - tvec[0:-1]
        g = self.I_wave[j : k + 1] / np.max(np.abs(self.I_wave[j : k + 1]))
        tvec = tvec[1:]

        eta = np.zeros(len(times))

        if fieldType in ["h", "b"]:
            for tt in range(0, len(eta)):
                eta[tt] = np.sum(
                    (g[1:] + (g[1:] - g[0:-1]) * (times[tt] - tvec) / dt)
                    * np.log(1 + dt / (times[tt] - tvec))
                    - g[1:]
                    + g[0:-1]
                )
        elif fieldType in ["dhdt", "dbdt"]:
            for tt in range(0, len(eta)):
                eta[tt] = np.sum(
                    ((g[1:] - g[0:-1]) / dt) * np.log(1 + dt / (times[tt] - tvec))
                    - (g[1:] + (g[1:] - g[0:-1]) * (times[tt] - tvec) / dt)
                    * (1 / (times[tt] - tvec + dt) - 1 / (times[tt] - tvec))
                )

        if fieldType in ["b", "dbdt"]:
            mu0 = 4 * np.pi * 1e-7
            eta = mu0 * eta

        return eta


###################################################
#               CUSTOM DECAY
###################################################


class Custom:
    """Define characteristic decay with function handle

    Parameters
    ----------
    waveform_function : function
        Function handle defining the characteristic decay as a function of time
    """

    def __init__(self, waveform_function):
        self.waveform_function = waveform_function

    @property
    def waveform_function(self):
        """Function handle for characteristic decay

        Returns
        -------
        function
            A function that returns the characteristic decay as a
            function of *times*.
        """
        return self._waveform_function

    @waveform_function.setter
    def waveform_function(self, value):
        if not callable(value):
            raise ValueError(
                "waveform_function must be a function. The input value is type: "
                f"{type(value)}"
            )
        self._waveform_function = value


    def getCharDecay(self, times):
        """Returns characteristic decay function at specified times

        Parameters
        ----------
        times : numpy.ndarray
            Off-times where characteristic decay is evaluated
        """

        return self.waveform_function(times)
