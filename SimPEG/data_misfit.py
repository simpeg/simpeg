import numpy as np
from .utils import Counter, sdiag, timeIt, Identity, validate_type
from .data import Data
from .simulation import BaseSimulation
from .objective_function import L2ObjectiveFunction

__all__ = ["L2DataMisfit"]


class BaseDataMisfit(L2ObjectiveFunction):
    """
    BaseDataMisfit

    .. note::
        You should inherit from this class to create your own data misfit
        term.
    """

    def __init__(self, data, simulation, debug=False, counter=None, **kwargs):
        super().__init__(**kwargs)

        self.data = data
        self.simulation = simulation
        self.debug = debug
        self.count = counter

    @property
    def data(self):
        """A SimPEG data class containing the observed data.

        Returns
        -------
        SimPEG.data.Data
        """
        return self._data

    @data.setter
    def data(self, value):
        self._data = validate_type("data", value, Data, cast=False)

    @property
    def simulation(self):
        """A SimPEG simulation.

        Returns
        -------
        SimPEG.simulation.BaseSimulation
        """
        return self._simulation

    @simulation.setter
    def simulation(self, value):
        self._simulation = validate_type(
            "simulation", value, BaseSimulation, cast=False
        )

    @property
    def debug(self):
        """Print debugging information.

        Returns
        -------
        bool
        """
        return self._debug

    @debug.setter
    def debug(self, value):
        self._debug = validate_type("debug", value, bool)

    @property
    def counter(self):
        """Set this to a ``SimPEG.utils.Counter`` if you want to count things.

        Returns
        -------
        SimPEG.utils.Counter or None
        """
        return self._counter

    @counter.setter
    def counter(self, value):
        if value is not None:
            value = validate_type("counter", value, Counter, cast=False)

    @property
    def nP(self):
        """
        number of model parameters
        """
        if self._mapping is not None:
            return self.mapping.nP
        elif self.simulation.model is not None:
            return len(self.simulation.model)
        else:
            return "*"

    @property
    def nD(self):
        """
        number of data
        """
        return self.data.nD

    @property
    def shape(self):
        """"""
        return (self.nD, self.nP)

    @property
    def W(self):
        """W
        The data weighting matrix.
        The default is based on the norm of the data plus a noise floor.
        :rtype: scipy.sparse.csr_matrix
        :return: W
        """

        if getattr(self, "_W", None) is None:
            if self.data is None:
                raise Exception(
                    "data with standard deviations must be set before the data "
                    "misfit can be constructed. Please set the data: "
                    "dmis.data = Data(dobs=dobs, relative_error=rel"
                    ", noise_floor=eps)"
                )
            standard_deviation = self.data.standard_deviation
            if standard_deviation is None:
                raise Exception(
                    "data standard deviations must be set before the data misfit "
                    "can be constructed (data.relative_error = 0.05, "
                    "data.noise_floor = 1e-5), alternatively, the W matrix "
                    "can be set directly (dmisfit.W = 1./standard_deviation)"
                )
            if any(standard_deviation <= 0):
                raise Exception(
                    "data.standard_deviation must be strictly positive to construct "
                    "the W matrix. Please set data.relative_error and or "
                    "data.noise_floor."
                )
            self._W = sdiag(1 / (standard_deviation))
        return self._W

    @W.setter
    def W(self, value):
        if isinstance(value, Identity):
            value = np.ones(self.data.nD)
        if len(value.shape) < 2:
            value = sdiag(value)
        assert value.shape == (
            self.data.nD,
            self.data.nD,
        ), "W must have shape ({nD},{nD}), not ({val0}, val{1})".format(
            nD=self.data.nD, val0=value.shape[0], val1=value.shape[1]
        )
        self._W = value

    def residual(self, m, f=None):
        if self.data is None:
            raise Exception("data must be set before a residual can be calculated.")
        return self.simulation.residual(m, self.data.dobs, f=f)


class L2DataMisfit(BaseDataMisfit):
    """
    The data misfit with an l_2 norm:

    .. math::

        \mu_\\text{data} = {1\over 2}\left|
        \mathbf{W}_d (\mathbf{d}_\\text{pred} -
        \mathbf{d}_\\text{obs}) \\right|_2^2
    """

    @timeIt
    def __call__(self, m, f=None):
        "__call__(m, f=None)"

        R = self.W * self.residual(m, f=f)
        return 0.5 * np.vdot(R, R)

    @timeIt
    def deriv(self, m, f=None):
        """
        deriv(m, f=None)
        Derivative of the data misfit

        .. math::

            \mathbf{J}^{\top} \mathbf{W}^{\top} \mathbf{W}
            (\mathbf{d} - \mathbf{d}^{obs})

        :param numpy.ndarray m: model
        :param SimPEG.fields.Fields f: fields object
        """

        if f is None:
            f = self.simulation.fields(m)

        return self.simulation.Jtvec(
            m, self.W.T * (self.W * self.residual(m, f=f)), f=f
        )

    @timeIt
    def deriv2(self, m, v, f=None):
        """
        deriv2(m, v, f=None)

        .. math::

            \mathbf{J}^{\top} \mathbf{W}^{\top} \mathbf{W} \mathbf{J}

        :param numpy.ndarray m: model
        :param numpy.ndarray v: vector
        :param SimPEG.fields.Fields f: fields object
        """

        if f is None:
            f = self.simulation.fields(m)

        return self.simulation.Jtvec_approx(
            m, self.W * (self.W * self.simulation.Jvec_approx(m, v, f=f)), f=f
        )
