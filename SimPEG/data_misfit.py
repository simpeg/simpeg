import numpy as np
import properties

from .utils import Counter, sdiag, timeIt
from .data import Data
from .simulation import BaseSimulation
from .objective_function import L2ObjectiveFunction


class BaseDataMisfit(L2ObjectiveFunction):
    """
    BaseDataMisfit
        .. note::
            You should inherit from this class to create your own data misfit
            term.
    """

    data = properties.Instance(
        "A SimPEG data class containing the observed data",
        Data,
        required=True
    )

    simulation = properties.Instance(
        "A SimPEG simulation",
        BaseSimulation,
        required=True
    )

    debug = properties.Bool(
        "Print debugging information",
        default=False
    )

    counter = properties.Instance(
        "Set this to a SimPEG.Utils.Counter() if you want to count things",
        Counter
    )

    _has_fields = properties.Bool(
        "Data Misfits take fields, handy to store them",
        default=True
    )

    def __init__(self, simulation=None, **kwargs):
        if simulation is not None:
            kwargs['simulation'] = simulation
        super(BaseDataMisfit, self).__init__(**kwargs)

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
            return '*'

    @property
    def nD(self):
        """
        number of data
        """
        return self.data.nD

    @property
    def shape(self):
        """
        """
        return (self.nD, self.nP)

    @property
    def Wd(self):
        raise AttributeError(
            'The `Wd` property been depreciated, please use: `W` instead'
        )

    @property
    def W(self):
        """W
            The data weighting matrix.
            The default is based on the norm of the data plus a noise floor.
            :rtype: scipy.sparse.csr_matrix
            :return: W
        """

        if getattr(self, '_W', None) is None:
            uncertainty = self.data.uncertainty
            if uncertainty is None:
                raise Exception(
                    "data uncertainties must be set before the data misfit "
                    "can be constructed (data.standard_deviation = 0.05, "
                    "data.noise_floor = 1e-5), alternatively, the W matrix "
                    "can be set directly (dmisfit.W = 1./uncertainty)"
                )
            self._W = sdiag(1/(uncertainty))
        return self._W

    @W.setter
    def W(self, value):
        if len(value.shape) < 2:
            value = sdiag(value)
        assert value.shape == (self.data.nD, self.data.nD), (
            'W must have shape ({nD},{nD}), not ({val0}, val{1})'.format(
                nD=self.data.nD, val0=value.shape[0], val1=value.shape[1]
            )
        )
        self._W = value

    def residual(self, m, f=None):
        return self.simulation.residual(m, self.data.dobs, f=f)

    @property
    def std(self):
        raise Exception(
            "L2DataMisfit no longer has the attribute 'std'. Please use "
            "data.standard_deviation"
        )


class L2DataMisfit(BaseDataMisfit):
    """
    The data misfit with an l_2 norm:

    .. math::
        \mu_\\text{data} = {1\over 2}\left|
        \mathbf{W}_d (\mathbf{d}_\\text{pred} -
        \mathbf{d}_\\text{obs}) \\right|_2^2
    """

    def __init__(self, **kwargs):
        BaseDataMisfit.__init__(self, **kwargs)

    @timeIt
    def __call__(self, m, f=None):
        "__call__(m, f=None)"
        R = self.W * self.residual(m, f=f)
        return 0.5*np.vdot(R, R)

    @timeIt
    def deriv(self, m, f=None):
        """
        deriv(m, f=None)
        Derivative of the data misfit
        .. math::
            \mathbf{J}^{\top} \mathbf{W}^{\top} \mathbf{W}
            (\mathbf{d} - \mathbf{d}^{obs})
        :param numpy.ndarray m: model
        :param SimPEG.Fields.Fields f: fields object
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
        :param SimPEG.Fields.Fields f: fields object
        """
        if f is None:
            f = self.simulation.fields(m)
        return self.simulation.Jtvec_approx(
            m, self.W * (self.W * self.simulation.Jvec_approx(m, v, f=f)), f=f
        )

class l2_DataMisfit(L2DataMisfit):
    """
    This class will be deprecated in the next release of SimPEG. Please use
    `L2DataMisfit` instead.
    """
    def __init__(self, **kwargs):
        warnings.warn(
            "l2_DataMisfit has been depreciated in favor of L2DataMisfit. Please "
            "update your code to use 'L2DataMisfit'", DeprecationWarning
        )
        super(l2_DataMisfit, self).__init__(**kwargs)
