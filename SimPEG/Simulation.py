from __future__ import print_function

import numpy as np
import discretize
import properties

from . import Utils
from . import Models
from . import Maps
from . import Props
# from . import Source
from .Data import SyntheticData
from .NewSurvey import BaseSurvey

__all__ = ['LinearSimulation', 'ExponentialSinusoidSimulation']


##############################################################################
#                                                                            #
#                             Custom Properties                              #
#                                                                            #
##############################################################################

class TimeStepArray(properties.Array):

    class_info = "an array or list of tuples specifying the mesh tensor"

    def validate(self, instance, value):
        if isinstance(value, list):
            value = discretize.utils.meshTensor(value)
        return super(TimeStepArray, self).validate(instance, value)


##############################################################################
#                                                                            #
#                       Simulation Base Classes                              #
#                                                                            #
##############################################################################

class BaseSimulation(Props.HasModel):
    """
    BaseSimulation is the base class for all geophysical forward simulations in
    SimPEG.
    """

    _REGISTRY = {}

    mesh = properties.Instance(
        "a discretize mesh instance",
        discretize.BaseMesh
    )

    survey = properties.Instance(
        "a list of sources",
        BaseSurvey,
        default=BaseSurvey()
    )

    counter = properties.Instance(
        "A SimPEG.Utils.Counter object",
        Utils.Counter
    )

    # TODO: Solver code needs to be cleaned up so this is either a pymatsolver
    # solver or a SimPEG solver (or similar)
    solver = Utils.SolverUtils.Solver

    solver_opts = properties.Instance(
        "solver options as a kwarg dict",
        dict,
        default={}
    )

    @properties.observer('mesh')
    def _update_registry(self, change):
        self._REGISTRY.update(change['value']._REGISTRY)

    def fields(self, m):
        """
        u = fields(m)

        The field given the model.

        :param numpy.array m: model
        :rtype: numpy.array
        :return: u, the fields
        """
        raise NotImplementedError(
            "fields has not been implemented for this "
        )

    def dpred(self, m=None, f=None):
        """
        dpred(m, f=None)

        Create the projected data from a model.
        The fields, f, (if provided) will be used for the predicted data
        instead of recalculating the fields (which may be expensive!).

        .. math::

            d_\\text{pred} = P(f(m))

        Where P is a projection of the fields onto the data space.
        """
        raise NotImplementedError('dpred is not yet implemented')

    @Utils.timeIt
    def Jvec(self, m, v, f=None):
        """
        Jv = Jvec(m, v, f=None)

        Effect of J(m) on a vector v.

        :param numpy.array m: model
        :param numpy.array v: vector to multiply
        :param Fields f: fields
        :rtype: numpy.array
        :return: Jv
        """
        raise NotImplementedError('Jvec is not yet implemented.')

    @Utils.timeIt
    def Jtvec(self, m, v, f=None):
        """
        Jtv = Jtvec(m, v, f=None)

        Effect of transpose of J(m) on a vector v.

        :param numpy.array m: model
        :param numpy.array v: vector to multiply
        :param Fields f: fields
        :rtype: numpy.array
        :return: JTv
        """
        raise NotImplementedError('Jt is not yet implemented.')

    @Utils.timeIt
    def Jvec_approx(self, m, v, f=None):
        """Jvec_approx(m, v, f=None)

        Approximate effect of J(m) on a vector v

        :param numpy.array m: model
        :param numpy.array v: vector to multiply
        :param Fields f: fields
        :rtype: numpy.array
        :return: approxJv
        """
        return self.Jvec(m, v, f)

    @Utils.timeIt
    def Jtvec_approx(self, m, v, f=None):
        """Jtvec_approx(m, v, f=None)

        Approximate effect of transpose of J(m) on a vector v.

        :param numpy.array m: model
        :param numpy.array v: vector to multiply
        :param Fields f: fields
        :rtype: numpy.array
        :return: JTv
        """
        return self.Jtvec(m, v, f)

    @Utils.count
    def residual(self, m, dobs, f=None):
        """residual(m, dobs, f=None)

            :param numpy.array m: geophysical model
            :param numpy.array f: fields
            :rtype: numpy.array
            :return: data residual

            The data residual:

            .. math::

                \mu_\\text{data} = \mathbf{d}_\\text{pred} - \mathbf{d}_\\text{obs}

        """
        return Utils.mkvc(self.dpred(m, f=f) - dobs)

    def make_synthetic_data(self, m, standard_deviation=0.05, f=None):
        """
        Make synthetic data given a model, and a standard deviation.

        :param numpy.array m: geophysical model
        :param numpy.array standard_deviation: standard deviation
        :param numpy.array f: fields for the given model (if pre-calculated)
        """

        dclean = self.dpred(m, f=f)
        noise = standard_deviation*abs(dclean)*np.random.randn(*dclean.shape)
        dobs = dclean + noise

        return SyntheticData(
            dobs=dobs,
            dclean=dclean,
            survey=self.survey,
            standard_deviation=standard_deviation
        )


class BaseTimeSimulation(BaseSimulation):
    """
    Base class for a time domain simulation
    """

    time_steps = TimeStepArray(
        """
        Sets/gets the timeSteps for the time domain simulation.

        You can set as an array of dt's or as a list of tuples/floats.
        Tuples must be length two with [..., (dt, repeat), ...]

        For example, the following setters are the same::

            sim.timeSteps = [(1e-6, 3), 1e-5, (1e-4, 2)]
            sim.timeSteps = np.r_[1e-6,1e-6,1e-6,1e-5,1e-4,1e-4]

        """,
        dtype=float
    )

    t0 = properties.Float(
        "Origin of the time discretization",
        default=0.0
    )

    def __init__(self, **kwargs):
        super(BaseTimeSimulation, self).__init__(**kwargs)

    @properties.observer('time_steps')
    def _remove_time_mesh_on_time_step_update(self, change):
        del self.time_mesh

    @properties.observer('t0')
    def _remove_time_mesh_on_t0_update(self, change):
        del self.time_mesh

    @property
    def time_mesh(self):
        if getattr(self, '_time_mesh', None) is None:
            self._time_mesh = discretize.TensorMesh(
                [self.time_steps], x0=[self.t0]
            )
        return self._time_mesh

    @time_mesh.deleter
    def time_mesh(self):
        if hasattr(self, '_time_mesh'):
            del self._time_mesh

    @property
    def nT(self):
        return self.time_mesh.nC

    @property
    def times(self):
        "Modeling times"
        return self.time_mesh.vectorNx


##############################################################################
#                                                                            #
#                           Linear Simulation                                #
#                                                                            #
##############################################################################

class LinearSimulation(BaseSimulation):
    """
    Class for a linear simulation of the form

    .. math::

        d = Gm

    where :math:`d` is a vector of the data, `G` is the simulation matrix and
    :math:`m` is the model.

    Inherit this class to build a linear simulatio.
    """

    linear_model, model_map, model_deriv = Props.Invertible(
        "The model for a linear problem"
    )

    def __init__(self, **kwargs):
        super(LinearSimulation, self).__init__(**kwargs)

    @property
    def G(self):
        raise NotImplementedError('G must be implemented for this simulation')

    def fields(self, m):
        self.model = m
        return self.G.dot(self.model)

    def dpred(self, m=None, f=None):
        self.model = m
        if f is not None:
            return f
        return self.fields(self.model)

    def getJ(self, m, f=None):
        self.model = m
        return self.G.dot(self.model_deriv)

    def Jvec(self, m, v, f=None):
        self.model = m
        return self.G.dot(self.model_deriv * v)

    def Jtvec(self, m, v, f=None):
        self.model = m
        return self.model_deriv.T * self.G.T.dot(v)


class ExponentialSinusoidSimulation(LinearSimulation):
    """
    This is the simulation class for the linear problem consisting of
    exponentially decaying sinusoids
    """
    n_kernels = properties.Integer(
        "number of kernels defining the linear problem",
        default = 20
    )

    p = properties.Float(
        "rate of exponential decay of the kernel",
        default=-0.25
    )

    q = properties.Float(
        "rate of oscillation of the kernel",
        default = 0.25
    )

    def __init__(self, **kwargs):
        super(ExponentialSinusoidSimulation, self).__init__(**kwargs)

    @property
    def _jk(self):
        return np.linspace(1., 60., self.n_kernels)

    def _g(self, k):
        return (
            np.exp(self.p*self._jk[k]*self.mesh.vectorCCx) *
            np.cos(np.pi*self.q*self._jk[k]*self.mesh.vectorCCx)
        )

    @property
    def G(self):
        if getattr(self, '_G', None) is None:
            G = np.empty((self.n_kernels, self.mesh.nC))

            for i in range(self.n_kernels):
                G[i, :] = self._g(i)

            self._G = G
        return self._G

