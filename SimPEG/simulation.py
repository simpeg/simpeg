from __future__ import print_function

import os
import inspect
import numpy as np
import warnings

from discretize.base import BaseMesh
from discretize import TensorMesh
from discretize.utils import unpack_widths

from . import props
from .data import SyntheticData, Data
from .survey import BaseSurvey
from .utils import (
    Counter,
    timeIt,
    count,
    mkvc,
    validate_ndarray_with_shape,
    validate_float,
    validate_type,
    validate_string,
    validate_integer,
)

try:
    from pymatsolver import Pardiso as DefaultSolver
except ImportError:
    from .utils.solver_utils import SolverLU as DefaultSolver

__all__ = ["LinearSimulation", "ExponentialSinusoidSimulation"]


##############################################################################
#                                                                            #
#                       Simulation Base Classes                              #
#                                                                            #
##############################################################################


class BaseSimulation(props.HasModel):
    """
    BaseSimulation is the base class for all geophysical forward simulations in
    SimPEG.
    """

    ###########################################################################
    # Properties

    _REGISTRY = {}

    @property
    def mesh(self):
        """Discretize mesh for the simulation

        Returns
        -------
        discretize.base.BaseMesh
        """
        return self._mesh

    @mesh.setter
    def mesh(self, value):
        if value is not None:
            value = validate_type("mesh", value, BaseMesh, cast=False)
        self._mesh = value

    @property
    def survey(self):
        """The survey for the simulation.

        Returns
        -------
        SimPEG.survey.BaseSurvey
        """
        return self._survey

    @survey.setter
    def survey(self, value):
        if value is not None:
            value = validate_type("survey", value, BaseSurvey, cast=False)
        self._survey = value

    @property
    def counter(self):
        """The counter.

        Returns
        -------
        None or SimPEG.utils.Counter
        """
        return self._counter

    @counter.setter
    def counter(self, value):
        if value is not None:
            value = validate_type("counter", value, Counter, cast=False)
        self._counter = value

    @property
    def sensitivity_path(self):
        """Path to store the sensitivty.

        Returns
        -------
        str
        """
        return self._sensitivity_path

    @sensitivity_path.setter
    def sensitivity_path(self, value):
        self._sensitivity_path = validate_string("sensitivity_path", value)

    @property
    def solver(self):
        """Linear algebra solver (e.g. from pymatsolver).

        Returns
        -------
        class
            A solver class that, when instantiated allows a multiplication with the
            returned object.
        """
        return self._solver

    @solver.setter
    def solver(self, cls):
        if cls is not None:
            if not inspect.isclass(cls):
                raise TypeError(f"solver must be a class, not a {type(cls)}")
            if not hasattr(cls, "__mul__"):
                raise TypeError("solver must support the multiplication operator, `*`.")
        self._solver = cls

    @property
    def solver_opts(self):
        """Options passed to the `solver` class on initialization.

        Returns
        -------
        dict
            Passed as keyword arguments to the solver.
        """
        return self._solver_opts

    @solver_opts.setter
    def solver_opts(self, value):
        self._solver_opts = validate_type("solver_opts", value, dict, cast=False)

    @property
    def verbose(self):
        """Verbosity flag.

        Returns
        -------
        bool
        """
        return self._verbose

    @verbose.setter
    def verbose(self, value):
        self._verbose = validate_type("verbose", value, bool)

    ###########################################################################
    # Instantiation

    def __init__(
        self,
        mesh=None,
        survey=None,
        solver=None,
        solver_opts=None,
        sensitivity_path=os.path.join(".", "sensitivity"),
        counter=None,
        verbose=False,
        **kwargs,
    ):
        self.mesh = mesh
        self.survey = survey
        if solver is None:
            solver = DefaultSolver
        self.solver = solver
        if solver_opts is None:
            solver_opts = {}
        self.solver_opts = solver_opts
        self.sensitivity_path = sensitivity_path
        self.counter = counter
        self.verbose = verbose

        super().__init__(**kwargs)

    ###########################################################################
    # Methods

    def fields(self, m=None):
        """
        u = fields(m)
        The field given the model.
        :param numpy.ndarray m: model
        :rtype: numpy.ndarray
        :return: u, the fields
        """
        raise NotImplementedError("fields has not been implemented for this ")

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
        if self.survey is None:
            raise AttributeError(
                "The survey has not yet been set and is required to compute "
                "data. Please set the survey for the simulation: "
                "simulation.survey = survey"
            )

        if f is None:
            if m is None:
                m = self.model

            f = self.fields(m)

        data = Data(self.survey)
        for src in self.survey.source_list:
            for rx in src.receiver_list:
                data[src, rx] = rx.eval(src, self.mesh, f)
        return mkvc(data)

    @timeIt
    def Jvec(self, m, v, f=None):
        """
        Jv = Jvec(m, v, f=None)
        Effect of J(m) on a vector v.
        :param numpy.ndarray m: model
        :param numpy.ndarray v: vector to multiply
        :param Fields f: fields
        :rtype: numpy.ndarray
        :return: Jv
        """
        raise NotImplementedError("Jvec is not yet implemented.")

    @timeIt
    def Jtvec(self, m, v, f=None):
        """
        Jtv = Jtvec(m, v, f=None)
        Effect of transpose of J(m) on a vector v.
        :param numpy.ndarray m: model
        :param numpy.ndarray v: vector to multiply
        :param Fields f: fields
        :rtype: numpy.ndarray
        :return: JTv
        """
        raise NotImplementedError("Jt is not yet implemented.")

    @timeIt
    def Jvec_approx(self, m, v, f=None):
        """Jvec_approx(m, v, f=None)
        Approximate effect of J(m) on a vector v
        :param numpy.ndarray m: model
        :param numpy.ndarray v: vector to multiply
        :param Fields f: fields
        :rtype: numpy.ndarray
        :return: approxJv
        """
        return self.Jvec(m, v, f)

    @timeIt
    def Jtvec_approx(self, m, v, f=None):
        """Jtvec_approx(m, v, f=None)
        Approximate effect of transpose of J(m) on a vector v.
        :param numpy.ndarray m: model
        :param numpy.ndarray v: vector to multiply
        :param Fields f: fields
        :rtype: numpy.ndarray
        :return: JTv
        """
        return self.Jtvec(m, v, f)

    @count
    def residual(self, m, dobs, f=None):
        """residual(m, dobs, f=None)
        The data residual:

        .. math::

            \mu_\\text{data} = \mathbf{d}_\\text{pred} - \mathbf{d}_\\text{obs}

        :param numpy.ndarray m: geophysical model
        :param numpy.ndarray f: fields
        :rtype: numpy.ndarray
        :return: data residual
        """
        return mkvc(self.dpred(m, f=f) - dobs)

    def make_synthetic_data(
        self, m, relative_error=0.05, noise_floor=0.0, f=None, add_noise=False, **kwargs
    ):
        """
        Make synthetic data given a model, and a standard deviation.
        :param numpy.ndarray m: geophysical model
        :param numpy.ndarray | float relative_error: standard deviation
        :param numpy.ndarray | float noise_floor: noise floor
        :param numpy.ndarray f: fields for the given model (if pre-calculated)
        """

        std = kwargs.pop("std", None)
        if std is not None:
            raise TypeError(
                "The std parameter has been removed. " "Please use relative_error."
            )

        if f is None:
            f = self.fields(m)

        dclean = self.dpred(m, f=f)

        if add_noise is True:
            std = np.sqrt((relative_error * np.abs(dclean)) ** 2 + noise_floor ** 2)
            noise = std * np.random.randn(*dclean.shape)
            dobs = dclean + noise
        else:
            dobs = dclean

        return SyntheticData(
            survey=self.survey,
            dobs=dobs,
            dclean=dclean,
            relative_error=relative_error,
            noise_floor=noise_floor,
        )


class BaseTimeSimulation(BaseSimulation):
    """
    Base class for a time domain simulation
    """

    @property
    def time_steps(self):
        """The time steps for the time domain simulation.

        You can set as an array of dt's or as a list of tuples/floats.
        If it is set as a list, tuples are unpacked with
        `discretize.utils.unpack_widths``.

        For example, the following setters are the same::

        >>> sim.time_steps = [(1e-6, 3), 1e-5, (1e-4, 2)]
        >>> sim.time_steps = np.r_[1e-6,1e-6,1e-6,1e-5,1e-4,1e-4]

        Returns
        -------
        numpy.ndarray

        See Also
        --------
        discretize.utils.unpack_widths
        """
        return self._time_steps

    @time_steps.setter
    def time_steps(self, value):
        if value is not None:
            if isinstance(value, list):
                value = unpack_widths(value)
            value = validate_ndarray_with_shape("time_steps", value, shape=("*",))
        self._time_steps = value
        del self.time_mesh

    @property
    def t0(self):
        """Start time for the discretization.

        Returns
        -------
        float
        """
        return self._t0

    @t0.setter
    def t0(self, value):
        self._t0 = validate_float("t0", value)
        del self.time_mesh

    def __init__(self, mesh=None, t0=0.0, time_steps=None, **kwargs):
        self.t0 = t0
        self.time_steps = time_steps
        super().__init__(mesh=mesh, **kwargs)

    @property
    def time_mesh(self):
        if getattr(self, "_time_mesh", None) is None:
            self._time_mesh = TensorMesh(
                [
                    self.time_steps,
                ],
                x0=[self.t0],
            )
        return self._time_mesh

    @time_mesh.deleter
    def time_mesh(self):
        if hasattr(self, "_time_mesh"):
            del self._time_mesh

    @property
    def nT(self):
        return self.time_mesh.n_cells

    @property
    def times(self):
        "Modeling times"
        return self.time_mesh.nodes_x

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
        if self.survey is None:
            raise AttributeError(
                "The survey has not yet been set and is required to compute "
                "data. Please set the survey for the simulation: "
                "simulation.survey = survey"
            )

        if f is None:
            f = self.fields(m)

        data = Data(self.survey)
        for src in self.survey.source_list:
            for rx in src.receiver_list:
                data[src, rx] = rx.eval(src, self.mesh, self.time_mesh, f)
        return data.dobs


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
    Inherit this class to build a linear simulation.
    """

    linear_model, model_map, model_deriv = props.Invertible(
        "The model for a linear problem"
    )

    def __init__(self, mesh=None, linear_model=None, model_map=None, G=None, **kwargs):
        super().__init__(mesh=mesh, **kwargs)
        self.linear_model = linear_model
        self.model_map = model_map
        self.solver = None
        if G is not None:
            self.G = G

        if self.survey is None:
            # Give it an empty survey
            self.survey = BaseSurvey([])
        if self.survey.nD == 0:
            # try seting the number of data to G
            if getattr(self, "G", None) is not None:
                self.survey._vnD = np.r_[self.G.shape[0]]

    @property
    def G(self):
        if getattr(self, "_G", None) is not None:
            return self._G
        else:
            warnings.warn("G has not been implemented for the simulation")
        return None

    @G.setter
    def G(self, G):
        # Allows setting G in a LinearSimulation
        # TODO should be validated
        self._G = G

    def fields(self, m):
        self.model = m
        return self.G.dot(self.linear_model)

    def dpred(self, m=None, f=None):
        if m is not None:
            self.model = m
        if f is not None:
            return f
        return self.fields(self.model)

    def getJ(self, m, f=None):
        self.model = m
        # self.model_deriv is likely a sparse matrix
        # and G is possibly dense, thus we need to do..
        return (self.model_deriv.T.dot(self.G.T)).T

    def Jvec(self, m, v, f=None):
        self.model = m
        return self.G.dot(self.model_deriv * v)

    def Jtvec(self, m, v, f=None):
        self.model = m
        return self.model_deriv.T * self.G.T.dot(v)


class ExponentialSinusoidSimulation(LinearSimulation):
    """
    This is the simulation class for the linear problem consisting of
    exponentially decaying sinusoids. The rows of the G matrix are

    .. math::

        \\int_x e^{p j_k x} \\cos(\\pi q j_k x) \\quad, j_k \\in [j_0, ..., j_n]
    """

    @property
    def n_kernels(self):
        """The number of kernels for the linear problem

        Returns
        -------
        int
        """
        return self._n_kernels

    @n_kernels.setter
    def n_kernels(self, value):
        self._n_kernels = validate_integer("n_kernels", value, min_val=1)

    @property
    def p(self):
        """Rate of exponential decay of the kernel.

        Returns
        -------
        float
        """
        return self._p

    @p.setter
    def p(self, value):
        self._p = validate_float("p", value)

    @property
    def q(self):
        """rate of oscillation of the kernel.

        Returns
        -------
        float
        """
        return self._q

    @q.setter
    def q(self, value):
        self._q = validate_float("q", value)

    @property
    def j0(self):
        """Maximum value for :math:`j_k = j_0`.

        Returns
        -------
        float
        """
        return self._j0

    @j0.setter
    def j0(self, value):
        self._j0 = validate_float("j0", value)

    @property
    def jn(self):
        """Maximum value for :math:`j_k = j_n`.

        Returns
        -------
        float
        """
        return self._jn

    @jn.setter
    def jn(self, value):
        self._jn = validate_float("jn", value)

    def __init__(self, n_kernels=20, p=-0.25, q=0.25, j0=0.0, jn=60.0, **kwargs):
        self.n_kernels = n_kernels
        self.p = p
        self.q = q
        self.j0 = j0
        self.jn = jn
        super(ExponentialSinusoidSimulation, self).__init__(**kwargs)

    @property
    def jk(self):
        """
        Parameters controlling the spread of kernel functions
        """
        if getattr(self, "_jk", None) is None:
            self._jk = np.linspace(self.j0, self.jn, self.n_kernels)
        return self._jk

    def g(self, k):
        """
        Kernel functions for the decaying oscillating exponential functions.
        """
        return np.exp(self.p * self.jk[k] * self.mesh.cell_centers_x) * np.cos(
            np.pi * self.q * self.jk[k] * self.mesh.cell_centers_x
        )

    @property
    def G(self):
        """
        Matrix whose rows are the kernel functions
        """
        if getattr(self, "_G", None) is None:
            G = np.empty((self.n_kernels, self.mesh.nC))

            for i in range(self.n_kernels):
                G[i, :] = self.g(i) * self.mesh.h[0]

            self._G = G
        return self._G
