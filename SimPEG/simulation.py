"""
Define simulation classes.
"""
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
    r"""Base class for all geophysical forward simulations in SimPEG.

    The ``BaseSimulation`` class defines properties and methods inherited by
    practical simulation classes in SimPEG. Instances of ``BaseSimulation``
    are not used direction to perform forward simulations in SimPEG.

    Parameters
    ----------
    mesh : discretize.base.BaseMesh
        Mesh on which the forward problem is discretized. This is not necessarily
        the same as the mesh on which the simulation is defined.
    survey : SimPEG.survey.BaseSurvey
        The survey for the simulation.
    solver : None, pymatsolver.base.Base
        Numerical solver used to solve the forward problem. If ``None``,
        an appropriate solver specific to the simulation class is set by default.
    solver_opts : dict, optional
        Solver-specific parameters. If ``None``, default parameters are used for
        the solver set by ``solver``. Otherwise, the ``dict`` must contain appropriate
        pairs of keyword arguments and parameter values for the solver. Please visit
        `pymatsolver <https://pymatsolver.readthedocs.io/en/latest/>`__ to learn more
        about solvers and their parameters.
    sensitivity_path : str, optional
        Path to directory where sensitivity file is stored.
    counter : None, SimPEG.utils.Counter
        SimPEG ``Counter`` object to store iterations and run-times.
    verbose : bool, optional
        Verbose progress printout.
    """

    _REGISTRY = {}

    def __init__(
        self,
        mesh=None,
        survey=None,
        solver=None,
        solver_opts=None,
        sensitivity_path=None,
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
        if sensitivity_path is None:
            sensitivity_path = os.path.join(".", "sensitivity")
        self.sensitivity_path = sensitivity_path
        self.counter = counter
        self.verbose = verbose

        super().__init__(**kwargs)

    @property
    def mesh(self):
        """Mesh for the simulation.

        For more on meshes, visit :py:class:`discretize.base.BaseMesh`.

        Returns
        -------
        discretize.base.BaseMesh
            Mesh on which the forward problem is discretized. This is not necessarily
            the same as the mesh on which the simulation is defined.
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
            The survey for the simulation.
        """
        return self._survey

    @survey.setter
    def survey(self, value):
        if value is not None:
            value = validate_type("survey", value, BaseSurvey, cast=False)
        self._survey = value

    @property
    def counter(self):
        """SimPEG ``Counter`` object to store iterations and run-times.

        Returns
        -------
        None, SimPEG.utils.Counter
            SimPEG ``Counter`` object to store iterations and run-times.
        """
        return self._counter

    @counter.setter
    def counter(self, value):
        if value is not None:
            value = validate_type("counter", value, Counter, cast=False)
        self._counter = value

    @property
    def sensitivity_path(self):
        """Path to directory where sensitivity file is stored.

        Returns
        -------
        str
            Path to directory where sensitivity file is stored.
        """
        return self._sensitivity_path

    @sensitivity_path.setter
    def sensitivity_path(self, value):
        self._sensitivity_path = validate_string("sensitivity_path", value)

    @property
    def solver(self):
        r"""Numerical solver used in the forward simulation.

        Many forward simulations in SimPEG require solutions to discrete linear
        systems of the form:

        .. math::
            \mathbf{A}(\mathbf{m}) \, \mathbf{u} = \mathbf{q}

        where :math:`\mathbf{A}` is an invertible matrix that depends on the
        model :math:`\mathbf{m}`. The numerical solver can be set using the
        ``solver`` property. In SimPEG, the
        `pymatsolver <https://pymatsolver.readthedocs.io/en/latest/>`__ package
        is used to create solver objects. Parameters specific to each solver
        can be set manually using the ``solver_opts`` property.

        Returns
        -------
        pymatsolver.base.Base
            Numerical solver used to solve the forward problem.
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
        """Solver-specific parameters.

        The parameters specific to the solver set with the ``solver`` property are set
        upon instantiation. The ``solver_opts`` property is used to set solver-specific properties.
        This is done by providing a ``dict`` that contains appropriate pairs of keyword arguments
        and parameter values. Please visit `pymatsolver <https://pymatsolver.readthedocs.io/en/latest/>`__
        to learn more about solvers and their parameters.

        Returns
        -------
        dict
            keyword arguments and parameters passed to the solver.
        """
        return self._solver_opts

    @solver_opts.setter
    def solver_opts(self, value):
        self._solver_opts = validate_type("solver_opts", value, dict, cast=False)

    @property
    def verbose(self):
        """Verbose progress printout.

        Returns
        -------
        bool
            Verbose progress printout.
        """
        return self._verbose

    @verbose.setter
    def verbose(self, value):
        self._verbose = validate_type("verbose", value, bool)

    def fields(self, m=None):
        r"""Return the computed geophysical fields for the model provided.

        Parameters
        ----------
        m : (n_param,) numpy.ndarray
            The model parameters.

        Returns
        -------
        SimPEG.fields.Fields
            Computed geophysical fields for the model provided

        """
        raise NotImplementedError("fields has not been implemented for this ")

    def dpred(self, m=None, f=None):
        r"""Predicted data for the model provided.

        Parameters
        ----------
        m : (n_param,) numpy.ndarray
            The model parameters.
        f : SimPEG.fields.Fields, optional
            If provided, will be used to compute the predicted data
            without recalculating the fields.

        Returns
        -------
        (n_data, ) numpy.ndarray
            The predicted data vector.
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
        r"""Compute the Jacobian times a vector for the model provided.

        The Jacobian defines the derivative of the predicted data vector with respect to the
        model parameters. For a data vector :math:`\mathbf{d}` predicted for a set of model parameters
        :math:`\mathbf{m}`, the Jacobian is an (n_data, n_param) matrix whose elements
        are given by:

        .. math::
            J_{ij} = \frac{\partial d_i}{\partial m_j}

        For a model `m` and vector `v`, the ``Jvec`` method computes the matrix-vector product

        .. math::
            \mathbf{u} = \mathbf{J \, v}

        Parameters
        ----------
        m : (n_param, ) numpy.ndarray
            The model parameters.
        v : (n_param, ) numpy.ndarray
            Vector we are multiplying.
        f : SimPEG.field.Fields, optional
            If provided, fields will not need to be recomputed for the
            current model to compute `Jvec`.

        Returns
        -------
        (n_data, ) numpy.ndarray
            The Jacobian times a vector for the model provided.
        """
        raise NotImplementedError("Jvec is not yet implemented.")

    @timeIt
    def Jtvec(self, m, v, f=None):
        r"""Compute the Jacobian transpose times a vector for the model provided.

        The Jacobian defines the derivative of the predicted data vector with respect to the
        model parameters. For a data vector :math:`\mathbf{d}` predicted for a set of model parameters
        :math:`\mathbf{m}`, the Jacobian is an (n_data, n_param) matrix whose elements
        are given by:

        .. math::
            J_{ij} = \frac{\partial d_i}{\partial m_j}

        For a model `m` and vector `v`, the ``Jtvec`` method computes the matrix-vector product

        .. math::
            \mathbf{u} = \mathbf{J^T \, v}

        Parameters
        ----------
        m : (n_param, ) numpy.ndarray
            The model parameters.
        v : (n_data, ) numpy.ndarray
            Vector we are multiplying.
        f : SimPEG.field.Fields, optional
            If provided, fields will not need to be recomputed for the
            current model to compute `Jtvec`.

        Returns
        -------
        (n_param, ) numpy.ndarray
            The Jacobian transpose times a vector for the model provided.
        """
        raise NotImplementedError("Jtvec is not yet implemented.")

    @timeIt
    def Jvec_approx(self, m, v, f=None):
        r"""Approximation of the Jacobian times a vector for the model provided.

        The Jacobian defines the derivative of the predicted data vector with respect to the
        model parameters. For a data vector :math:`\mathbf{d}` predicted for a set of model parameters
        :math:`\mathbf{m}`, the Jacobian is an (n_data, n_param) matrix whose elements
        are given by:

        .. math::
            J_{ij} = \frac{\partial d_i}{\partial m_j}

        For a model `m` and vector `v`, the ``Jvec_approx`` method **approximates**
        the matrix-vector product:

        .. math::
            \mathbf{u} = \mathbf{J \, v}

        Parameters
        ----------
        m : (n_param, ) numpy.ndarray
            The model parameters.
        v : (n_data, ) numpy.ndarray
            Vector we are multiplying.
        f : SimPEG.field.Fields, optional
            If provided, fields will not need to be recomputed for the
            current model to compute `Jtvec`.

        Returns
        -------
        (n_param, ) numpy.ndarray
            Approximation of the Jacobian times a vector for the model provided.
        """
        return self.Jvec(m, v, f)

    @timeIt
    def Jtvec_approx(self, m, v, f=None):
        r"""Approximation of the Jacobian transpose times a vector for the model provided.

        The Jacobian defines the derivative of the predicted data vector with respect to the
        model parameters. For a data vector :math:`\mathbf{d}` predicted for a set of model parameters
        :math:`\mathbf{m}`, the Jacobian is an (n_data, n_param) matrix whose elements
        are given by:

        .. math::
            J_{ij} = \frac{\partial d_i}{\partial m_j}

        For a model `m` and vector `v`, the ``Jtvec_approx`` method **approximates**
        the matrix-vector product:

        .. math::
            \mathbf{u} = \mathbf{J^T \, v}

        Parameters
        ----------
        m : (n_param, ) numpy.ndarray
            The model parameters.
        v : (n_data, ) numpy.ndarray
            Vector we are multiplying.
        f : SimPEG.field.Fields, optional
            If provided, fields will not need to be recomputed for the
            current model to compute `Jtvec`.

        Returns
        -------
        (n_param, ) numpy.ndarray
            Approximation of the Jacobian transpose times a vector for the model provided.
        """
        return self.Jtvec(m, v, f)

    @count
    def residual(self, m, dobs, f=None):
        r"""The data residual.

        This method computes and returns the data residual for the model provided.
        Where :math:`\mathbf{d}_{obs}` are the observed data values, and :math:`\mathbf{d}_pred`
        are the predicted data values for model parameters :math:`\mathbf{m}`, the data
        residual is given by:

        .. math::
            \mathbf{r}(\mathbf{m}) = \mathbf{d}_{pred} - \mathbf{d}_{obs}

        Parameters
        ----------
        m : (n_param, ) numpy.ndarray
            The model parameters.
        dobs : (n_data, ) numpy.ndarray
            The observed data values.
        f : SimPEG.fields.Fields, optional
            If provided, fields will not need to be recomputed when solving the forward problem.

        Returns
        -------
        (n_data, ) numpy.ndarray
            The data residual.

        """
        return mkvc(self.dpred(m, f=f) - dobs)

    def make_synthetic_data(
        self,
        m,
        relative_error=0.05,
        noise_floor=0.0,
        f=None,
        add_noise=False,
        random_seed=None,
        **kwargs,
    ):
        """
        Make synthetic data for the model and noise level provided.

        Parameters
        ----------
        m : (n_param, ) numpy.ndarray
            The model parameters.
        relative_error : float
            Standard deviation.
        noise_floor : float
            Noise floor.
        f : array or None
            Fields for the given model (if pre-calculated).
        add_noise : bool
            Whether to add gaussian noise to the synthetic data or not.
        random_seed : int or None
            Random seed to pass to `numpy.random.default_rng`.

        Returns
        -------
        SimPEG.data.Data
            A SimPEG data object.
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
            std = np.sqrt((relative_error * np.abs(dclean)) ** 2 + noise_floor**2)
            random_num_generator = np.random.default_rng(seed=random_seed)
            noise = random_num_generator.normal(loc=0, scale=std, size=dclean.shape)
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
    """Base class for a time domain simulation."""

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
        discretize.utils.unpack_widths.
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
        """Time steps."""
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
        """Total number of time steps."""
        return self.time_mesh.n_cells

    @property
    def times(self):
        """Modeling times."""
        return self.time_mesh.nodes_x

    def dpred(self, m=None, f=None):
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
    r"""Base class for the definition of a linear forward problem.

    Simulation is of the form

    .. math::

        \mathbf{d} = G\mathbf{m},

    where :math:`\mathbf{d}` is the data vector, `G` is the simulation matrix and
    :math:`\mathbf{m}` is the model parameter vector.

    Inherit this class to build a linear simulation.

    """

    linear_model, model_map, model_deriv = props.Invertible(
        "the model for a linear problem"
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
            warnings.warn("G has not been implemented for the simulation", stacklevel=2)
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
        r"""Returns the derivative of the forward problem w.r.t **m**.

        Parameters
        ----------
        m : numpy.ndarray
            Current model parameter.
        f : optional
            Precomputed fields.

        Returns
        -------
        J : (nD, nP) numpy.ndarray
            :math:`J = G\frac{\partial f}{\partial\mathbf{m}}`.
            Where :math:`f` is :attr:`model_map`.
        """
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
    r"""Exponentially decaying sinusoids.

    This is the simulation class for the linear problem consisting of
    exponentially decaying sinusoids. The rows of the G matrix are

    .. math::

        \int_x e^{p j_k x} \cos(\pi q j_k x) \quad, j_k \in [j_0, ..., j_n]
    """

    @property
    def n_kernels(self):
        """The number of kernels for the linear problem.

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
        Parameters controlling the spread of kernel functions.
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
        Matrix whose rows are the kernel functions.
        """
        if getattr(self, "_G", None) is None:
            G = np.empty((self.n_kernels, self.mesh.nC))

            for i in range(self.n_kernels):
                G[i, :] = self.g(i) * self.mesh.h[0]

            self._G = G
        return self._G
