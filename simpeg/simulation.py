"""
Define simulation classes.
"""

from __future__ import annotations  # needed to use type operands in Python 3.8
import os
import inspect
import numpy as np
import warnings

from discretize.base import BaseMesh
from discretize import TensorMesh
from discretize.utils import unpack_widths, sdiag

from . import props
from .typing import RandomSeed
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
    practical simulation classes in SimPEG.

    .. important::
        This class is not meant to be instantiated. You should inherit from it to
        create your own simulation class.

    Parameters
    ----------
    mesh : discretize.base.BaseMesh, optional
        Mesh on which the forward problem is discretized.
    survey : simpeg.survey.BaseSurvey, optional
        The survey for the simulation.
    solver : None or pymatsolver.base.Base, optional
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
    counter : None or simpeg.utils.Counter
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
            Mesh on which the forward problem is discretized.
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
        simpeg.survey.BaseSurvey
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
        None or simpeg.utils.Counter
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
            Verbose progress printout status.
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
        simpeg.fields.Fields
            Computed geophysical fields for the model provided.

        """
        raise NotImplementedError("fields has not been implemented for this ")

    def dpred(self, m=None, f=None):
        r"""Predicted data for the model provided.

        Parameters
        ----------
        m : (n_param,) numpy.ndarray
            The model parameters.
        f : simpeg.fields.Fields, optional
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
        f : simpeg.field.Fields, optional
            If provided, fields will not need to be recomputed for the
            current model to compute `Jvec`.

        Returns
        -------
        (n_data, ) numpy.ndarray
            The Jacobian times a vector for the model and vector provided.
        """
        raise NotImplementedError("Jvec is not yet implemented.")

    @timeIt
    def Jtvec(self, m, v, f=None):
        r"""Compute the Jacobian transpose times a vector for the model provided.

        The Jacobian defines the derivative of the predicted data vector with respect to the
        model parameters. For a data vector :math:`\mathbf{d}` predicted for a set of model parameters
        :math:`\mathbf{m}`, the Jacobian is an ``(n_data, n_param)`` matrix whose elements
        are given by:

        .. math::
            J_{ij} = \frac{\partial d_i}{\partial m_j}

        For a model `m` and vector `v`, the ``Jtvec`` method computes the matrix-vector product with the adjoint-sensitivity

        .. math::
            \mathbf{u} = \mathbf{J^T \, v}

        Parameters
        ----------
        m : (n_param, ) numpy.ndarray
            The model parameters.
        v : (n_data, ) numpy.ndarray
            Vector we are multiplying.
        f : simpeg.field.Fields, optional
            If provided, fields will not need to be recomputed for the
            current model to compute `Jtvec`.

        Returns
        -------
        (n_param, ) numpy.ndarray
            The Jacobian transpose times a vector for the model and vector provided.
        """
        raise NotImplementedError("Jtvec is not yet implemented.")

    @timeIt
    def Jvec_approx(self, m, v, f=None):
        r"""Approximation of the Jacobian times a vector for the model provided.

        The Jacobian defines the derivative of the predicted data vector with respect to the
        model parameters. For a data vector :math:`\mathbf{d}` predicted for a set of model parameters
        :math:`\mathbf{m}`, the Jacobian is an ``(n_data, n_param)`` matrix whose elements
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
        f : simpeg.field.Fields, optional
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
        :math:`\mathbf{m}`, the Jacobian is an ``(n_data, n_param)`` matrix whose elements
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
        f : simpeg.field.Fields, optional
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
        Where :math:`\mathbf{d}_\text{obs}` are the observed data values, and :math:`\mathbf{d}_\text{pred}`
        are the predicted data values for model parameters :math:`\mathbf{m}`, the data
        residual is given by:

        .. math::
            \mathbf{r}(\mathbf{m}) = \mathbf{d}_\text{pred} - \mathbf{d}_\text{obs}

        Parameters
        ----------
        m : (n_param, ) numpy.ndarray
            The model parameters.
        dobs : (n_data, ) numpy.ndarray
            The observed data values.
        f : simpeg.fields.Fields, optional
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
        random_seed: RandomSeed | None = None,
        **kwargs,
    ):
        r"""Make synthetic data for the model and Gaussian noise provided.

        This method generates and returns a :py:class:`simpeg.data.SyntheticData` object
        for the model and standard deviation of Gaussian noise provided.

        Parameters
        ----------
        m : (n_param, ) numpy.ndarray
            The model parameters.
        relative_error : float, numpy.ndarray
            Assign relative uncertainties to the data using relative error; sometimes
            referred to as percent uncertainties. For each datum, we assume the
            standard deviation of Gaussian noise is the relative error times the
            absolute value of the datum; i.e. :math:`C_\text{err} \times |d|`.
        noise_floor : float, numpy.ndarray
            Assign floor/absolute uncertainties to the data. For each datum, we assume
            standard deviation of Gaussian noise is equal to `noise_floor`.
        f : simpeg.fields.Fields, optional
            If provided, fields will not need to be recomputed when solving the
            forward problem to obtain noiseless data.
        add_noise : bool
            Whether to add gaussian noise to the synthetic data or not.
        random_seed : None or :class:`~simpeg.typing.RandomSeed`, optional
            Random seed used for random sampling. It can either be an int or
            a predefined Numpy random number generator (see
            ``numpy.random.default_rng``).

        Returns
        -------
        simpeg.data.SyntheticData
            A SimPEG synthetic data object, which organizes both clean and noisy data.
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
            random_num_generator = np.random.default_rng(seed=random_seed)
            std = np.sqrt((relative_error * np.abs(dclean)) ** 2 + noise_floor**2)
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
    r"""Base class for time domain simulations.

    The ``BaseTimeSimulation`` defines properties and methods that are required
    when the finite volume approach is used to solve time-dependent forward simulations.
    Presently, SimPEG discretizes in time using the backward Euler approach.
    And as such, the user must now define the step lengths for the forward simulation.

    Parameters
    ----------
    mesh : discretize.base.BaseMesh, optional
        Mesh on which the forward problem is discretized. This is not necessarily
        the same as the mesh on which the simulation is defined.
    t0 : float, optional
        Initial time, in seconds, for the time-dependent forward simulation.
    time_steps : (n_steps, ) numpy.ndarray, optional
        The time step lengths, in seconds, for the time domain simulation.
        This property can be also be set using a compact form; see *Notes*.

    Notes
    -----
    There are two ways in which the user can set the ``time_steps`` property
    for the forward simulation. The most basic approach is to use a ``(n_steps, )``
    :py:class:`numpy.ndarray` that explicitly defines the step lengths in order.
    I.e.:

    >>> sim.time_steps = np.r_[1e-6, 1e-6, 1e-6, 1e-5, 1e-5, 1e-4, 1e-4]

    We can define also define the step lengths in compact for when the same
    step length is reused multiple times in succession. In this case, the
    ``time_steps`` property is set using a ``list`` of ``tuple``. Each
    ``tuple`` contains the step length and number of times that step is repeated.
    The time stepping defined above can be set equivalently with:

    >>> sim.time_steps = [(1e-6, 3), (1e-5, 2), (1e-4, 2)]

    When set, the :py:func:`discretize.utils.unpack_widths` utility is
    used to convert the ``list`` of ``tuple`` to its (n_steps, ) :py:class:`numpy.ndarray`
    representation.
    """

    def __init__(self, mesh=None, t0=0.0, time_steps=None, **kwargs):
        self.t0 = t0
        self.time_steps = time_steps
        super().__init__(mesh=mesh, **kwargs)

    @property
    def time_steps(self):
        """Time step lengths, in seconds, for the time domain simulation.

        There are two ways in which the user can set the ``time_steps`` property
        for the forward simulation. The most basic approach is to use a ``(n_steps, )``
        :py:class:`numpy.ndarray` that explicitly defines the step lengths in order.
        I.e.:

        >>> sim.time_steps = np.r_[1e-6, 1e-6, 1e-6, 1e-5, 1e-5, 1e-4, 1e-4]

        We can define also define the step lengths in compact for when the same
        step length is reused multiple times in succession. In this case, the
        ``time_steps`` property is set using a ``list`` of ``tuple``. Each
        ``tuple`` contains the step length and number of times that step is repeated.
        The time stepping defined above can be set equivalently with:

        >>> sim.time_steps = [(1e-6, 3), (1e-5, 2), (1e-4, 2)]

        When set, the :py:func:`discretize.utils.unpack_widths` utility is
        used to convert the ``list`` of ``tuple`` to its ``(n_steps, )`` :py:class:`numpy.ndarray`
        representation.

        Returns
        -------
        (n_steps, ) numpy.ndarray
            The time step lengths for the time domain simulation.
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
        """Initial time, in seconds, for the time-dependent forward simulation.

        Returns
        -------
        float
            Initial time, in seconds, for the time-dependent forward simulation.
        """
        return self._t0

    @t0.setter
    def t0(self, value):
        self._t0 = validate_float("t0", value)
        del self.time_mesh

    @property
    def time_mesh(self):
        r"""Time mesh for easy interpolation to observation times.

        The time mesh is constructed internally from the :py:attr:`t0` and
        :py:attr:`time_steps` properties using the :py:class:`discretize.TensorMesh` class.
        The ``time_mesh`` property allows for easy interpolation from fields computed at
        discrete time-steps, to an arbitrary set of observation
        times within the continuous interval (:math:`t_0 , t_\text{end}`).

        Returns
        -------
        discretize.TensorMesh
            The time mesh.
        """
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
        """Total number of time steps.

        Returns
        -------
        int
            Total number of time steps.
        """
        return self.time_mesh.n_cells

    @property
    def times(self):
        """Evaluation times.

        Returns the discrete set of times at which the fields are computed for
        the forward simulation.

        Returns
        -------
        (nT, ) numpy.ndarray
            The discrete set of times at which the fields are computed for
            the forward simulation.
        """
        return self.time_mesh.nodes_x

    def dpred(self, m=None, f=None):
        # Docstring inherited from BaseSimulation.
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
    r"""Linear forward simulation class.

    The ``LinearSimulation`` class is used to define forward simulations of the form:

    .. math::
        \mathbf{d} = \mathbf{G \, f}(\mathbf{m})

    where :math:`\mathbf{m}` are the model parameters, :math:`\mathbf{f}` is a
    mapping operator (optional) from the model space to a user-defined parameter space,
    :math:`\mathbf{d}` is the predicted data vector, and :math:`\mathbf{G}` is an
    ``(n_data, n_param)`` linear operator.

    The ``LinearSimulation`` class is generally used as a base class that is inherited by
    other simulation classes within SimPEG. However, it can be used directly as a
    simulation class if the :py:attr:`G` property is used to set the linear forward
    operator directly.

    By default, we assume the mapping operator :math:`\mathbf{f}` is the identity map,
    and that the forward simulation reduces to:

    .. math::
        \mathbf{d} = \mathbf{G \, m}

    Parameters
    ----------
    mesh : discretize.BaseMesh, optional
        Mesh on which the forward problem is discretized. This is not necessarily
        the same as the mesh on which the simulation is defined.
    model_map : simpeg.maps.BaseMap
        Mapping from the model parameters to vector that the linear operator acts on.
    G : (n_data, n_param) numpy.ndarray or scipy.sparse.csr_matrx
        The linear operator. For a ``model_map`` that maps within the same vector space
        (e.g. the identity map), the dimension ``n_param`` equals the number of model parameters.
        If not, the dimension ``n_param`` of the linear operator will depend on the mapping.
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
        """The linear operator.

        Returns
        -------
        (n_data, n_param) numpy.ndarray or scipy.sparse.csr_matrix
            The linear operator. For a :py:attr:`model_map` that maps within the same vector space
            (e.g. the identity map), the dimension ``n_param`` equals the number of model parameters.
            If not, the dimension ``n_param`` of the linear operator will depend on the mapping.
        """
        if getattr(self, "_G", None) is not None:
            return self._G
        else:
            warnings.warn("G has not been implemented for the simulation", stacklevel=2)
        return None

    @G.setter
    def G(self, G):
        # Allows setting G in a LinearSimulation.
        # TODO should be validated
        self._G = G

    def fields(self, m):
        # Docstring inherited from BaseSimulation.
        self.model = m
        return self.G.dot(self.linear_model)

    def dpred(self, m=None, f=None):
        # Docstring inherited from BaseSimulation
        if m is not None:
            self.model = m
        if f is not None:
            return f
        return self.fields(self.model)

    def getJ(self, m, f=None):
        r"""Returns the full Jacobian.

        The general definition of the linear forward simulation is:

        .. math::
            \mathbf{d} = \mathbf{G \, f}(\mathbf{m})

        where :math:`\mathbf{f}` is a mapping operator (optional) from the model space
        to a user-defined parameter space, and :math:`\mathbf{G}` is an (n_data, n_param)
        linear operator. The ``getJ`` method forms and returns the full Jacobian:

        .. math::
            \mathbf{J}(\mathbf{m}) = \mathbf{G} \frac{\partial \mathbf{f}}{\partial \mathbf{m}}

        for the model :math:`\mathbf{m}` provided. When :math:`\mathbf{f}` is the identity map
        (default), the Jacobian is no longer model-dependent and reduces to:

        .. math::
            \mathbf{J} = \mathbf{G}

        Parameters
        ----------
        m : numpy.ndarray
            The model vector.
        f : None
            Precomputed fields are not used to speed up the computation of the
            Jacobian for linear problems.

        Returns
        -------
        J : (n_data, n_param) numpy.ndarray
            :math:`J = G\frac{\partial f}{\partial\mathbf{m}}`.
            Where :math:`f` is :attr:`model_map`.
        """
        self.model = m
        # self.model_deriv is likely a sparse matrix
        # and G is possibly dense, thus we need to do..
        return (self.model_deriv.T.dot(self.G.T)).T

    def Jvec(self, m, v, f=None):
        # Docstring inherited from BaseSimulation
        self.model = m
        return self.G.dot(self.model_deriv * v)

    def Jtvec(self, m, v, f=None):
        # Docstring inherited from BaseSimulation
        self.model = m
        return self.model_deriv.T * self.G.T.dot(v)


class ExponentialSinusoidSimulation(LinearSimulation):
    r"""Simulation class for exponentially decaying sinusoidal kernel functions.

    This is the simulation class for the linear problem consisting of
    exponentially decaying sinusoids. The entries of the linear operator
    :math:`\mathbf{G}` are:

    .. math::

        G_{ik} = \int_\Omega e^{p \, j_i \, x_k} \cos(\pi \, q \, j_i \, x_k) \, dx

    The model is defined on a 1D :py:class:`discretize.TensorMesh`, and :math:`x_k`
    are the cell center locations. :math:`p \leq 0` defines the rate of exponential
    decay of the kernel functions. :math:`q` defines the rate of oscillation of
    the kernel functions. And :math:`j_i \in [j_0, ... , j_n]` controls the spread
    of the kernel functions; the number of which is set using the ``n_kernels``
    property.

    .. tip::

        For proper scaling, we advise defining the 1D tensor mesh to
        discretize the interval [0, 1].

    The kernel functions take the form:

    .. math::

        \int_x e^{p j_k x} \cos(\pi q j_k x) \quad, j_k \in [j_0, ..., j_n]

    The model is defined at cell centers while the kernel functions are defined on nodes.
    The trapezoid rule is used to evaluate the integral

    .. math::

        d_j = \int g_j(x) m(x) dx

    to define our data.

    Parameters
    ----------
    n_kernels : int
        The number of kernel factors for the linear problem; i.e. the number of
        :math:`j_i \in [j_0, ... , j_n]`. This sets the number of rows
        in the linear forward operator.
    p : float
        Exponent specifying the decay (`p \leq 0`) or growth (`p \geq 0`) of the kernel. For decay, set :math:`p \leq 0`.
    q : float
        Rate of oscillation of the kernel.
    j0 : float
        Minimum value for the spread of the kernel factors.
    jn : float
        Maximum value for the spread of the kernel factors.
    """

    def __init__(self, n_kernels=20, p=-0.25, q=0.25, j0=0.0, jn=60.0, **kwargs):
        self.n_kernels = n_kernels
        self.p = p
        self.q = q
        self.j0 = j0
        self.jn = jn
        super(ExponentialSinusoidSimulation, self).__init__(**kwargs)

    @property
    def n_kernels(self):
        r"""The number of kernel factors for the linear problem.

        Where :math:`j_0` represents the minimum value for the spread of
        kernel factors and :math:`j_n` represents the maximum, ``n_kernels``
        defines the number of kernel factors :math:`j_i \in [j_0, ... , j_n]`.
        This ultimately sets the number of rows in the linear forward operator.

        Returns
        -------
        int
            The number of kernel factors for the linear problem.
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
            Rate of exponential decay of the kernel.
        """
        return self._p

    @p.setter
    def p(self, value):
        self._p = validate_float("p", value)

    @property
    def q(self):
        """Rate of oscillation of the kernel.

        Returns
        -------
        float
            Rate of oscillation of the kernel.
        """
        return self._q

    @q.setter
    def q(self, value):
        self._q = validate_float("q", value)

    @property
    def j0(self):
        """Minimum value for the spread of the kernel factors.

        Returns
        -------
        float
            Minimum value for the spread of the kernel factors.
        """
        return self._j0

    @j0.setter
    def j0(self, value):
        self._j0 = validate_float("j0", value)

    @property
    def jn(self):
        """Maximum value for the spread of the kernel factors.

        Returns
        -------
        float
            Maximum value for the spread of the kernel factors.
        """
        return self._jn

    @jn.setter
    def jn(self, value):
        self._jn = validate_float("jn", value)

    @property
    def jk(self):
        """The set of kernel factors controlling the spread of the kernel functions.

        Returns
        -------
        (n_kernels, ) numpy.ndarray
            The set of kernel factors controlling the spread of the kernel functions.
        """
        if getattr(self, "_jk", None) is None:
            self._jk = np.linspace(self.j0, self.jn, self.n_kernels)
        return self._jk

    def g(self, k):
        """Kernel functions evaluated for kernel factor :math:`j_k`.

        This method computes the row of the linear forward operator for
        the kernel functions for kernel factor :math:`j_k`, given :math:`k`

        Parameters
        ----------
        k : int
            Kernel functions for kernel factor *k*

        Returns
        -------
        (n_param, ) numpy.ndarray
            Kernel functions evaluated for kernel factor *k*.
        """
        return np.exp(self.p * self.jk[k] * self.mesh.nodes_x) * np.cos(
            np.pi * self.q * self.jk[k] * self.mesh.nodes_x
        )

    @property
    def G(self):
        """The linear forward operator.

        Returns
        -------
        (n_kernels, n_param) numpy.ndarray
            The linear forward operator.
        """
        if getattr(self, "_G", None) is None:
            G_nodes = np.empty((self.mesh.n_nodes, self.n_kernels))

            for i in range(self.n_kernels):
                G_nodes[:, i] = self.g(i)

            self._G = (self.mesh.average_node_to_cell @ G_nodes).T @ sdiag(
                self.mesh.cell_volumes
            )
        return self._G
