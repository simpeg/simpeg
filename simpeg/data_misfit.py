import numpy as np
from .utils import Counter, sdiag, timeIt, Identity, validate_type
from .data import Data
from .simulation import BaseSimulation
from .objective_function import L2ObjectiveFunction

__all__ = ["L2DataMisfit"]


class BaseDataMisfit(L2ObjectiveFunction):
    r"""Base data misfit class.

    Inherit this class to build your own data misfit function. The ``BaseDataMisfit``
    class inherits the :py:class:`simpeg.objective_function.L2ObjectiveFunction`.
    And as a result, it is limited to building data misfit functions of the form:

    .. important::
        This class is not meant to be instantiated. You should inherit from it to
        create your own data misfit class.

    .. math::
        \phi_d (\mathbf{m}) = \| \mathbf{W} f(\mathbf{m}) \|_2^2

    where :math:`\mathbf{m}` is the model vector, :math:`\mathbf{W}` is a linear weighting
    matrix, and :math:`f` is a mapping function that acts on the model.

    Parameters
    ----------
    data : simpeg.data.Data
        A SimPEG data object.
    simulation : simpeg.simulation.BaseSimulation
        A SimPEG simulation object.
    debug : bool
        Print debugging information.
    counter : None or simpeg.utils.Counter
        Assign a SimPEG ``Counter`` object to store iterations and run-times.
    """

    def __init__(self, data, simulation, debug=False, counter=None, **kwargs):
        super().__init__(has_fields=True, debug=debug, counter=counter, **kwargs)

        self.data = data
        self.simulation = simulation

    @property
    def data(self):
        """A SimPEG data object.

        Returns
        -------
        simpeg.data.Data
            A SimPEG data object.
        """
        return self._data

    @data.setter
    def data(self, value):
        self._data = validate_type("data", value, Data, cast=False)

    @property
    def simulation(self):
        """A SimPEG simulation object.

        Returns
        -------
        simpeg.simulation.BaseSimulation
            A SimPEG simulation object.
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
            Print debugging information.
        """
        return self._debug

    @debug.setter
    def debug(self, value):
        self._debug = validate_type("debug", value, bool)

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

    @property
    def nP(self):
        """Number of model parameters.

        Returns
        -------
        int
            Number of model parameters.
        """
        if self._mapping is not None:
            return self.mapping.nP
        elif self.simulation.model is not None:
            return len(self.simulation.model)
        else:
            return "*"

    @property
    def nD(self):
        """Number of data.

        Returns
        -------
        int
            Number of data.
        """
        return self.data.nD

    @property
    def shape(self):
        """Shape of the Jacobian.

        The number of data by the number of model parameters.

        Returns
        -------
        tuple of int (n_data, n_param)
            Shape of the Jacobian; i.e. number of data by the number of model parameters.
        """
        return (self.nD, self.nP)

    @property
    def W(self):
        r"""The data weighting matrix.

        For a discrete least-squares data misfit function of the form:

        .. math::
            \phi_d (\mathbf{m}) = \| \mathbf{W} \mathbf{f}(\mathbf{m}) \|_2^2

        :math:`\mathbf{W}` is a linear weighting matrix, :math:`\mathbf{m}` is the model vector,
        and :math:`\mathbf{f}` is a discrete mapping function that acts on the model vector.

        Returns
        -------
        scipy.sparse.csr_matrix
            The data weighting matrix.
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
        ), "W must have shape ({nD},{nD}), not ({val0}, {val1})".format(
            nD=self.data.nD, val0=value.shape[0], val1=value.shape[1]
        )
        self._W = value

    def residual(self, m, f=None):
        r"""Computes the data residual vector for a given model.

        Where :math:`\mathbf{d}_\text{obs}` is the observed data vector and :math:`\mathbf{d}_\text{pred}`
        is the predicted data vector for a model vector :math:`\mathbf{m}`, this function
        computes the data residual:

        .. math::
            \mathbf{r} = \mathbf{d}_\text{pred} - \mathbf{d}_\text{obs}

        Parameters
        ----------
        m : (n_param, ) numpy.ndarray
            The model for which the function is evaluated.
        f : None or simpeg.fields.Fields, optional
            A SimPEG fields object. Used when the fields for the model *m* have
            already been computed.

        Returns
        -------
        (n_data, ) numpy.ndarray
            The data residual vector.
        """
        if self.data is None:
            raise Exception("data must be set before a residual can be calculated.")
        return self.simulation.residual(m, self.data.dobs, f=f)


class L2DataMisfit(BaseDataMisfit):
    r"""Least-squares data misfit.

    Define the data misfit as the L2-norm of the weighted residual between observed
    data and predicted data for a given model. I.e.:

    .. math::
        \phi_d (\mathbf{m}) = \big \| \mathbf{W_d}
        \big ( \mathbf{d}_\text{pred} - \mathbf{d}_\text{obs} \big ) \big \|_2^2

    where :math:`\mathbf{d}_\text{obs}` is the observed data vector, :math:`\mathbf{d}_\text{pred}`
    is the predicted data vector for a model vector :math:`\mathbf{m}`, and
    :math:`\mathbf{W_d}` is the data weighting matrix. The diagonal elements of
    :math:`\mathbf{W_d}` are the reciprocals of the data uncertainties
    :math:`\boldsymbol{\varepsilon}`. Thus:

    .. math::
        \mathbf{W_d} = \text{diag} \left ( \boldsymbol{\varepsilon}^{-1} \right )

    Parameters
    ----------
    data : simpeg.data.Data
        A SimPEG data object that has observed data and uncertainties.
    simulation : simpeg.simulation.BaseSimulation
        A SimPEG simulation object.
    debug : bool
        Print debugging information.
    counter : None or simpeg.utils.Counter
        Assign a SimPEG ``Counter`` object to store iterations and run-times.
    """

    @timeIt
    def __call__(self, m, f=None):
        """Evaluate the residual for a given model."""

        R = self.W * self.residual(m, f=f)
        return np.vdot(R, R)

    @timeIt
    def deriv(self, m, f=None):
        r"""Gradient of the data misfit function evaluated for the model provided.

        Where :math:`\phi_d (\mathbf{m})` is the data misfit function,
        this method evaluates and returns the derivative with respect to the model parameters; i.e.
        the gradient:

        .. math::
            \frac{\partial \phi_d}{\partial \mathbf{m}}

        Parameters
        ----------
        m : (n_param, ) numpy.ndarray
            The model for which the gradient is evaluated.

        Returns
        -------
        (n_param, ) numpy.ndarray
            The gradient of the data misfit function evaluated for the model provided.
        """

        if f is None:
            f = self.simulation.fields(m)

        return 2 * self.simulation.Jtvec(
            m, self.W.T * (self.W * self.residual(m, f=f)), f=f
        )

    @timeIt
    def deriv2(self, m, v, f=None):
        r"""Hessian of the data misfit function evaluated for the model provided.

        Where :math:`\phi_d (\mathbf{m})` is the data misfit function,
        this method returns the second-derivative (Hessian) with respect to the model parameters:

        .. math::
            \frac{\partial^2 \phi_d}{\partial \mathbf{m}^2}

        or the second-derivative (Hessian) multiplied by a vector :math:`(\mathbf{v})`:

        .. math::
            \frac{\partial^2 \phi_d}{\partial \mathbf{m}^2} \, \mathbf{v}

        Parameters
        ----------
        m : (n_param, ) numpy.ndarray
            The model for which the Hessian is evaluated.
        v : None or (n_param, ) numpy.ndarray, optional
            A vector.

        Returns
        -------
        (n_param, n_param) scipy.sparse.csr_matrix or (n_param, ) numpy.ndarray
            If the input argument *v* is ``None``, the Hessian of the data misfit
            function for the model provided is returned. If *v* is not ``None``,
            the Hessian multiplied by the vector provided is returned.
        """

        if f is None:
            f = self.simulation.fields(m)

        return 2 * self.simulation.Jtvec_approx(
            m, self.W * (self.W * self.simulation.Jvec_approx(m, v, f=f)), f=f
        )
