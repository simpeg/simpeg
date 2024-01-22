from abc import abstractmethod
import numpy as np
from .utils import sdiag, Identity, validate_type
from .data import Data
from .simulation import BaseSimulation
from .objective_function import BaseObjectiveFunction

__all__ = ["L2DataMisfit"]


class BaseDataMisfit(BaseObjectiveFunction):
    """
    Base data misfit class.

    Parameters
    ----------
    data : SimPEG.data.Data
        A SimPEG data object.
    simulation : SimPEG.simulation.BaseSimulation
        A SimPEG simulation object.
    """

    def __init__(self, data, simulation):
        self.data = data
        self.simulation = simulation

    @property
    def data(self):
        """A SimPEG data object.

        Returns
        -------
        SimPEG.data.Data
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
        SimPEG.simulation.BaseSimulation
            A SimPEG simulation object.
        """
        return self._simulation

    @simulation.setter
    def simulation(self, value):
        self._simulation = validate_type(
            "simulation", value, BaseSimulation, cast=False
        )

    @abstractmethod
    def __call__(self, model, f=None) -> float:
        """
        Evaluate the objective function for a given model.
        """
        pass

    @abstractmethod
    def deriv(self, model):
        """
        Gradient of the objective function evaluated on a given model.
        """
        pass

    @abstractmethod
    def deriv2(self, model):
        """
        Hessian of the objective function evaluated on a given model.
        """
        pass

    @property
    def nP(self):
        """Number of model parameters.

        Returns
        -------
        int
            Number of model parameters.
        """
        if self.simulation.model is not None:
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
        """
        The data weighting matrix.

        Returns
        -------
        scipy.sparse.csr_matrix
            The data weighting matrix.
        """

        if getattr(self, "_W", None) is None:
            if self.data is None:
                raise TypeError(
                    "data with standard deviations must be set before the data "
                    "misfit can be constructed. Please set the data: "
                    "dmis.data = Data(dobs=dobs, relative_error=rel"
                    ", noise_floor=eps)"
                )
            standard_deviation = self.data.standard_deviation
            if standard_deviation is None:
                raise TypeError(
                    "data standard deviations must be set before the data misfit "
                    "can be constructed (data.relative_error = 0.05, "
                    "data.noise_floor = 1e-5), alternatively, the W matrix "
                    "can be set directly (dmisfit.W = 1./standard_deviation)"
                )
            if any(standard_deviation <= 0):
                raise ValueError(
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

        Where :math:`\mathbf{d}_\text{obs}` is the observed data vector and
        :math:`\mathbf{d}_\text{pred}` is the predicted data vector for a model
        vector :math:`\mathbf{m}`, this function computes the data residual:

        .. math::
            \mathbf{r} = \mathbf{d}_\text{pred} - \mathbf{d}_\text{obs}

        Parameters
        ----------
        m : (n_param, ) numpy.ndarray
            The model for which the function is evaluated.
        f : None or SimPEG.fields.Fields, optional
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
        \phi_d (\mathbf{m}) = \frac{1}{2} \big \| \mathbf{W_d}
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
    data : SimPEG.data.Data
        A SimPEG data object that has observed data and uncertainties.
    simulation : SimPEG.simulation.BaseSimulation
        A SimPEG simulation object.
    """

    def __call__(self, m, f=None):
        """Evaluate the residual for a given model."""

        R = self.W * self.residual(m, f=f)
        return 0.5 * np.vdot(R, R)

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

        return self.simulation.Jtvec(
            m, self.W.T * (self.W * self.residual(m, f=f)), f=f
        )

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

        return self.simulation.Jtvec_approx(
            m, self.W * (self.W * self.simulation.Jvec_approx(m, v, f=f)), f=f
        )
