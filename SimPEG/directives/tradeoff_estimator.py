import numpy as np

from ..regularization import BaseSimilarityMeasure
from ..utils import (
    eigenvalue_by_power_iteration,
)
from ..utils.code_utils import (
    validate_integer,
    validate_float,
)
from .base import InversionDirective


class BaseBetaEstimator(InversionDirective):
    """Base class for estimating initial trade-off parameter (beta).

    This class has properties and methods inherited by directive classes which estimate
    the initial trade-off parameter (beta). This class is not used directly to create
    directives for the inversion.

    Parameters
    ----------
    beta0_ratio : float
        Desired ratio between data misfit and model objective function at initial beta iteration.
    seed : int, None
        Seed used for random sampling.

    """

    def __init__(
        self,
        beta0_ratio=1.0,
        n_pw_iter=4,
        seed=None,
        method="power_iteration",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.beta0_ratio = beta0_ratio
        self.seed = seed

    @property
    def beta0_ratio(self):
        """The estimated ratio is multiplied by this to obtain beta.

        Returns
        -------
        float
        """
        return self._beta0_ratio

    @beta0_ratio.setter
    def beta0_ratio(self, value):
        self._beta0_ratio = validate_float(
            "beta0_ratio", value, min_val=0.0, inclusive_min=False
        )

    @property
    def seed(self):
        """Random seed to initialize with.

        Returns
        -------
        int
        """
        return self._seed

    @seed.setter
    def seed(self, value):
        if value is not None:
            value = validate_integer("seed", value, min_val=1)
        self._seed = value

    def validate(self, directive_list):
        ind = [isinstance(d, BaseBetaEstimator) for d in directive_list.dList]
        assert np.sum(ind) == 1, (
            "Multiple directives for computing initial beta detected in directives list. "
            "Only one directive can be used to set the initial beta."
        )

        return True


class BetaEstimateMaxDerivative(BaseBetaEstimator):
    r"""Estimate initial trade-off parameter (beta) using largest derivatives.

    The initial trade-off parameter (beta) is estimated by scaling the ratio
    between the largest derivatives in the gradient of the data misfit and
    model objective function. The estimated trade-off parameter is used to
    update the **beta** property in the associated :class:`SimPEG.inverse_problem.BaseInvProblem`
    object prior to running the inversion. A separate directive is used for updating the
    trade-off parameter at successive beta iterations; see :class:`BetaSchedule`.

    Parameters
    ----------
    beta0_ratio: float
        Desired ratio between data misfit and model objective function at initial beta iteration.
    seed : int, None
        Seed used for random sampling.

    Notes
    -----
    Let :math:`\phi_d` represent the data misfit, :math:`\phi_m` represent the model
    objective function and :math:`\mathbf{m_0}` represent the starting model. The first
    model update is obtained by minimizing the a global objective function of the form:

    .. math::
        \phi (\mathbf{m_0}) = \phi_d (\mathbf{m_0}) + \beta_0 \phi_m (\mathbf{m_0})

    where :math:`\beta_0` represents the initial trade-off parameter (beta).

    We define :math:`\gamma` as the desired ratio between the data misfit and model objective
    functions at the initial beta iteration (defined by the 'beta0_ratio' input argument).
    Here, the initial trade-off parameter is computed according to:

    .. math::
        \beta_0 = \gamma \frac{| \nabla_m \phi_d (\mathbf{m_0}) |_{max}}{| \nabla_m \phi_m (\mathbf{m_0 + \delta m}) |_{max}}

    where

    .. math::
        \delta \mathbf{m} = \frac{m_{max}}{\mu_{max}} \boldsymbol{\mu}

    and :math:`\boldsymbol{\mu}` is a set of independent samples from the
    continuous uniform distribution between 0 and 1.

    """

    def __init__(self, beta0_ratio=1.0, seed=None, **kwargs):
        super().__init__(beta0_ratio, seed, **kwargs)

    def initialize(self):
        if self.seed is not None:
            np.random.seed(self.seed)

        if self.verbose:
            print("Calculating the beta0 parameter.")

        m = self.invProb.model

        x0 = np.random.rand(*m.shape)
        phi_d_deriv = np.abs(self.dmisfit.deriv(m)).max()
        dm = x0 / x0.max() * m.max()
        phi_m_deriv = np.abs(self.reg.deriv(m + dm)).max()

        self.ratio = np.asarray(phi_d_deriv / phi_m_deriv)
        self.beta0 = self.beta0_ratio * self.ratio
        self.invProb.beta = self.beta0


class BetaEstimate_ByEig(BaseBetaEstimator):
    r"""Estimate initial trade-off parameter (beta) by power iteration.

    The initial trade-off parameter (beta) is estimated by scaling the ratio
    between the largest eigenvalue in the second derivative of the data
    misfit and the model objective function. The largest eigenvalues are estimated
    using the power iteration method; see :func:`SimPEG.utils.eigenvalue_by_power_iteration`.
    The estimated trade-off parameter is used to update the **beta** property in the
    associated :class:`SimPEG.inverse_problem.BaseInvProblem` object prior to running the inversion.
    Note that a separate directive is used for updating the trade-off parameter at successive
    beta iterations; see :class:`BetaSchedule`.

    Parameters
    ----------
    beta0_ratio: float
        Desired ratio between data misfit and model objective function at initial beta iteration.
    n_pw_iter : int
        Number of power iterations used to estimate largest eigenvalues.
    seed : int, None
        Seed used for random sampling.

    Notes
    -----
    Let :math:`\phi_d` represent the data misfit, :math:`\phi_m` represent the model
    objective function and :math:`\mathbf{m_0}` represent the starting model. The first
    model update is obtained by minimizing the a global objective function of the form:

    .. math::
        \phi (\mathbf{m_0}) = \phi_d (\mathbf{m_0}) + \beta_0 \phi_m (\mathbf{m_0})

    where :math:`\beta_0` represents the initial trade-off parameter (beta).
    Let :math:`\gamma` define the desired ratio between the data misfit and model
    objective functions at the initial beta iteration (defined by the 'beta0_ratio' input argument).
    Using the power iteration approach, our initial trade-off parameter is given by:

    .. math::
        \beta_0 = \gamma \frac{\lambda_d}{\lambda_m}

    where :math:`\lambda_d` as the largest eigenvalue of the Hessian of the data misfit, and
    :math:`\lambda_m` as the largest eigenvalue of the Hessian of the model objective function.
    For each Hessian, the largest eigenvalue is computed using power iteration. The input
    parameter 'n_pw_iter' sets the number of power iterations used in the estimate.

    For a description of the power iteration approach for estimating the larges eigenvalue,
    see :func:`SimPEG.utils.eigenvalue_by_power_iteration`.

    """

    def __init__(self, beta0_ratio=1.0, n_pw_iter=4, seed=None, **kwargs):
        super().__init__(beta0_ratio, seed, **kwargs)
        self.n_pw_iter = n_pw_iter

    @property
    def n_pw_iter(self):
        """Number of power iterations for estimating largest eigenvalues.

        Returns
        -------
        int
            Number of power iterations for estimating largest eigenvalues.
        """
        return self._n_pw_iter

    @n_pw_iter.setter
    def n_pw_iter(self, value):
        self._n_pw_iter = validate_integer("n_pw_iter", value, min_val=1)

    def initialize(self):
        if self.seed is not None:
            np.random.seed(self.seed)

        if self.verbose:
            print("Calculating the beta0 parameter.")

        m = self.invProb.model

        dm_eigenvalue = eigenvalue_by_power_iteration(
            self.dmisfit,
            m,
            n_pw_iter=self.n_pw_iter,
        )
        reg_eigenvalue = eigenvalue_by_power_iteration(
            self.reg,
            m,
            n_pw_iter=self.n_pw_iter,
        )

        self.ratio = np.asarray(dm_eigenvalue / reg_eigenvalue)
        self.beta0 = self.beta0_ratio * self.ratio
        self.invProb.beta = self.beta0


class PairedBetaEstimate_ByEig(InversionDirective):
    """
    Estimate the trade-off parameter, beta, between pairs of data misfit(s) and the
    regularization(s) as a multiple of the ratio between the highest eigenvalue of the
    data misfit term and the highest eigenvalue of the regularization.
    The highest eigenvalues are estimated through power iterations and Rayleigh
    quotient.

    Notes
    -----
    This class assumes the order of the data misfits for each model parameter match
    the order for the respective regularizations, i.e.

    >>> data_misfits = [phi_d_m1, phi_d_m2, phi_d_m3]
    >>> regs = [phi_m_m1, phi_m_m2, phi_m_m3]

    In which case it will estimate regularization parameters for each respective pair.
    """

    beta0_ratio = 1.0  #: the estimated ratio is multiplied by this to obtain beta
    n_pw_iter = 4  #: number of power iterations for estimation.
    seed = None  #: Random seed for the directive

    def initialize(self):
        r"""
        The initial beta is calculated by comparing the estimated
        eigenvalues of :math:`J^T J` and :math:`W^T W`.
        To estimate the eigenvector of **A**, we will use one iteration
        of the *Power Method*:

        .. math::

            \mathbf{x_1 = A x_0}

        Given this (very course) approximation of the eigenvector, we can
        use the *Rayleigh quotient* to approximate the largest eigenvalue.

        .. math::

            \lambda_0 = \frac{\mathbf{x^\top A x}}{\mathbf{x^\top x}}

        We will approximate the largest eigenvalue for both JtJ and WtW,
        and use some ratio of the quotient to estimate beta0.

        .. math::

            \beta_0 = \gamma \frac{\mathbf{x^\top J^\top J x}}{\mathbf{x^\top W^\top W x}}

        :rtype: float
        :return: beta0
        """
        if self.seed is not None:
            np.random.seed(self.seed)

        if self.debug:
            print("Calculating the beta0 parameter.")

        m = self.invProb.model
        dmis_eigenvalues = []
        reg_eigenvalues = []
        dmis_objs = self.dmisfit.objfcts
        reg_objs = [
            obj
            for obj in self.reg.objfcts
            if not isinstance(obj, BaseSimilarityMeasure)
        ]
        if len(dmis_objs) != len(reg_objs):
            raise ValueError(
                f"There must be the same number of data misfit and regularizations."
                f"Got {len(dmis_objs)} and {len(reg_objs)} respectively."
            )
        for dmis, reg in zip(dmis_objs, reg_objs):
            dmis_eigenvalues.append(
                eigenvalue_by_power_iteration(
                    dmis,
                    m,
                    n_pw_iter=self.n_pw_iter,
                )
            )

            reg_eigenvalues.append(
                eigenvalue_by_power_iteration(
                    reg,
                    m,
                    n_pw_iter=self.n_pw_iter,
                )
            )

        self.ratios = np.array(dmis_eigenvalues) / np.array(reg_eigenvalues)
        self.invProb.betas = self.beta0_ratio * self.ratios
        self.reg.multipliers[:-1] = self.invProb.betas
