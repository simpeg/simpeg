import numpy as np

from ..utils import sdiag, validate_integer, Zero
from ..utils.code_utils import (
    validate_type,
    validate_float,
)

from .base import InversionDirective


class UpdatePreconditioner(InversionDirective):
    """
    Create a Jacobi preconditioner for the linear problem
    """

    def __init__(self, update_every_iteration=True, **kwargs):
        super().__init__(**kwargs)
        self.update_every_iteration = update_every_iteration

    @property
    def update_every_iteration(self):
        """Whether to update the preconditioner at every iteration.

        Returns
        -------
        bool
        """
        return self._update_every_iteration

    @update_every_iteration.setter
    def update_every_iteration(self, value):
        self._update_every_iteration = validate_type(
            "update_every_iteration", value, bool
        )

    def initialize(self):
        # Create the pre-conditioner
        regDiag = np.zeros_like(self.invProb.model)
        m = self.invProb.model

        for reg in self.reg.objfcts:
            # Check if regularization has a projection
            rdg = reg.deriv2(m)
            if not isinstance(rdg, Zero):
                regDiag += rdg.diagonal()

        JtJdiag = np.zeros_like(self.invProb.model)
        for sim, dmisfit in zip(self.simulation, self.dmisfit.objfcts):
            if getattr(sim, "getJtJdiag", None) is None:
                assert getattr(sim, "getJ", None) is not None, (
                    "Simulation does not have a getJ attribute."
                    + "Cannot form the sensitivity explicitly"
                )
                JtJdiag += np.sum(np.power((dmisfit.W * sim.getJ(m)), 2), axis=0)
            else:
                JtJdiag += sim.getJtJdiag(m, W=dmisfit.W)

        diagA = JtJdiag + self.invProb.beta * regDiag
        diagA[diagA != 0] = diagA[diagA != 0] ** -1.0
        PC = sdiag((diagA))

        self.opt.approxHinv = PC

    def endIter(self):
        # Cool the threshold parameter
        if self.update_every_iteration is False:
            return

        # Create the pre-conditioner
        regDiag = np.zeros_like(self.invProb.model)
        m = self.invProb.model

        for reg in self.reg.objfcts:
            # Check if he has wire
            regDiag += reg.deriv2(m).diagonal()

        JtJdiag = np.zeros_like(self.invProb.model)
        for sim, dmisfit in zip(self.simulation, self.dmisfit.objfcts):
            if getattr(sim, "getJtJdiag", None) is None:
                assert getattr(sim, "getJ", None) is not None, (
                    "Simulation does not have a getJ attribute."
                    + "Cannot form the sensitivity explicitly"
                )
                JtJdiag += np.sum(np.power((dmisfit.W * sim.getJ(m)), 2), axis=0)
            else:
                JtJdiag += sim.getJtJdiag(m, W=dmisfit.W)

        diagA = JtJdiag + self.invProb.beta * regDiag
        diagA[diagA != 0] = diagA[diagA != 0] ** -1.0
        PC = sdiag((diagA))
        self.opt.approxHinv = PC


class BetaSchedule(InversionDirective):
    """Reduce trade-off parameter (beta) at successive iterations using a cooling schedule.

    Updates the **beta** property in the associated :class:`SimPEG.inverse_problem.BaseInvProblem`
    while the inversion is running.
    For linear least-squares problems, the optimization problem can be solved in a
    single step and the cooling rate can be set to *1*. For non-linear optimization
    problems, multiple steps are required obtain the minimizer for a fixed trade-off
    parameter. In this case, the cooling rate should be larger than 1.

    Parameters
    ----------
    coolingFactor : float
        The factor by which the trade-off parameter is decreased when updated.
        The preexisting value of the trade-off parameter is divided by the cooling factor.
    coolingRate : int
        Sets the number of successive iterations before the trade-off parameter is reduced.
        Use *1* for linear least-squares optimization problems. Use *2* for weakly non-linear
        optimization problems. Use *3* for general non-linear optimization problems.

    """

    def __init__(self, coolingFactor=8.0, coolingRate=3, **kwargs):
        super().__init__(**kwargs)
        self.coolingFactor = coolingFactor
        self.coolingRate = coolingRate

    @property
    def coolingFactor(self):
        """Beta is divided by this value every `coolingRate` iterations.

        Returns
        -------
        float
        """
        return self._coolingFactor

    @coolingFactor.setter
    def coolingFactor(self, value):
        self._coolingFactor = validate_float(
            "coolingFactor", value, min_val=0.0, inclusive_min=False
        )

    @property
    def coolingRate(self):
        """Cool after this number of iterations.

        Returns
        -------
        int
        """
        return self._coolingRate

    @coolingRate.setter
    def coolingRate(self, value):
        self._coolingRate = validate_integer("coolingRate", value, min_val=1)

    def endIter(self):
        if self.opt.iter > 0 and self.opt.iter % self.coolingRate == 0:
            if self.verbose:
                print(
                    "BetaSchedule is cooling Beta. Iteration: {0:d}".format(
                        self.opt.iter
                    )
                )
            self.invProb.beta /= self.coolingFactor
