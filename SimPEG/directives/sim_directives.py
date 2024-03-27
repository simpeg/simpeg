import numpy as np
from ..regularization import BaseSimilarityMeasure
from ..utils import eigenvalue_by_power_iteration
from ..optimization import IterationPrinters, StoppingCriteria
from .directives import InversionDirective, SaveEveryIteration


###############################################################################
#                                                                             #
#              Directives of joint inversion                                  #
#                                                                             #
###############################################################################
class SimilarityMeasureInversionPrinters:
    betas = {
        "title": "betas",
        "value": lambda M: ["{:.2e}".format(elem) for elem in M.parent.betas],
        "width": 26,
        "format": "%s",
    }
    lambd = {
        "title": "lambda",
        "value": lambda M: M.parent.lambd,
        "width": 10,
        "format": "%1.2e",
    }
    phi_d_list = {
        "title": "phi_d",
        "value": lambda M: ["{:.2e}".format(elem) for elem in M.parent.phi_d_list],
        "width": 26,
        "format": "%s",
    }
    phi_m_list = {
        "title": "phi_m",
        "value": lambda M: ["{:.2e}".format(elem) for elem in M.parent.phi_m_list],
        "width": 26,
        "format": "%s",
    }
    phi_sim = {
        "title": "phi_sim",
        "value": lambda M: M.parent.phi_sim,
        "width": 10,
        "format": "%1.2e",
    }
    iterationCG = {
        "title": "iterCG",
        "value": lambda M: M.cg_count,
        "width": 10,
        "format": "%3d",
    }


class SimilarityMeasureInversionDirective(InversionDirective):
    """
    Directive for two model similiraty measure joint inversions. Sets Printers and
    StoppingCriteria.

    Notes
    -----
    Methods assume we are working with two models, and a single similarity measure.
    Also, the SimilarityMeasure objective function must be the last regularization.
    """

    printers = [
        IterationPrinters.iteration,
        SimilarityMeasureInversionPrinters.betas,
        SimilarityMeasureInversionPrinters.lambd,
        IterationPrinters.f,
        SimilarityMeasureInversionPrinters.phi_d_list,
        SimilarityMeasureInversionPrinters.phi_m_list,
        SimilarityMeasureInversionPrinters.phi_sim,
        SimilarityMeasureInversionPrinters.iterationCG,
    ]

    def initialize(self):
        if not isinstance(self.reg.objfcts[-1], BaseSimilarityMeasure):
            raise TypeError(
                f"The last regularization function must be an instance of "
                f"BaseSimilarityMeasure, got {type(self.reg.objfcts[-1])}."
            )

        # define relevant attributes
        self.betas = self.reg.multipliers[:-1]
        self.lambd = self.reg.multipliers[-1]
        self.phi_d_list = []
        self.phi_m_list = []
        self.phi_sim = 0.0

        # pass attributes to invProb
        self.invProb.betas = self.betas
        self.invProb.num_models = len(self.betas)
        self.invProb.lambd = self.lambd
        self.invProb.phi_d_list = self.phi_d_list
        self.invProb.phi_m_list = self.phi_m_list
        self.invProb.phi_sim = self.phi_sim

        self.opt.printers = self.printers
        self.opt.stoppers = [StoppingCriteria.iteration]

    def validate(self, directiveList):
        # check that this directive is first in the DirectiveList
        dList = directiveList.dList
        self_ind = dList.index(self)
        if self_ind != 0:
            raise IndexError(
                "The CrossGradientInversionDirective must be first in directive list."
            )
        return True

    def endIter(self):
        # compute attribute values
        phi_d = []
        for dmis in self.dmisfit.objfcts:
            phi_d.append(dmis(self.opt.xc))

        phi_m = []
        for reg in self.reg.objfcts:
            phi_m.append(reg(self.opt.xc))

        # pass attributes values to invProb
        self.invProb.phi_d_list = phi_d
        self.invProb.phi_m_list = phi_m[:-1]
        self.invProb.phi_sim = phi_m[-1]
        self.invProb.betas = self.reg.multipliers[:-1]
        # Assume last reg.objfct is the coupling
        self.invProb.lambd = self.reg.multipliers[-1]


class SimilarityMeasureSaveOutputEveryIteration(SaveEveryIteration):
    """
    SaveOutputEveryIteration for Joint Inversions.
    Saves information on the tradeoff parameters, data misfits, regularizations,
    coupling term, number of CG iterations, and value of cost function.
    """

    header = None
    save_txt = True
    betas = None
    phi_d = None
    phi_m = None
    phi_sim = None
    phi = None

    def initialize(self):
        if self.save_txt is True:
            print(
                "CrossGradientSaveOutputEveryIteration will save your inversion "
                "progress as: '###-{0!s}.txt'".format(self.fileName)
            )
            f = open(self.fileName + ".txt", "w")
            self.header = "  #          betas            lambda         joint_phi_d                joint_phi_m            phi_sim       iterCG     phi    \n"
            f.write(self.header)
            f.close()

        # Create a list of each
        self.betas = []
        self.lambd = []
        self.phi_d = []
        self.phi_m = []
        self.phi = []
        self.phi_sim = []

    def endIter(self):
        self.betas.append(["{:.2e}".format(elem) for elem in self.invProb.betas])
        self.phi_d.append(["{:.3e}".format(elem) for elem in self.invProb.phi_d_list])
        self.phi_m.append(["{:.3e}".format(elem) for elem in self.invProb.phi_m_list])
        self.lambd.append("{:.2e}".format(self.invProb.lambd))
        self.phi_sim.append(self.invProb.phi_sim)
        self.phi.append(self.opt.f)

        if self.save_txt:
            f = open(self.fileName + ".txt", "a")
            i = self.opt.iter
            f.write(
                " {0:2d}  {1}  {2}  {3}  {4}  {5:1.4e}  {6:d}  {7:1.4e}\n".format(
                    i,
                    self.betas[i - 1],
                    self.lambd[i - 1],
                    self.phi_d[i - 1],
                    self.phi_m[i - 1],
                    self.phi_sim[i - 1],
                    self.opt.cg_count,
                    self.phi[i - 1],
                )
            )
            f.close()

    def load_results(self):
        results = np.loadtxt(self.fileName + str(".txt"), comments="#")
        self.betas = results[:, 1]
        self.lambd = results[:, 2]
        self.phi_d = results[:, 3]
        self.phi_m = results[:, 4]
        self.phi_sim = results[:, 5]
        self.f = results[:, 7]


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
        rng = np.random.default_rng(seed=self.seed)

        if self.verbose:
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
                    seed=rng,
                )
            )

            reg_eigenvalues.append(
                eigenvalue_by_power_iteration(
                    reg,
                    m,
                    n_pw_iter=self.n_pw_iter,
                    seed=rng,
                )
            )

        self.ratios = np.array(dmis_eigenvalues) / np.array(reg_eigenvalues)
        self.invProb.betas = self.beta0_ratio * self.ratios
        self.reg.multipliers[:-1] = self.invProb.betas


class PairedBetaSchedule(InversionDirective):
    """
    Directive for beta cooling schedule to determine the tradeoff
    parameters when using paired data misfits and regularizations for a joint inversion.
    """

    chifact_target = 1.0
    beta_tol = 1e-1
    update_beta = True
    cooling_rate = 1
    cooling_factor = 2
    dmis_met = False

    @property
    def target(self):
        if getattr(self, "_target", None) is None:
            nD = np.array([survey.nD for survey in self.survey])

            self._target = nD * self.chifact_target

        return self._target

    @target.setter
    def target(self, val):
        self._target = val

    def initialize(self):
        self.dmis_met = np.zeros_like(self.invProb.betas, dtype=bool)

    def endIter(self):
        # Check if target misfit has been reached, if so, set dmis_met to True
        for i, phi_d in enumerate(self.invProb.phi_d_list):
            self.dmis_met[i] = phi_d < self.target[i]

        # check separately if misfits are within the tolerance,
        # otherwise, scale beta individually
        for i, phi_d in enumerate(self.invProb.phi_d_list):
            if self.opt.iter > 0 and self.opt.iter % self.cooling_rate == 0:
                target = self.target[i]
                ratio = phi_d / target
                if self.update_beta and ratio <= (1.0 + self.beta_tol):
                    if ratio <= 1:
                        ratio = np.maximum(0.75, ratio)
                    else:
                        ratio = np.minimum(1.5, ratio)

                    self.invProb.betas[i] /= ratio
                elif ratio > 1.0:
                    self.invProb.betas[i] /= self.cooling_factor

        self.reg.multipliers[:-1] = self.invProb.betas


class MovingAndMultiTargetStopping(InversionDirective):
    r"""
    Directive for setting stopping criteria for a joint inversion.
    Ensures both that all target misfits are met and there is a small change in the
    model. Computes the percentage change of the current model from the previous model.

    ..math::
        \frac {\| \mathbf{m_i} - \mathbf{m_{i-1}} \|} {\| \mathbf{m_{i-1}} \|}
    """

    tol = 1e-5
    beta_tol = 1e-1
    chifact_target = 1.0

    @property
    def target(self):
        if getattr(self, "_target", None) is None:
            nD = []
            for survey in self.survey:
                nD += [survey.nD]
            nD = np.array(nD)

            self._target = nD * self.chifact_target

        return self._target

    @target.setter
    def target(self, val):
        self._target = val

    def endIter(self):
        for phi_d, target in zip(self.invProb.phi_d_list, self.target):
            if np.abs(1.0 - phi_d / target) >= self.beta_tol:
                return
        if (
            np.linalg.norm(self.opt.xc - self.opt.x_last)
            / np.linalg.norm(self.opt.x_last)
            > self.tol
        ):
            return

        print(
            "stopping criteria met: ",
            np.linalg.norm(self.opt.xc - self.opt.x_last)
            / np.linalg.norm(self.opt.x_last),
        )
        self.opt.stopNextIteration = True
