import numpy as np

from ..utils import (
    eigenvalue_by_power_iteration,
)
from ..utils.code_utils import (
    validate_integer,
    validate_float,
    validate_ndarray_with_shape,
)
from .base import InversionDirective
from .optimization import MultiTargetMisfits


class ScalingMultipleDataMisfits_ByEig(InversionDirective):
    """
    For multiple data misfits only: multiply each data misfit term
    by the inverse of its highest eigenvalue and then
    normalize the sum of the data misfit multipliers to one.
    The highest eigenvalue are estimated through power iterations and Rayleigh quotient.
    """

    def __init__(self, chi0_ratio=None, n_pw_iter=4, seed=None, **kwargs):
        super().__init__(**kwargs)
        self.chi0_ratio = chi0_ratio
        self.n_pw_iter = n_pw_iter
        self.seed = seed

    @property
    def chi0_ratio(self):
        """the estimated Alpha_smooth is multiplied by this ratio (int or array)

        Returns
        -------
        numpy.ndarray
        """
        return self._chi0_ratio

    @chi0_ratio.setter
    def chi0_ratio(self, value):
        if value is not None:
            value = validate_ndarray_with_shape("chi0_ratio", value, shape=("*",))
        self._chi0_ratio = value

    @property
    def n_pw_iter(self):
        """Number of power iterations for estimation.

        Returns
        -------
        int
        """
        return self._n_pw_iter

    @n_pw_iter.setter
    def n_pw_iter(self, value):
        self._n_pw_iter = validate_integer("n_pw_iter", value, min_val=1)

    @property
    def seed(self):
        """Random seed to initialize with

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

    def initialize(self):
        """"""
        if self.seed is not None:
            np.random.seed(self.seed)

        if self.verbose:
            print("Calculating the scaling parameter.")

        if (
            getattr(self.dmisfit, "objfcts", None) is None
            or len(self.dmisfit.objfcts) == 1
        ):
            raise TypeError(
                "ScalingMultipleDataMisfits_ByEig only applies to joint inversion"
            )

        ndm = len(self.dmisfit.objfcts)
        if self.chi0_ratio is not None:
            self.chi0_ratio = self.chi0_ratio * np.ones(ndm)
        else:
            self.chi0_ratio = self.dmisfit.multipliers

        m = self.invProb.model

        dm_eigenvalue_list = []
        for dm in self.dmisfit.objfcts:
            dm_eigenvalue_list += [eigenvalue_by_power_iteration(dm, m)]

        self.chi0 = self.chi0_ratio / np.r_[dm_eigenvalue_list]
        self.chi0 = self.chi0 / np.sum(self.chi0)
        self.dmisfit.multipliers = self.chi0

        if self.verbose:
            print("Scale Multipliers: ", self.dmisfit.multipliers)


class JointScalingSchedule(InversionDirective):
    """
    For multiple data misfits only: rebalance each data misfit term
    during the inversion when some datasets are fit, and others not
    using the ratios of current misfits and their respective target.
    It implements the strategy described in https://doi.org/10.1093/gji/ggaa378.
    """

    def __init__(
        self, warmingFactor=1.0, chimax=1e10, chimin=1e-10, update_rate=1, **kwargs
    ):
        super().__init__(**kwargs)
        self.mode = 1
        self.warmingFactor = warmingFactor
        self.chimax = chimax
        self.chimin = chimin
        self.update_rate = update_rate

    @property
    def mode(self):
        """The type of update to perform.

        Returns
        -------
        {1, 2}
        """
        return self._mode

    @mode.setter
    def mode(self, value):
        self._mode = validate_integer("mode", value, min_val=1, max_val=2)

    @property
    def warmingFactor(self):
        """Factor to adjust scaling of the data misfits by.

        Returns
        -------
        float
        """
        return self._warmingFactor

    @warmingFactor.setter
    def warmingFactor(self, value):
        self._warmingFactor = validate_float(
            "warmingFactor", value, min_val=0.0, inclusive_min=False
        )

    @property
    def chimax(self):
        """Maximum chi factor.

        Returns
        -------
        float
        """
        return self._chimax

    @chimax.setter
    def chimax(self, value):
        self._chimax = validate_float("chimax", value, min_val=0.0, inclusive_min=False)

    @property
    def chimin(self):
        """Minimum chi factor.

        Returns
        -------
        float
        """
        return self._chimin

    @chimin.setter
    def chimin(self, value):
        self._chimin = validate_float("chimin", value, min_val=0.0, inclusive_min=False)

    @property
    def update_rate(self):
        """Will update the data misfit scalings after this many iterations.

        Returns
        -------
        int
        """
        return self._update_rate

    @update_rate.setter
    def update_rate(self, value):
        self._update_rate = validate_integer("update_rate", value, min_val=1)

    def initialize(self):
        if (
            getattr(self.dmisfit, "objfcts", None) is None
            or len(self.dmisfit.objfcts) == 1
        ):
            raise TypeError("JointScalingSchedule only applies to joint inversion")

        targetclass = np.r_[
            [
                isinstance(dirpart, MultiTargetMisfits)
                for dirpart in self.inversion.directiveList.dList
            ]
        ]
        if ~np.any(targetclass):
            self.DMtarget = None
        else:
            self.targetclass = np.where(targetclass)[0][-1]
            self.DMtarget = self.inversion.directiveList.dList[
                self.targetclass
            ].DMtarget

        if self.verbose:
            print("Initial data misfit scales: ", self.dmisfit.multipliers)

    def endIter(self):
        self.dmlist = self.inversion.directiveList.dList[self.targetclass].dmlist

        if np.any(self.dmlist < self.DMtarget):
            self.mode = 2
        else:
            self.mode = 1

        if self.opt.iter > 0 and self.opt.iter % self.update_rate == 0:
            if self.mode == 2:
                if np.all(np.r_[self.dmisfit.multipliers] > self.chimin) and np.all(
                    np.r_[self.dmisfit.multipliers] < self.chimax
                ):
                    indx = self.dmlist > self.DMtarget
                    if np.any(indx):
                        multipliers = self.warmingFactor * np.median(
                            self.DMtarget[~indx] / self.dmlist[~indx]
                        )
                        if np.sum(indx) == 1:
                            indx = np.where(indx)[0][0]
                        self.dmisfit.multipliers[indx] *= multipliers
                        self.dmisfit.multipliers /= np.sum(self.dmisfit.multipliers)

                        if self.verbose:
                            print("Updating scaling for data misfits by ", multipliers)
                            print("New scales:", self.dmisfit.multipliers)
