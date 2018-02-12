from __future__ import print_function

import numpy as np
import discretize
import properties

from . import Utils
from . import Models
from . import Maps
from . import Props
# from . import Source
from .Data import Data
from .NewSurvey import BaseSurvey


class BaseSimulation(Props.HasModel):
    """
    BaseSimulation is the base class for all geophysical forward simulations in
    SimPEG.
    """

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

    mesh = properties.Instance(
        "a discretize mesh instance",
        discretize.BaseMesh
    )

    survey = properties.Instance(
        "a list of sources",
        BaseSurvey
    )

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

        return Data(
            dobs=dobs, survey=self.survey,
            standard_deviation=standard_deviation
        )



