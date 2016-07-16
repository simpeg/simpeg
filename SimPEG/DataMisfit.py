from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()
from past.utils import old_div
from builtins import object
from . import Utils, Survey, Problem
import numpy as np, scipy.sparse as sp, gc
from future.utils import with_metaclass


class BaseDataMisfit(with_metaclass(Utils.SimPEGMetaClass, object)):
    """BaseDataMisfit

        .. note::

            You should inherit from this class to create your own data misfit term.
    """

    debug   = False  #: Print debugging information
    counter = None   #: Set this to a SimPEG.Utils.Counter() if you want to count things

    def __init__(self, survey, **kwargs):
        assert survey.ispaired, 'The survey must be paired to a problem.'
        if isinstance(survey, Survey.BaseSurvey):
            self.survey = survey
            self.prob   = survey.prob
        Utils.setKwargs(self,**kwargs)

    @Utils.timeIt
    def eval(self, m, f=None):
        """eval(m, f=None)

            :param numpy.array m: geophysical model
            :param Fields f: fields
            :rtype: float
            :return: data misfit

        """
        raise NotImplementedError('This method should be overwritten.')

    @Utils.timeIt
    def evalDeriv(self, m, f=None):
        """evalDeriv(m, f=None)

            :param numpy.array m: geophysical model
            :param Fields f: fields
            :rtype: numpy.array
            :return: data misfit derivative

        """
        raise NotImplementedError('This method should be overwritten.')


    @Utils.timeIt
    def eval2Deriv(self, m, v, f=None):
        """eval2Deriv(m, v, f=None)

            :param numpy.array m: geophysical model
            :param numpy.array v: vector to multiply
            :param Fields f: fields
            :rtype: numpy.array
            :return: data misfit derivative

        """
        raise NotImplementedError('This method should be overwritten.')



class l2_DataMisfit(BaseDataMisfit):
    """

    The data misfit with an l_2 norm:

    .. math::

        \mu_\\text{data} = {1\over 2}\left| \mathbf{W}_d (\mathbf{d}_\\text{pred} - \mathbf{d}_\\text{obs}) \\right|_2^2

    """

    def __init__(self, survey, **kwargs):
        BaseDataMisfit.__init__(self, survey, **kwargs)

    @property
    def Wd(self):
        """getWd(survey)

            The data weighting matrix.

            The default is based on the norm of the data plus a noise floor.

            :rtype: scipy.sparse.csr_matrix
            :return: Wd

        """

        if getattr(self, '_Wd', None) is None:

            survey = self.survey

            if getattr(survey,'std', None) is None:
                print('SimPEG.DataMisfit.l2_DataMisfit assigning default std of 5%')
                survey.std = 0.05

            if getattr(survey, 'eps', None) is None:
                print('SimPEG.DataMisfit.l2_DataMisfit assigning default eps of 1e-5 * ||dobs||')
                survey.eps = np.linalg.norm(Utils.mkvc(survey.dobs),2)*1e-5

            self._Wd = Utils.sdiag(old_div(1,(abs(survey.dobs)*survey.std+survey.eps)))
        return self._Wd

    @Wd.setter
    def Wd(self, value):
        self._Wd = value

    @Utils.timeIt
    def eval(self, m, f=None):
        "eval(m, f=None)"
        if f is None: f = self.prob.fields(m)
        R = self.Wd * self.survey.residual(m, f)
        return 0.5*np.vdot(R, R)

    @Utils.timeIt
    def evalDeriv(self, m, f=None):
        "evalDeriv(m, f=None)"
        if f is None: f = self.prob.fields(m)
        return self.prob.Jtvec(m, self.Wd * (self.Wd * self.survey.residual(m, f=f)), f=f)

    @Utils.timeIt
    def eval2Deriv(self, m, v, f=None):
        "eval2Deriv(m, v, f=None)"
        if f is None: f = self.prob.fields(m)
        return self.prob.Jtvec_approx(m, self.Wd * (self.Wd * self.prob.Jvec_approx(m, v, f=f)), f=f)
