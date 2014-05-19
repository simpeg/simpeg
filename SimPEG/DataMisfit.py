import Utils, Survey, Problem, numpy as np, scipy.sparse as sp, gc


class BaseDataMisfit(object):
    """BaseDataMisfit

        .. note::

            You should inherit from this class to create your own data misfit term.
    """

    __metaclass__ = Utils.SimPEGMetaClass

    debug   = False  #: Print debugging information
    counter = None   #: Set this to a SimPEG.Utils.Counter() if you want to count things

    def __init__(self, survey, **kwargs):
        assert survey.ispaired, 'The survey must be paired to a problem.'
        if isinstance(survey, Survey.BaseSurvey):
            self.survey = survey
            self.prob   = survey.prob
        Utils.setKwargs(self,**kwargs)

    @Utils.timeIt
    def eval(self, m, u=None):
        """eval(m, u=None)

            :param Problem,Survey forward: forward simulation
            :param numpy.array m: geophysical model
            :param numpy.array u: fields
            :rtype: float
            :return: data misfit

        """
        raise NotImplementedError('This method should be overwritten.')

    @Utils.timeIt
    def evalDeriv(self, m, u=None):
        """evalDeriv(m, u=None)

            :param Problem,Survey forward: forward simulation
            :param numpy.array m: geophysical model
            :param numpy.array u: fields
            :rtype: numpy.array
            :return: data misfit derivative

        """
        raise NotImplementedError('This method should be overwritten.')


    @Utils.timeIt
    def eval2Deriv(self, m, v, u=None):
        """eval2Deriv(m, v, u=None)

            :param Problem,Survey forward: forward simulation
            :param numpy.array m: geophysical model
            :param numpy.array v: vector to multiply
            :param numpy.array u: fields
            :rtype: numpy.array
            :return: data misfit derivative

        """
        raise NotImplementedError('This method should be overwritten.')

    def target(self, forward):
        """target(forward)

            Target for data misfit. By default this is the number of data,
            which satisfies the Discrepancy Principle.

            :param Problem,Survey forward: forward simulation
            :rtype: float
            :return: data misfit target

        """
        prob, survey = self.splitForward(forward)
        return survey.nD


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
            print 'SimPEG.l2_DataMisfit is creating default weightings for Wd.'
            survey = self.survey
            eps = np.linalg.norm(Utils.mkvc(survey.dobs),2)*1e-5
            self._Wd = Utils.sdiag(1/(abs(survey.dobs)*survey.std+eps))
        return self._Wd

    @Wd.setter
    def Wd(self, value):
        self._Wd = value

    @Utils.timeIt
    def eval(self, m, u=None):
        "eval(m, u=None)"
        prob   = self.prob
        survey = self.survey
        R = self.Wd * survey.residual(m, u=u)
        return 0.5*np.vdot(R, R)

    @Utils.timeIt
    def evalDeriv(self, m, u=None):
        "evalDeriv(m, u=None)"
        prob   = self.prob
        survey = self.survey
        if u is None: u = prob.fields(m)
        return prob.Jtvec(m, self.Wd * (self.Wd * survey.residual(m, u=u)), u=u)

    @Utils.timeIt
    def eval2Deriv(self, m, v, u=None):
        "eval2Deriv(m, v, u=None)"
        prob   = self.prob
        if u is None: u = prob.fields(m)
        return prob.Jtvec_approx(m, self.Wd * (self.Wd * prob.Jvec_approx(m, v, u=u)), u=u)
