import Utils, Survey, Problem, numpy as np, scipy.sparse as sp, gc


def _splitForward(forward):
    assert forward.ispaired, 'The problem and survey must be paired.'
    if isinstance(forward, Survey.BaseSurvey):
        survey = forward
        prob = forward.prob
    elif isinstance(forward, Problem.BaseProblem):
        prob = forward
        survey = forward.survey
    else:
        raise Exception('The forward simulation must either be a problem or a survey.')

    return prob, survey


class BaseDataMisfit(object):
    """BaseDataMisfit

        .. note::

            You should inherit from this class to create your own data misfit term.
    """

    __metaclass__ = Utils.SimPEGMetaClass

    debug   = False  #: Print debugging information
    counter = None   #: Set this to a SimPEG.Utils.Counter() if you want to count things

    def __init__(self):
        pass

    def splitForward(self, forward):
        """splitForward(forward)

            Split the forward simulation into a problem and a survey

            :param Problem,Survey forward: forward simulation
            :rtype: Problem,Survey
            :return: (prob, survey)

        """
        prob, survey = _splitForward(forward)
        return prob, survey

    @Utils.timeIt
    def dataObj(self, forward, m, u=None):
        """dataObj(forward, m, u=None)

            :param Problem,Survey forward: forward simulation
            :param numpy.array m: geophysical model
            :param numpy.array u: fields
            :rtype: float
            :return: data misfit

        """
        raise NotImplementedError('This method should be overwritten.')

    @Utils.timeIt
    def dataObjDeriv(self, forward, m, u=None):
        """dataObjDeriv(forward, m, u=None)

            :param Problem,Survey forward: forward simulation
            :param numpy.array m: geophysical model
            :param numpy.array u: fields
            :rtype: numpy.array
            :return: data misfit derivative

        """
        raise NotImplementedError('This method should be overwritten.')


    @Utils.timeIt
    def dataObj2Deriv(self, forward, m, v, u=None):
        """dataObj2Deriv(forward, m, v, u=None)

            :param Problem,Survey forward: forward simulation
            :param numpy.array m: geophysical model
            :param numpy.array v: vector to multiply
            :param numpy.array u: fields
            :rtype: numpy.array
            :return: data misfit derivative

        """
        raise NotImplementedError('This method should be overwritten.')


class l2_DataMisfit(object):
    """

    The data misfit with an l_2 norm:

    .. math::

        \mu_\\text{data} = {1\over 2}\left| \mathbf{W}_d (\mathbf{d}_\\text{pred} - \mathbf{d}_\\text{obs}) \\right|_2^2

    """

    def __init__(self, **kwargs):
        pass

    def getWd(self, survey):
        """getWd(survey)

            Get the data weighting matrix.

            This is based on the norm of the data plus a noise floor.

            :param Survey survey: geophysical survey
            :rtype: scipy.sparse.csr_matrix
            :return: Wd

        """
        eps = np.linalg.norm(Utils.mkvc(survey.dobs),2)*1e-5
        return Utils.sdiag(1/(abs(survey.dobs)*survey.std+eps))

    @Utils.timeIt
    def dataObj(self, forward, m, u=None):
        "dataObj2Deriv(forward, m, u=None)"
        prob, survey = _splitForward(forward)
        Wd = self.getWd(survey)
        R = Wd * survey.residual(m, u=u)
        return 0.5*np.vdot(R, R)

    @Utils.timeIt
    def dataObjDeriv(self, forward, m, u=None):
        "dataObj2Deriv(forward, m, u=None)"
        prob, survey = _splitForward(forward)
        if u is None: u = prob.fields(m)
        Wd = self.getWd(survey)
        return prob.Jtvec(m, Wd * (Wd * survey.residual(m, u=u)), u=u)

    @Utils.timeIt
    def dataObj2Deriv(self, forward, m, v, u=None):
        "dataObj2Deriv(forward, m, v, u=None)"
        prob, survey = _splitForward(forward)
        if u is None: u = prob.fields(m)
        Wd = self.getWd(survey)
        return prob.Jtvec_approx(m, Wd * (Wd * prob.Jvec_approx(m, v, u=u)), u=u)
