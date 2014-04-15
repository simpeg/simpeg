import Utils, Survey, numpy as np, scipy.sparse as sp
import Maps

class BaseProblem(object):
    """
        Problem is the base class for all geophysical forward problems in SimPEG.
    """

    __metaclass__ = Utils.SimPEGMetaClass

    counter = None   #: A SimPEG.Utils.Counter object

    surveyPair = Survey.BaseSurvey   #: A SimPEG.Survey Class
    mapPair    = Maps.IdentityMap    #: A SimPEG.Map Class

    mapping = None    #: A SimPEG.Map instance.
    mesh    = None    #: A SimPEG.Mesh instance.

    def __init__(self, mesh, mapping=None, **kwargs):
        Utils.setKwargs(self, **kwargs)
        self.mesh = mesh
        self.mapping = mapping or Maps.IdentityMap(mesh)
        self.mapping._assertMatchesPair(self.mapPair)

    @property
    def survey(self):
        """
        The survey object for this problem.
        """
        return getattr(self, '_survey', None)

    def pair(self, d):
        """Bind a survey to this problem instance using pointers."""
        assert isinstance(d, self.surveyPair), "Data object must be an instance of a %s class."%(self.surveyPair.__name__)
        if d.ispaired:
            raise Exception("The survey object is already paired to a problem. Use survey.unpair()")
        self._survey = d
        d._prob = self

    def unpair(self):
        """Unbind a survey from this problem instance."""
        if not self.ispaired: return
        self.survey._prob = None
        self._survey = None

    @property
    def ispaired(self):
        """True if the problem is paired to a survey."""
        return self.survey is not None

    @Utils.timeIt
    def Jvec(self, m, v, u=None):
        """
            Effect of J(m) on a vector v.

            :param numpy.array m: model
            :param numpy.array v: vector to multiply
            :param numpy.array u: fields
            :rtype: numpy.array
            :return: Jv
        """
        raise NotImplementedError('J is not yet implemented.')

    @Utils.timeIt
    def Jtvec(self, m, v, u=None):
        """
            Effect of transpose of J(m) on a vector v.

            :param numpy.array m: model
            :param numpy.array v: vector to multiply
            :param numpy.array u: fields
            :rtype: numpy.array
            :return: JTv
        """
        raise NotImplementedError('Jt is not yet implemented.')


    @Utils.timeIt
    def Jvec_approx(self, m, v, u=None):
        """
            Approximate effect of J(m) on a vector v

            :param numpy.array m: model
            :param numpy.array v: vector to multiply
            :param numpy.array u: fields
            :rtype: numpy.array
            :return: approxJv
        """
        return self.Jvec(m, v, u)

    @Utils.timeIt
    def Jtvec_approx(self, m, v, u=None):
        """
            Approximate effect of transpose of J(m) on a vector v.

            :param numpy.array m: model
            :param numpy.array v: vector to multiply
            :param numpy.array u: fields
            :rtype: numpy.array
            :return: JTv
        """
        return self.Jtvec(m, v, u)

    def fields(self, m):
        """
            The field given the model.

            :param numpy.array m: model
            :rtype: numpy.array
            :return: u, the fields

        """
        raise NotImplementedError('fields is not yet implemented.')

    def createSyntheticSurvey(self, m, std=0.05, u=None, **survey_kwargs):
        """
            Create synthetic survey given a model, and a standard deviation.

            :param numpy.array m: geophysical model
            :param numpy.array std: standard deviation
            :param numpy.array u: fields for the given model (if pre-calculated)
            :param numpy.array survey_kwargs: Keyword arguments for initiating the survey.
            :rtype: SurveyObject
            :return: survey

            Returns the observed data with random Gaussian noise
            and Wd which is the same size as data, and can be used to weight the inversion.
        """
        survey = self.surveyPair(mtrue=m, **survey_kwargs)
        survey.pair(self)
        survey.dtrue = survey.dpred(m, u=u)
        noise = std*abs(survey.dtrue)*np.random.randn(*survey.dtrue.shape)
        survey.dobs = survey.dtrue+noise
        survey.std = survey.dobs*0 + std
        return survey



