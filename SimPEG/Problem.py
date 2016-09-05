from __future__ import print_function
from . import Utils
from . import Survey
from . import Models
import numpy as np
from . import Maps
from .Fields import Fields, TimeFields
from . import Mesh


Solver = Utils.SolverUtils.Solver


class BaseProblem(object):
    """
        Problem is the base class for all geophysical forward problems in SimPEG.
    """

    counter = None   #: A SimPEG.Utils.Counter object

    surveyPair = Survey.BaseSurvey   #: A SimPEG.Survey Class
    mapPair    = Maps.IdentityMap    #: A SimPEG.Map Class

    Solver = Solver   #: A SimPEG Solver class.
    solverOpts = {}   #: Sovler options as a kwarg dict

    mesh    = None    #: A SimPEG.Mesh instance.

    PropMap = None    #: A SimPEG PropertyMap class.

    @property
    def mapping(self):
        "A SimPEG.Map instance or a property map is PropMap is not None"
        return getattr(self, '_mapping', None)
    @mapping.setter
    def mapping(self, val):
        if self.PropMap is None:
            val._assertMatchesPair(self.mapPair)
            self._mapping = val
        else:
            self._mapping = self.PropMap(val)

    def __init__(self, mesh, mapping=None, **kwargs):
        Utils.setKwargs(self, **kwargs)
        assert isinstance(mesh, Mesh.BaseMesh), "mesh must be a SimPEG.Mesh object."
        self.mesh = mesh
        self.mapping = mapping or Maps.IdentityMap(mesh)

    @property
    def survey(self):
        """
        The survey object for this problem.
        """
        return getattr(self, '_survey', None)

    def pair(self, d):
        """Bind a survey to this problem instance using pointers."""
        assert isinstance(d, self.surveyPair), "Data object must be an instance of a {0!s} class.".format((self.surveyPair.__name__))
        if d.ispaired:
            raise Exception("The survey object is already paired to a problem. Use survey.unpair()")
        self._survey = d
        d._prob = self

    def unpair(self):
        """Unbind a survey from this problem instance."""
        if not self.ispaired: return
        self.survey._prob = None
        self._survey = None


    deleteTheseOnModelUpdate = [] # List of strings, e.g. ['_MeSigma', '_MeSigmaI']

    @property
    def curModel(self):
        """
            Sets the current model, and removes dependent mass matrices.
        """
        return getattr(self, '_curModel', None)
    @curModel.setter
    def curModel(self, value):
        if value is self.curModel:
            return # it is the same!
        if self.PropMap is not None:
            self._curModel = self.mapping(value)
        else:
            self._curModel = Models.Model(value, self.mapping)
        for prop in self.deleteTheseOnModelUpdate:
            if hasattr(self, prop):
                delattr(self, prop)

    @property
    def ispaired(self):
        """True if the problem is paired to a survey."""
        return self.survey is not None

    @Utils.timeIt
    def Jvec(self, m, v, f=None):
        """Jvec(m, v, f=None)

            Effect of J(m) on a vector v.

            :param numpy.array m: model
            :param numpy.array v: vector to multiply
            :param Fields f: fields
            :rtype: numpy.array
            :return: Jv
        """
        raise NotImplementedError('J is not yet implemented.')

    @Utils.timeIt
    def Jtvec(self, m, v, f=None):
        """Jtvec(m, v, f=None)

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

    def fields(self, m):
        """
            The field given the model.

            :param numpy.array m: model
            :rtype: numpy.array
            :return: u, the fields

        """
        raise NotImplementedError('fields is not yet implemented.')


class BaseTimeProblem(BaseProblem):
    """Sets up that basic needs of a time domain problem."""

    @property
    def timeSteps(self):
        """Sets/gets the timeSteps for the time domain problem.

        You can set as an array of dt's or as a list of tuples/floats.
        Tuples must be length two with [..., (dt, repeat), ...]

        For example, the following setters are the same::

            prob.timeSteps = [(1e-6, 3), 1e-5, (1e-4, 2)]
            prob.timeSteps = np.r_[1e-6,1e-6,1e-6,1e-5,1e-4,1e-4]

        """
        return getattr(self, '_timeSteps', None)

    @timeSteps.setter
    def timeSteps(self, value):
        if isinstance(value, np.ndarray):
            self._timeSteps = value
            del self.timeMesh
            return

        self._timeSteps = Utils.meshTensor(value)
        del self.timeMesh

    @property
    def nT(self):
        "Number of time steps."
        return self.timeMesh.nC

    @property
    def t0(self):
        return getattr(self, '_t0', 0.0)
    @t0.setter
    def t0(self, value):
        assert Utils.isScalar(value), 't0 must be a scalar'
        del self.timeMesh
        self._t0 = float(value)

    @property
    def times(self):
        "Modeling times"
        return self.timeMesh.vectorNx

    @property
    def timeMesh(self):
        if getattr(self, '_timeMesh', None) is None:
            self._timeMesh = Mesh.TensorMesh([self.timeSteps], x0=[self.t0])
        return self._timeMesh
    @timeMesh.deleter
    def timeMesh(self):
        if hasattr(self, '_timeMesh'):
            del self._timeMesh

class LinearProblem(BaseProblem):

    surveyPair = Survey.LinearSurvey

    def __init__(self, mesh, G, **kwargs):
        BaseProblem.__init__(self, mesh, **kwargs)
        self.G = G

    def fields(self, m):
        return self.G.dot(m)

    def Jvec(self, m, v, f=None):
        return self.G.dot(v)

    def Jtvec(self, m, v, f=None):
        return self.G.T.dot(v)

