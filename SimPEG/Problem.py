from __future__ import print_function
from . import Utils
from . import Survey
from . import Models
import numpy as np
from . import Maps
from .Fields import Fields, TimeFields
from . import Mesh
from . import Props
import properties


Solver = Utils.SolverUtils.Solver


class BaseProblem(Props.BaseSimPEG):
    """Problem is the base class for all geophysical forward problems
    in SimPEG.
    """

    _depreciate_maps = False
    _depreciate_main_map = None

    #: A SimPEG.Utils.Counter object
    counter = None

    #: A SimPEG.Survey Class
    surveyPair = Survey.BaseSurvey

    #: A SimPEG.Map Class
    mapPair = Maps.IdentityMap

    #: A SimPEG Solver class.
    Solver = Solver

    #: Solver options as a kwarg dict
    solverOpts = {}

    #: A SimPEG.Mesh instance.
    mesh = None

    model = Props.Model("Inversion model.")

    def __init__(self, mesh, **kwargs):
        if 'mapping' in kwargs:
            import warnings
            warnings.warn(
                'The `mapping` property has been depreciated, '
                'please use:\n\n\n'
                '\tfrom SimPEG import Depreciate\n'
                '\tDepreciate.use_old_mappings()\n\n\n'
                'To bring back old functionality.'
            )

        if self._depreciate_maps:
            mapping = kwargs.pop('mapping', None)
            if isinstance(mapping, Maps.IdentityMap):
                kwargs[self._depreciate_main_map] = mapping
            elif isinstance(mapping, list):
                # this is a prop map style
                for name, propmap in mapping:
                    kwargs['{}Map'.format(name)] = propmap
        super(BaseProblem, self).__init__(**kwargs)
        assert isinstance(mesh, Mesh.BaseMesh), (
            "mesh must be a SimPEG.Mesh object."
        )
        self.mesh = mesh

    @property
    def mapping(self):
        """setting an unnamed mapping has been depreciated.

        To use the old style Please use the Depreciate module.

        .. code::

            from SimPEG import Depreciate
            Depreciate.use_old_mappings()

        """
        raise Exception(
            'Depreciate: use `SimPEG.Depreciate.use_old_mappings()`'
        )

    @property
    def curModel(self):
        """
        Setting the curModel is depreciated.

        Use `SimPEG.Problem.model` instead.
        """
        raise AttributeError(
            'curModel is depreciated. Use `SimPEG.Problem.model` instead'
            )

    @curModel.setter
    def curModel(self, value):
        raise AttributeError(
            'curModel is depreciated. Use `SimPEG.Problem.model` instead'
            )

    @property
    def survey(self):
        """
        The survey object for this problem.
        """
        return getattr(self, '_survey', None)

    def pair(self, d):
        """Bind a survey to this problem instance using pointers."""
        assert isinstance(d, self.surveyPair), (
            "Data object must be an instance of a {0!s} class.".format(
                self.surveyPair.__name__
            )
        )
        if d.ispaired:
            raise Exception(
                "The survey object is already paired to a problem. "
                "Use survey.unpair()"
            )
        self._survey = d
        d._prob = self

    def unpair(self):
        """Unbind a survey from this problem instance."""
        if not self.ispaired:
            return
        self.survey._prob = None
        self._survey = None

    #: List of strings, e.g. ['_MeSigma', '_MeSigmaI']
    deleteTheseOnModelUpdate = []

    @properties.observer('model')
    def _on_model_update(self, value):
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
        """The field given the model.

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

    # surveyPair = Survey.LinearSurvey

    G = None

    def __init__(self, mesh, **kwargs):
        BaseProblem.__init__(self, mesh, **kwargs)
        self.mapping = kwargs.pop('mapping', Maps.IdentityMap(mesh))

    @property
    def mapping(self):
        "A SimPEG.Map instance."
        return getattr(self, '_mapping', None)

    @mapping.setter
    def mapping(self, val):
        val._assertMatchesPair(self.mapPair)
        self._mapping = val

    def fields(self, m):
        return self.G.dot(m)

    def Jvec(self, m, v, f=None):
        return self.G.dot(v)

    def Jtvec(self, m, v, f=None):
        return self.G.T.dot(v)
