import Utils, Survey, Models, numpy as np, scipy.sparse as sp
Solver = Utils.SolverUtils.Solver
import Maps, Mesh


class Fields(object):
    """Fancy Field Storage

        u[:,'phi'] = phi
        print u[src0,'phi']

    """

    knownFields = None  #: Known fields,   a dict with locations,         e.g. {"e": "E", "phi": "CC"}
    aliasFields = None  #: Aliased fields, a dict with [alias, location, function], e.g. {"b":["e","F",lambda(F,e,ind)]}
    dtype = float       #: dtype is the type of the storage matrix. This can be a dictionary.

    def __init__(self, mesh, survey, **kwargs):
        self.survey = survey
        self.mesh = mesh
        Utils.setKwargs(self, **kwargs)
        self._fields = {}

        if self.knownFields is None:
            raise Exception('knownFields cannot be set to None')
        if self.aliasFields is None:
            self.aliasFields = {}

        allFields = [k for k in self.knownFields] + [a for a in self.aliasFields]
        assert len(allFields) == len(set(allFields)), 'Aliased fields and Known Fields have overlapping definitions.'
        self.startup()

    def startup(self):
        pass

    @property
    def approxSize(self):
        """The approximate cost to storing all of the known fields."""
        sz = 0.0
        for f in self.knownFields:
            loc =self.knownFields[f]
            sz += np.array(self._storageShape(loc)).prod()*8.0/(1024**2)
        return "%e MB"%sz

    def _storageShape(self, loc):
        nSrc = self.survey.nSrc

        nP = {'CC': self.mesh.nC,
              'N':  self.mesh.nN,
              'F':  self.mesh.nF,
              'E':  self.mesh.nE}[loc]

        return (nP, nSrc)

    def _initStore(self, name):
        if name in self._fields:
            return self._fields[name]

        assert name in self.knownFields, 'field name is not known.'

        loc = self.knownFields[name]

        if type(self.dtype) is dict:
            dtype = self.dtype[name]
        else:
            dtype = self.dtype
        field = np.zeros(self._storageShape(loc), dtype=dtype)

        self._fields[name] = field

        return field

    def _srcIndex(self, srcTestList):
        if type(srcTestList) is slice:
            ind = srcTestList
        else:
            if type(srcTestList) is not list:
                srcTestList = [srcTestList]
            for srcTest in srcTestList:
                if srcTest not in self.survey.srcList:
                    raise KeyError('Invalid Source, not in survey list.')

            ind = np.in1d(self.survey.srcList, srcTestList)
        return ind

    def _nameIndex(self, name, accessType):

        if type(name) is slice:
            assert name == slice(None,None,None), 'Fancy field name slicing is not supported... yet.'
            name = None

        if name is None:
            return
        if accessType=='set' and name not in self.knownFields:
            if name in self.aliasFields:
                raise KeyError("Invalid field name (%s) for setter, you can't set an aliased property"%name)
            else:
                raise KeyError('Invalid field name (%s) for setter'%name)

        elif accessType=='get' and (name not in self.knownFields and name not in self.aliasFields):
            raise KeyError('Invalid field name (%s) for getter'%name)
        return name

    def _indexAndNameFromKey(self, key, accessType):
        if type(key) is not tuple:
            key = (key,)
        if len(key) == 1:
            key += (None,)

        assert len(key) == 2, 'must be [Src, fieldName]'

        srcTestList, name = key
        name = self._nameIndex(name, accessType)
        ind = self._srcIndex(srcTestList)
        return ind, name

    def __setitem__(self, key, value):
        ind, name = self._indexAndNameFromKey(key, 'set')
        if name is None:
            freq = key
            assert type(value) is dict, 'New fields must be a dictionary, if field is not specified.'
            newFields = value
        elif name in self.knownFields:
            newFields = {name: value}
        else:
            raise Exception('Unknown setter')

        for name in newFields:
            field = self._initStore(name)
            self._setField(field, newFields[name], name, ind)

    def __getitem__(self, key):
        ind, name = self._indexAndNameFromKey(key, 'get')
        if name is None:
            out = {}
            for name in self._fields:
                out[name] = self._getField(name, ind)
            return out
        return self._getField(name, ind)

    def _setField(self, field, val, name, ind):
        if isinstance(val, np.ndarray) and (field.shape[1] == 1 or val.ndim == 1):
            val = Utils.mkvc(val,2)
        field[:,ind] = val

    def _getField(self, name, ind):
        if name in self._fields:
            out = self._fields[name][:,ind]
        else:
            # Aliased fields
            alias, loc, func = self.aliasFields[name]

            srcII   = np.array(self.survey.srcList)[ind]
            if isinstance(srcII, np.ndarray):
                srcII = srcII.tolist()
            if len(srcII) == 1:
                srcII = srcII[0]

            if type(func) is str:
                assert hasattr(self, func), 'The alias field function is a string, but it does not exist in the Fields class.'
                func = getattr(self, func)
            out = func(self._fields[alias][:,ind], srcII)
        if out.shape[0] == out.size:
            out = Utils.mkvc(out)
        return out

    def __contains__(self, other):
        if other in self.aliasFields:
            other = self.aliasFields[other][0]
        return self._fields.__contains__(other)


class TimeFields(Fields):
    """Fancy Field Storage for time domain problems

        u[:,'phi', timeInd] = phi
        print u[src0,'phi']

    """

    def _storageShape(self, loc):
        nP = {'CC': self.mesh.nC,
              'N':  self.mesh.nN,
              'F':  self.mesh.nF,
              'E':  self.mesh.nE}[loc]
        nSrc = self.survey.nSrc
        nT = self.survey.prob.nT + 1
        return (nP, nSrc, nT)

    def _indexAndNameFromKey(self, key, accessType):
        if type(key) is not tuple:
            key = (key,)
        if len(key) == 1:
            key += (None,)
        if len(key) == 2:
            key += (slice(None,None,None),)

        assert len(key) == 3, 'must be [Src, fieldName, times]'

        srcTestList, name, timeInd = key

        name = self._nameIndex(name, accessType)
        srcInd = self._srcIndex(srcTestList)

        return (srcInd, timeInd), name

    def _correctShape(self, name, ind, deflate=False):
        srcInd, timeInd = ind
        if name in self.knownFields:
            loc = self.knownFields[name]
        else:
            loc = self.aliasFields[name][1]
        nP, total_nSrc, total_nT = self._storageShape(loc)
        nSrc = np.ones(total_nSrc, dtype=bool)[srcInd].sum()
        nT  = np.ones(total_nT, dtype=bool)[timeInd].sum()
        shape = nP, nSrc, nT
        if deflate:
             shape = tuple([s for s in shape if s > 1])
        return shape

    def _setField(self, field, val, name, ind):
        srcInd, timeInd = ind
        shape = self._correctShape(name, ind)
        if Utils.isScalar(val):
            field[:,srcInd,timeInd] = val
            return
        if val.size != np.array(shape).prod():
            raise ValueError('Incorrect size for data.')
        correctShape = field[:,srcInd,timeInd].shape
        field[:,srcInd,timeInd] = val.reshape(correctShape, order='F')

    def _getField(self, name, ind):
        srcInd, timeInd = ind

        if name in self._fields:
            out = self._fields[name][:,srcInd,timeInd]
        else:
            # Aliased fields
            alias, loc, func = self.aliasFields[name]
            if type(func) is str:
                assert hasattr(self, func), 'The alias field function is a string, but it does not exist in the Fields class.'
                func = getattr(self, func)
            pointerFields = self._fields[alias][:,srcInd,timeInd]
            pointerShape = self._correctShape(alias, ind)
            pointerFields = pointerFields.reshape(pointerShape, order='F')

            timeII = np.arange(self.survey.prob.nT + 1)[timeInd]
            srcII   = np.array(self.survey.srcList)[srcInd]
            if isinstance(srcII, np.ndarray):
                srcII = srcII.tolist()
            if len(srcII) == 1:
                srcII = srcII[0]

            if timeII.size == 1:
                pointerShapeDeflated = self._correctShape(alias, ind, deflate=True)
                pointerFields = pointerFields.reshape(pointerShapeDeflated, order='F')
                out = func(pointerFields, srcII, timeII)
            else: #loop over the time steps
                nT = pointerShape[2]
                out = range(nT)
                for i, TIND_i in enumerate(timeII):
                    fieldI = pointerFields[:,:,i]
                    if fieldI.ndim == 2 and fieldI.shape[1] == 1:
                        fieldI = Utils.mkvc(fieldI)
                    out[i] = func(fieldI, srcII, TIND_i)
                    if out[i].ndim == 1:
                        out[i] = out[i][:,np.newaxis,np.newaxis]
                    elif out[i].ndim == 2:
                        out[i] = out[i][:,:,np.newaxis]
                out = np.concatenate(out, axis=2)

        shape = self._correctShape(name, ind, deflate=True)
        return out.reshape(shape, order='F')



class BaseProblem(object):
    """
        Problem is the base class for all geophysical forward problems in SimPEG.
    """

    __metaclass__ = Utils.SimPEGMetaClass

    counter = None   #: A SimPEG.Utils.Counter object

    surveyPair = Survey.BaseSurvey   #: A SimPEG.Survey Class
    mapPair    = Maps.IdentityMap    #: A SimPEG.Map Class

    Solver = Solver   #: A SimPEG Solver class.
    solverOpts = {}   #: Sovler options as a kwarg dict

    mapping = None    #: A SimPEG.Map instance.
    mesh    = None    #: A SimPEG.Mesh instance.

    def __init__(self, mesh, mapping=None, **kwargs):
        Utils.setKwargs(self, **kwargs)
        assert isinstance(mesh, Mesh.BaseMesh), "mesh must be a SimPEG.Mesh object."
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
        self._curModel = Models.Model(value, self.mapping)
        for prop in self.deleteTheseOnModelUpdate:
            if hasattr(self, prop):
                delattr(self, prop)

    @property
    def ispaired(self):
        """True if the problem is paired to a survey."""
        return self.survey is not None

    @Utils.timeIt
    def Jvec(self, m, v, u=None):
        """Jvec(m, v, u=None)

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
        """Jtvec(m, v, u=None)

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
        """Jvec_approx(m, v, u=None)

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
        """Jtvec_approx(m, v, u=None)

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



