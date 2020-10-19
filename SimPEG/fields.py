from six import string_types
import numpy as np
import properties

from .simulation import BaseSimulation, BaseTimeSimulation
from .utils import mkvc


class Fields(properties.HasProperties):
    """Fancy Field Storage
    .. code::python
        fields = Fields(
            simulation=simulation, knownFields={"phi": "CC"}
        )
        fields[:,'phi'] = phi
        print(fields[src0,'phi'])
    """

    simulation = properties.Instance("a SimPEG simulation", BaseSimulation)

    knownFields = properties.Dictionary(
        """
        a dictionary with the names of the know fields and their location on
        a mesh e.g. {"e": "E", "phi": "CC"}
        """,
        required=True,
    )

    aliasFields = properties.Dictionary(
        """
        a dictionary of the aliased fields with [alias, location, function],
        e.g. {"b":["e","F",lambda(F,e,ind)]}
        """,
        default={},
    )
    #: dtype is the type of the storage matrix. This can be a dictionary.
    dtype = float

    def __init__(self, simulation=None, **kwargs):
        super(Fields, self).__init__(**kwargs)
        if simulation is not None:
            self.simulation = simulation
        self._fields = {}
        self.startup()

    @properties.validator("knownFields")
    def _check_overlap_with_aliased(self, change):
        allFields = [k for k in change["value"]] + [a for a in self.aliasFields]
        assert len(allFields) == len(
            set(allFields)
        ), "Aliased fields and Known Fields have overlapping definitions."

    @properties.validator("aliasFields")
    def _check_overlap_with_known(self, change):
        allFields = [k for k in self.knownFields] + [a for a in change["value"]]
        assert len(allFields) == len(
            set(allFields)
        ), "Aliased fields and Known Fields have overlapping definitions."

    @property
    def mesh(self):
        return self.simulation.mesh

    @property
    def survey(self):
        return self.simulation.survey

    def startup(self):
        pass

    @property
    def approxSize(self):
        """The approximate cost to storing all of the known fields."""
        sz = 0.0
        for f in self.knownFields:
            loc = self.knownFields[f]
            sz += np.array(self._storageShape(loc)).prod() * 8.0 / (1024 ** 2)
        return "{0:e} MB".format(sz)

    def _storageShape(self, loc):
        nSrc = self.survey.nSrc

        nP = {
            "CC": self.mesh.nC,
            "N": self.mesh.nN,
            "F": self.mesh.nF,
            "E": self.mesh.nE,
        }[loc]

        return (nP, nSrc)

    def _initStore(self, name):
        if name in self._fields:
            return self._fields[name]

        assert name in self.knownFields, "field name is not known."

        loc = self.knownFields[name]

        if isinstance(self.dtype, dict):
            dtype = self.dtype[name]
        else:
            dtype = self.dtype

        # field = zarr.create(self._storageShape(loc), dtype=dtype)
        field = np.zeros(self._storageShape(loc), dtype=dtype)

        self._fields[name] = field

        return field

    def _srcIndex(self, srcTestList):
        if type(srcTestList) is slice:
            ind = srcTestList
        else:
            ind = self.survey.getSourceIndex(srcTestList)
        return ind

    def _nameIndex(self, name, accessType):

        if type(name) is slice:
            assert name == slice(
                None, None, None
            ), "Fancy field name slicing is not supported... yet."
            name = None

        if name is None:
            return
        if accessType == "set" and name not in self.knownFields:
            if name in self.aliasFields:
                raise KeyError(
                    "Invalid field name ({0!s}) for setter, you can't "
                    "set an aliased property".format(name)
                )
            else:
                raise KeyError("Invalid field name ({0!s}) for setter".format(name))

        elif accessType == "get" and (
            name not in self.knownFields and name not in self.aliasFields
        ):
            raise KeyError("Invalid field name ({0!s}) for getter".format(name))
        return name

    def _indexAndNameFromKey(self, key, accessType):
        if not isinstance(key, tuple):
            key = (key,)
        if len(key) == 1:
            key += (None,)

        assert len(key) == 2, "must be [Src, fieldName]"

        srcTestList, name = key
        name = self._nameIndex(name, accessType)
        ind = self._srcIndex(srcTestList)
        return ind, name

    def __setitem__(self, key, value):
        ind, name = self._indexAndNameFromKey(key, "set")
        if name is None:
            assert (
                isinstance(value, dict)
            ), "New fields must be a dictionary, if field is not specified."
            newFields = value
        elif name in self.knownFields:
            newFields = {name: value}
        else:
            raise Exception("Unknown setter")

        for name in newFields:
            field = self._initStore(name)
            self._setField(field, newFields[name], name, ind)

    def __getitem__(self, key):
        ind, name = self._indexAndNameFromKey(key, "get")
        if name is None:
            out = {}
            for name in self._fields:
                out[name] = self._getField(name, ind)
            return out
        return self._getField(name, ind)

    def _setField(self, field, val, name, ind):
        if isinstance(val, np.ndarray) and (
            field.shape[0] == field.size or val.ndim == 1
        ):
            val = mkvc(val, 2)
        field[:, ind] = val

    def _getField(self, name, ind):
        if name in self._fields:
            out = self._fields[name][:, ind]
        else:
            # Aliased fields
            alias, loc, func = self.aliasFields[name]

            srcII = np.array(self.survey.source_list)[ind]
            srcII = srcII.tolist()

            if isinstance(func, string_types):
                assert hasattr(self, func), (
                    "The alias field function is a string, but it does not "
                    "exist in the Fields class."
                )
                func = getattr(self, func)
            out = func(self._fields[alias][:, ind], srcII)
        if out.shape[0] == out.size or out.ndim == 1:
            out = mkvc(out, 2)
        return out

    def __contains__(self, other):
        if other in self.aliasFields:
            other = self.aliasFields[other][0]
        return self._fields.__contains__(other)


class TimeFields(Fields):
    """Fancy Field Storage for time domain problems
    .. code:: python

        fields = TimeFields(simulation=simulation, knownFields={'phi':'CC'})
        fields[:,'phi', timeInd] = phi
        print(fields[src0,'phi'])
    """

    simulation = properties.Instance("a SimPEG time simulation", BaseTimeSimulation)

    def __init__(self, simulation=None, **kwargs):
        super(TimeFields, self).__init__(simulation=simulation, **kwargs)

    def _storageShape(self, loc):
        nP = {
            "CC": self.mesh.nC,
            "N": self.mesh.nN,
            "F": self.mesh.nF,
            "E": self.mesh.nE,
        }[loc]
        nSrc = self.survey.nSrc
        nT = self.simulation.nT + 1
        return (nP, nSrc, nT)

    def _indexAndNameFromKey(self, key, accessType):
        if not isinstance(key, tuple):
            key = (key,)
        if len(key) == 1:
            key += (None,)
        if len(key) == 2:
            key += (slice(None, None, None),)

        assert len(key) == 3, "must be [Src, fieldName, times]"

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
        nT = np.ones(total_nT, dtype=bool)[timeInd].sum()
        shape = nP, nSrc, nT
        if deflate:
            shape = tuple([s for s in shape if s > 1])
        if len(shape) == 1:
            shape = shape + (1,)
        return shape

    def _setField(self, field, val, name, ind):
        srcInd, timeInd = ind
        shape = self._correctShape(name, ind)
        if isinstance(val, np.ndarray) and val.size == 1:
            val = val[0]
        if np.isscalar(val):
            field[:, srcInd, timeInd] = val
            return
        if val.size != np.array(shape).prod():
            raise ValueError("Incorrect size for data.")
        correctShape = field[:, srcInd, timeInd].shape
        field[:, srcInd, timeInd] = val.reshape(correctShape, order="F")

    def _getField(self, name, ind):
        srcInd, timeInd = ind

        if name in self._fields:
            out = self._fields[name][:, srcInd, timeInd]
        else:
            # Aliased fields
            alias, loc, func = self.aliasFields[name]
            if isinstance(func, string_types):
                assert hasattr(self, func), (
                    "The alias field function is a string, but it does "
                    "not exist in the Fields class."
                )
                func = getattr(self, func)
            pointerFields = self._fields[alias][:, srcInd, timeInd]
            pointerShape = self._correctShape(alias, ind)
            pointerFields = pointerFields.reshape(pointerShape, order="F")

            # First try to return the function as three arguments (without timeInd)
            if timeInd == slice(None, None, None):
                try:
                    # assume it will take care of integrating over all times
                    return func(pointerFields, srcInd)
                except TypeError:
                    pass

            timeII = np.arange(self.simulation.nT + 1)[timeInd]
            srcII = np.array(self.survey.source_list)[srcInd]
            srcII = srcII.tolist()

            if timeII.size == 1:
                pointerShapeDeflated = self._correctShape(alias, ind, deflate=True)
                pointerFields = pointerFields.reshape(pointerShapeDeflated, order="F")
                out = func(pointerFields, srcII, timeII)
            else:  # loop over the time steps
                nT = pointerShape[2]
                out = list(range(nT))
                for i, TIND_i in enumerate(timeII):
                    fieldI = pointerFields[:, :, i]
                    if fieldI.shape[0] == fieldI.size:
                        fieldI = mkvc(fieldI, 2)
                    out[i] = func(fieldI, srcII, TIND_i)
                    if out[i].ndim == 1:
                        out[i] = out[i][:, np.newaxis, np.newaxis]
                    elif out[i].ndim == 2:
                        out[i] = out[i][:, :, np.newaxis]
                out = np.concatenate(out, axis=2)

        shape = self._correctShape(name, ind, deflate=True)
        return out.reshape(shape, order="F")
