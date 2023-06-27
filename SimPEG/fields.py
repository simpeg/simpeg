import numpy as np

from .simulation import BaseSimulation, BaseTimeSimulation
from .utils import mkvc, validate_type


class Fields:
    """Fancy Field Storage

    Examples
    --------
    >>> fields = Fields(
    ...     simulation=simulation, knownFields={"phi": "CC"}
    ... )
    >>> fields[:,'phi'] = phi
    """

    _dtype = float
    _knownFields = {}
    _aliasFields = {}

    def __init__(
        self, simulation, knownFields=None, aliasFields=None, dtype=None, **kwargs
    ):
        super().__init__(**kwargs)
        self.simulation = simulation

        if knownFields is not None:
            knownFields = validate_type("knownFields", knownFields, dict, cast=False)
            self._knownFields = knownFields
        if aliasFields is not None:
            aliasFields = validate_type("aliasFields", aliasFields, dict, cast=False)
            self._aliasFields = aliasFields
        if dtype is not None:
            self._dtype = dtype

        # check overlapping fields
        if any(key in self.aliasFields for key in self.knownFields):
            raise KeyError(
                "Aliased fields and Known Fields have overlapping definitions."
            )

        self._fields = {}
        self.startup()

    @property
    def simulation(self):
        """The simulation object that created these fields

        Returns
        -------
        SimPEG.simulation.BaseSimulation
        """
        return self._simulation

    @simulation.setter
    def simulation(self, value):
        self._simulation = validate_type(
            "simulation", value, BaseSimulation, cast=False
        )

    @property
    def knownFields(self):
        """The known fields of this object.

        The dictionary representing the known fields and their locations on the simulation
        mesh. The keys are the names of the fields, and the values are the location on
        the mesh.

        >>> fields.knownFields
        {'e': 'E', 'phi': 'CC'}

        Would represent that the `e` field and `phi` fields are known, and they are
        located on the mesh edges and cell centers, respectively.

        Returns
        -------
        dict
            They keys are the field names and the values are the field locations.
        """
        return self._knownFields

    @property
    def aliasFields(self):
        """The aliased fields of this object.

        The dictionary representing the aliased fields that can be accessed on this
        object. The keys are the names of the fields, and the values are a list of the
        known field, the aliased field's location on the mesh, and a function that goes
        from the known field to the aliased field.

        >>> fields.aliasFields
        {'b': ['e', 'F', '_e']}

        Would represent that the `e` field and `phi` fields are known, and they are
        located on the mesh edges and cell centers, respectively.

        Returns
        -------
        dict of {str: list}
            They keys are the field names and the values are list consiting of the
            field's alias, it's location on the mesh, and the function (or the name of
            it) to create it from the aliased field.
        """
        return self._aliasFields

    @property
    def dtype(self):
        """The data type of the storage matrix

        Returns
        -------
        dtype or dict of {str : dtype}
        """
        return self._dtype

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
            sz += np.array(self._storageShape(loc)).prod() * 8.0 / (1024**2)
        return "{0:e} MB".format(sz)

    def _storageShape(self, loc):
        n_fields = self.survey._n_fields

        nP = {
            "CC": self.mesh.nC,
            "N": self.mesh.nN,
            "F": self.mesh.nF,
            "E": self.mesh.nE,
        }[loc]

        return (nP, n_fields)

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
            ind = self.survey.get_source_indices(srcTestList)
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

    def _index_name_srclist_from_key(self, key, accessType):
        if not isinstance(key, tuple):
            key = (key,)
        if len(key) == 1:
            key += (None,)

        assert len(key) == 2, "must be [Src, fieldName]"

        srcTestList, name = key
        name = self._nameIndex(name, accessType)
        ind = self._srcIndex(srcTestList)
        if isinstance(srcTestList, slice):
            srcTestList = self.survey.source_list[srcTestList]
        return ind, name, srcTestList

    def __setitem__(self, key, value):
        ind, name, src_list = self._index_name_srclist_from_key(key, "set")
        if name is None:
            assert isinstance(
                value, dict
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
        ind, name, src_list = self._index_name_srclist_from_key(key, "get")
        if name is None:
            out = {}
            for name in self._fields:
                out[name] = self._getField(name, ind, src_list)
            return out
        return self._getField(name, ind, src_list)

    def _setField(self, field, val, name, ind):
        if isinstance(val, np.ndarray) and (
            field.shape[0] == field.size or val.ndim == 1
        ):
            val = mkvc(val, 2)
        field[:, ind] = val

    def _getField(self, name, ind, src_list):
        # ind will always be an list, thus the output will always
        # be (len(fields), n_inds)
        if name in self._fields:
            out = self._fields[name][:, ind]
        else:
            # Aliased fields
            alias, loc, func = self.aliasFields[name]

            if isinstance(func, str):
                assert hasattr(self, func), (
                    "The alias field function is a string, but it does not "
                    "exist in the Fields class."
                )
                func = getattr(self, func)
            if not isinstance(src_list, list):
                src_list = [src_list]
            out = func(self._fields[alias][:, ind], src_list)
        # if out.shape[0] == out.size or out.ndim == 1:
        #     out = mkvc(out, 2)
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

    @property
    def simulation(self):
        """The simulation object that created these fields

        Returns
        -------
        SimPEG.simulation.BaseTimeSimulation
        """
        return self._simulation

    @simulation.setter
    def simulation(self, value):
        self._simulation = validate_type(
            "simulation", value, BaseTimeSimulation, cast=False
        )

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

    def _index_name_srclist_from_key(self, key, accessType):
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
        if isinstance(srcTestList, slice):
            srcTestList = self.survey.source_list[srcTestList]
        return (srcInd, timeInd), name, srcTestList

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

    def _getField(self, name, ind, src_list):
        srcInd, timeInd = ind

        if name in self._fields:
            out = self._fields[name][:, srcInd, timeInd]
        else:
            # Aliased fields
            alias, loc, func = self.aliasFields[name]
            if isinstance(func, str):
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
            if not isinstance(src_list, list):
                src_list = [src_list]

            if timeII.size == 1:
                pointerShapeDeflated = self._correctShape(alias, ind, deflate=True)
                pointerFields = pointerFields.reshape(pointerShapeDeflated, order="F")
                out = func(pointerFields, src_list, timeII)
            else:  # loop over the time steps
                nT = pointerShape[2]
                out = list(range(nT))
                for i, TIND_i in enumerate(timeII):
                    fieldI = pointerFields[:, :, i]
                    if fieldI.shape[0] == fieldI.size:
                        fieldI = mkvc(fieldI, 2)
                    out[i] = func(fieldI, src_list, TIND_i)
                    if out[i].ndim == 1:
                        out[i] = out[i][:, np.newaxis, np.newaxis]
                    elif out[i].ndim == 2:
                        out[i] = out[i][:, :, np.newaxis]
                out = np.concatenate(out, axis=2)

        shape = self._correctShape(name, ind, deflate=True)
        return out.reshape(shape, order="F")
