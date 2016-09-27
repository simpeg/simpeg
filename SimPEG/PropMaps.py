from __future__ import print_function
from . import Utils
import numpy as np
import scipy.sparse as sp
from six import add_metaclass


class Property(object):

    name = ''
    doc = ''

    defaultVal = None
    defaultInvProp = False

    def __init__(self, doc, **kwargs):
        # Set the default after all other params are set
        self.doc = doc
        Utils.setKwargs(self, **kwargs)

    @property
    def propertyLink(self):
        "Can be something like: ('sigma', Maps.ReciprocalMap)"
        return getattr(self, '_propertyLink', None)

    @propertyLink.setter
    def propertyLink(self, value):
        from .Maps import IdentityMap
        assert type(value) is tuple and len(value) == 2 and type(value[0]) is str and issubclass(value[1], IdentityMap), 'Use format: ("{0!s}", Maps.ReciprocalMap)'.format(self.name)
        self._propertyLink = value

    def _getMapProperty(self):
        prop = self

        def fget(self):
            return getattr(self, '_{0!s}Map'.format(prop.name), None)

        def fset(self, val):
            if prop.propertyLink is not None:
                linkName, linkMap = prop.propertyLink
                assert getattr(self, '{0!s}Map'.format(linkName), None) is None, 'Cannot set both sides of a linked property.'
            # TODO: Check if the mapping can be correct
            setattr(self, '_{0!s}Map'.format(prop.name), val)
        return property(fget=fget, fset=fset, doc=prop.doc)

    def _getIndexProperty(self):
        prop = self

        def fget(self):
            return getattr(self, '_{0!s}Index'.format(prop.name), slice(None))

        def fset(self, val):
            setattr(self, '_{0!s}Index'.format(prop.name), val)
        return property(fget=fget, fset=fset, doc=prop.doc)

    def _getProperty(self):
        prop = self

        def fget(self):
            mapping = getattr(self, '{0!s}Map'.format(prop.name))
            if mapping is None and prop.propertyLink is None:
                return prop.defaultVal

            if mapping is None and prop.propertyLink is not None:
                linkName, linkMapClass = prop.propertyLink
                linkMap = linkMapClass(None)
                if getattr(self, '{0!s}Map'.format(linkName), None) is None:
                    return prop.defaultVal
                m = getattr(self, '{0!s}'.format(linkName))
                return linkMap * m

            m = getattr(self, '{0!s}Model'.format(prop.name))
            return mapping * m
        return property(fget=fget)

    def _getModelDerivProperty(self):
        prop = self

        def fget(self):
            mapping = getattr(self, '{0!s}Map'.format(prop.name))
            if mapping is None and prop.propertyLink is None:
                return None

            if mapping is None and prop.propertyLink is not None:
                linkName, linkMapClass = prop.propertyLink
                linkedMap = getattr(self, '{0!s}Map'.format(linkName))
                if linkedMap is None:
                    return None
                linkMap = linkMapClass(None) * linkedMap
                m = getattr(self, '{0!s}Model'.format(linkName))
                return linkMap.deriv(m)

            m = getattr(self, '{0!s}Model'.format(prop.name))
            return mapping.deriv(m)
        return property(fget=fget)

    def _getModelProperty(self):
        prop = self
        def fget(self):
            mapping = getattr(self, '{0!s}Map'.format(prop.name))
            if mapping is None:
                return None
            index = getattr(self.propMap, '{0!s}Index'.format(prop.name))
            return self.vector[index]
        return property(fget=fget)

    def _getModelProjProperty(self):
        prop = self

        def fget(self):
            mapping = getattr(self, '{0!s}Map'.format(prop.name))
            if mapping is None:
                return None
            inds = getattr(self.propMap, '{0!s}Index'.format(prop.name))
            if type(inds) is slice:
                inds = list(range(*inds.indices(self.nP)))
            nI, nP = len(inds),self.nP
            return sp.csr_matrix((np.ones(nI), (range(nI), inds) ), shape=(nI, nP))
        return property(fget=fget)

    def _getModelMapProperty(self):
        prop = self

        def fget(self):
            return getattr(self.propMap, '_{0!s}Map'.format(prop.name), None)
        return property(fget=fget)


class PropModel(object):
    def __init__(self, propMap, vector):
        self.propMap = propMap
        self.vector  = vector
        assert len(self.vector) == self.nP

    @property
    def nP(self):
        inds = []
        if getattr(self, '_nP', None) is None:
            for name in self.propMap._properties:
                index = getattr(self.propMap, '{0!s}Index'.format(name), None)
                if index is not None:
                    if type(index) is slice:
                        inds += list(range(*index.indices(len(self.vector))))
                    else:
                        inds += list(index)
            self._nP = len(set(inds))
        return self._nP

    def __contains__(self, val):
        return val in self.propMap


_PROPMAPCLASSREGISTRY = {}


class _PropMapMetaClass(type):
    def __new__(cls, name, bases, attrs):
        assert name.endswith('PropMap'), 'Please use convention: ___PropMap, e.g. ElectromagneticPropMap'
        _properties = {}
        for base in bases:
            for baseProp in getattr(base, '_properties', {}):
                _properties[baseProp] = base._properties[baseProp]
        keys = [key for key in attrs]
        for attr in keys:
            if isinstance(attrs[attr], Property):
                attrs[attr].name = attr
                attrs[attr + 'Map'  ] = attrs[attr]._getMapProperty()
                attrs[attr + 'Index'] = attrs[attr]._getIndexProperty()
                _properties[attr] = attrs[attr]
                attrs.pop(attr)

        attrs['_properties'] = _properties

        defaultInvProps = []
        for p in _properties:
            prop = _properties[p]
            if prop.defaultInvProp:
                defaultInvProps += [p]
            if prop.propertyLink is not None:
                assert prop.propertyLink[0] in _properties, "You can only link to things that exist: '{0!s}' is trying to link to '{1!s}'".format(prop.name, prop.propertyLink[0])
        if len(defaultInvProps) > 1:
            raise Exception('You have more than one default inversion property: {0!s}'.format(defaultInvProps))

        newClass = super(_PropMapMetaClass, cls).__new__(cls, name, bases, attrs)

        newClass.PropModel = cls.createPropModelClass(newClass, name, _properties)

        _PROPMAPCLASSREGISTRY[name] = newClass
        return newClass

    def createPropModelClass(self, name, _properties):

        attrs = dict()

        for attr in _properties:
            prop = _properties[attr]

            attrs[attr          ] = prop._getProperty()
            attrs[attr + 'Map'  ] = prop._getModelMapProperty()
            attrs[attr + 'Proj' ] = prop._getModelProjProperty()
            attrs[attr + 'Model'] = prop._getModelProperty()
            attrs[attr + 'Deriv'] = prop._getModelDerivProperty()

        return type('PropModel', (PropModel, ), attrs)


@add_metaclass(_PropMapMetaClass)
class PropMap(object):
    #__metaclass__ = _PropMapMetaClass

    def __init__(self, mappings):
        from .Maps import IdentityMap
        """
            PropMap takes a multi parameter model and maps it to the equivalent PropModel
        """
        if type(mappings) is dict:
            assert np.all([k in ['maps', 'slices'] for k in mappings]), 'Dict must only have properties "maps" and "slices"'
            self.setup(mappings['maps'], slices=mappings['slices'])
        elif type(mappings) is list:
            self.setup(mappings)
        elif isinstance(mappings, IdentityMap):
            self.setup([(self.defaultInvProp, mappings)])
        else:
            raise Exception('mappings must be a dict, a mapping, or a list of tuples.')


    def setup(self, maps, slices=None):
        from .Maps import IdentityMap
        """
            Sets up the maps and slices for the PropertyMap


            :param list maps: [('sigma', sigmaMap), ('mu', muMap), ...]
            :param list slices: [('sigma', slice(0,nP)), ('mu', [1,2,5,6]), ...]

        """
        assert np.all([
                type(m) is tuple and
                len(m)==2 and
                type(m[0]) is str and
                m[0] in self._properties and
                isinstance(m[1], IdentityMap)
                for m in maps]), "Use signature: [{0!s}]".format((', '.join(["('{0!s}', {1!s}Map)".format(p, p) for p in self._properties])))
        if slices is None:
            slices = dict()
        else:
            assert np.all([
                s in self._properties and
                (type(slices[s]) in [slice, list] or isinstance(slices[s], np.ndarray))
                for s in slices]), 'Slices must be for each property'

        self.clearMaps()

        nP = 0
        for name, mapping in maps:
            setattr(self, '{0!s}Map'.format(name), mapping)
            setattr(self, '{0!s}Index'.format(name), slices.get(name, slice(nP, nP + mapping.nP)))
            nP += mapping.nP
        self.nP = nP

    @property
    def defaultInvProp(self):
        for name in self._properties:
            p = self._properties[name]
            if p.defaultInvProp:
                return p.name

    def clearMaps(self):
        for name in self._properties:
            setattr(self, '{0!s}Map'.format(name), None)
            setattr(self, '{0!s}Index'.format(name), None)

    def __call__(self, vec):
        return self.PropModel(self, vec)

    def __contains__(self, val):
        activeMaps = [name for name in self._properties if getattr(self, '{0!s}Map'.format(name)) is not None]
        return val in activeMaps
