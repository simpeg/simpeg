from SimPEG import Utils, Maps

class Property(object):

    name           = ''
    doc            = ''
    # mappingPair    = None

    defaultVal     = None
    defaultMap     = None
    defaultInvProp = False

    def __init__(self, doc, **kwargs):
        # Set the default after all other params are set
        self.doc = doc
        Utils.setKwargs(self, **kwargs)

    def _getMapProperty(self):
        prop = self
        def fget(self):
            return getattr(self, '_%sMap'%prop.name, prop.defaultMap)
        def fset(self, val):
            # TODO: Check if the mapping can be correct
            setattr(self, '_%sMap'%prop.name, val)
        return property(fget=fget, fset=fset, doc=prop.doc)

    def _getIndexProperty(self):
        prop = self
        def fget(self):
            return getattr(self, '_%sIndex'%prop.name, slice(None))
        def fset(self, val):
            setattr(self, '_%sIndex'%prop.name, val)
        return property(fget=fget, fset=fset, doc=prop.doc)

    def _getProperty(self):
        prop = self
        def fget(self):
            mapping = getattr(self, '%sMap'%prop.name)
            if mapping is None:
                return prop.defaultVal
            m       = getattr(self, '%sModel'%prop.name)
            return mapping * m
        return property(fget=fget)

    def _getModelDerivProperty(self):
        prop = self
        def fget(self):
            m       = getattr(self, '%sModel'%prop.name)
            mapping = getattr(self, '%sMap'%prop.name)
            if mapping is None:
                return None
            return mapping.deriv( m )
        return property(fget=fget)

    def _getModelProperty(self):
        prop = self
        def fget(self):
            mapping = getattr(self, '%sMap'%prop.name)
            if mapping is None:
                return None
            index = getattr(self.propMap, '_%sIndex'%prop.name, slice(None))
            return self.vector[index]
        return property(fget=fget)

    def _getModelMapProperty(self):
        prop = self
        def fget(self):
            return getattr(self.propMap, '_%sMap'%prop.name, prop.defaultMap)
        return property(fget=fget)



class PropModel(object):
    def __init__(self, propMap, vector):
        self.propMap = propMap
        self.vector  = vector

    # TODO: nP


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

        # You are only allowed one default inversion property.
        defaultInvProps = []
        for p in _properties:
            if _properties[p].defaultInvProp:
                defaultInvProps += [p]
        if len(defaultInvProps) > 1:
            raise Exception('You have more than one default inversion property: %s' % defaultInvProps)

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
            attrs[attr + 'Model'] = prop._getModelProperty()
            attrs[attr + 'Deriv'] = prop._getModelDerivProperty()

        return type(name.replace('PropMap', 'PropModel'), (PropModel, ), attrs)





class PropMap(object):
    __metaclass__ = _PropMapMetaClass

    def __init__(self, mappings):
        """
            PropMap takes a multi parameter model and maps it to the equivalent PropModel
        """
        if type(mappings) is dict:
            assert np.all([k in ['maps', 'slices'] for k in mappings]), 'Dict must only have properties "maps" and "slices"'
            self.setup(mappings['maps'], slices=mappings['slices'])
        if type(mappings) is list:
            self.setup(mappings)
        elif isinstance(mappings, Maps.IdentityMap):
            self.setup([(self.defaultInvProp, mappings)])
        else:
            raise Exception('mappings must be a dict, a mapping, or a list of tuples.')


    def setup(self, maps, slices=None):
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
                isinstance(m[1], Maps.IdentityMap)
                for m in maps]), "Use signature: [%s]" % (', '.join(["('%s', %sMap)"%(p,p) for p in self._properties]))
        if slices is None:
            slices = dict()
        else:
            assert np.all([
                s in self._properties and
                (type(s) in [slice, list] or isinstance(s, np.ndarray))
                for s in slices]), 'Slices must be for each property'

        self.clearMaps()

        nP = 0
        for name, mapping in maps:
            setattr(self, '%sMap'%name, mapping)
            setattr(self, '%sIndex'%name, slices.get(name, slice(nP, nP + mapping.nP)))
            nP += mapping.nP


    @property
    def defaultInvProp(self):
        for name in self._properties:
            p = self._properties[name]
            if p.defaultInvProp:
                return p.name

    def clearMaps(self):
        for name in self._properties:
            setattr(self, '%sMap'%name, None)
            setattr(self, '%sIndex'%name, None)

    def __call__(self, vec):
        return self.PropModel(self, vec)


class MyPropMap(PropMap):
    sigma = Property("Electrical Conductivity", defaultInvProp=True)
    mu    = Property("Electrical Conductivity", defaultVal=4e-10)


if __name__ == '__main__':


    from SimPEG import Mesh
    import numpy as np

    expMap = Maps.ExpMap(Mesh.TensorMesh((3,)))
    print expMap.nP

    # propMap = MyPropMap([('sigma', expMap), ('mu', IMap)], indices={'sigma':[1,2,3,4,7,8]})
    propMap = MyPropMap([('sigma',expMap)])
    print [n for n in dir(propMap) if n[0] is not '_']
    # propMap = My2PropMap()
    # print [n for n in dir(propMap) if n[0] is not '_']

    propMap.sigmaMap = expMap

    print propMap.defaultInvProp
    print propMap.sigmaMap
    print propMap.sigmaIndex

    mod = propMap(np.r_[1,2,3])
    print [n for n in dir(mod) if n[0] is not '_']
    print mod.sigmaModel
    print mod.sigma
    print mod.sigmaMap
    print mod.sigmaDeriv

    print mod.mu
    print mod.muMap
    print mod.muModel
    print mod.muDeriv
