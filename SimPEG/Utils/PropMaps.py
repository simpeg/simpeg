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

    def __init__(self, ):
        """
            PropMap takes a multi parameter model and maps it to the equivalent PropModel
        """
        pass

    def __call__(self, vec):
        return self.PropModel(self, vec)


class MyPropMap(PropMap):
    sigma = Property("Electrical Conductivity", defaultInvProp=True)
    # rho   = InveseProperty(sigma)

class My2PropMap(MyPropMap):
    mu    = Property("Electrical Conductivity", defaultVal=4e-10)


if __name__ == '__main__':


    from SimPEG import Mesh
    import numpy as np

    expMap = Maps.ExpMap(Mesh.TensorMesh((3,)))
    print expMap.nP

    # propMap = MyPropMap([('sigma', expMap), ('mu', IMap)], indices={'sigma':[1,2,3,4,7,8]})
    propMap = MyPropMap()
    print [n for n in dir(propMap) if n[0] is not '_']
    propMap = My2PropMap()
    print [n for n in dir(propMap) if n[0] is not '_']

    propMap.sigmaMap = expMap

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
