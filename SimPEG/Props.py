from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import properties
import numpy as np

from . import Maps
from . import Utils


class Model(properties.Array):

    info_text = 'a numpy array'


class Mapping(properties.Property):

    info_text = 'a SimPEG Map'

    @property
    def prop(self):
        return getattr(self, '_prop', None)

    @prop.setter
    def prop(self, value):
        assert isinstance(value, PhysicalProperty)
        value._mapping = self  # Skip the setter
        self._prop = value

    @property
    def reciprocal(self):
        if self.prop and self.prop.reciprocal:
            return self.prop.reciprocal.mapping

    @property
    def reciprocal_prop(self):
        if self.prop and self.prop.reciprocal:
            return self.prop.reciprocal

    def clear_props(self, instance):
        if self.prop:
            instance._set(self.prop.name, None)
        if self.reciprocal_prop:
            instance._set(self.reciprocal_prop.name, None)
        if self.reciprocal:
            instance._set(self.reciprocal.name, None)

    def validate(self, instance, value):
        if value is None:
            return None
        if not isinstance(value, Maps.IdentityMap):
            self.error(instance, value)
        return value

    def get_property(self):

        scope = self

        def fget(self):
            value = self._get(scope.name)
            if value is not None:
                return value
            if scope.reciprocal is None:
                return None
            reciprocal = self._get(scope.reciprocal.name)
            if reciprocal is None:
                return None
            return Maps.ReciprocalMap() * reciprocal

        def fset(self, value):
            value = scope.validate(self, value)
            self._set(scope.name, value)
            scope.clear_props(self)

        return property(fget=fget, fset=fset, doc=scope.doc)

    def as_pickle(self, instance):
        return instance._get(self.name)


class PhysicalProperty(properties.Property):

    info_text = 'a physical property'

    @property
    def mapping(self):
        return getattr(self, '_mapping', None)

    @mapping.setter
    def mapping(self, value):
        assert isinstance(value, Mapping)
        value._prop = self  # Skip the setter
        self._mapping = value

    reciprocal = None

    def clear_mappings(self, instance):
        if self.mapping:
            instance._set(self.mapping.name, None)
        if not self.reciprocal:
            return
        instance._set(self.reciprocal.name, None)
        if self.reciprocal.mapping:
            instance._set(self.reciprocal.mapping.name, None)

    def validate(self, instance, value):
        if value is None:
            return None
        assert isinstance(value, (np.ndarray, float)), (
            "Physical properties must be numpy arrays or floats."
        )
        return value

    def get_property(self):

        scope = self

        def fget(self):
            default = self._get(scope.name)
            if default is not None:
                return default
            if scope.reciprocal:
                default = self._get(scope.reciprocal.name)
                if default is not None:
                    return 1.0 / default
            if scope.mapping is None and scope.reciprocal is None:
                return None
            if scope.mapping is None:
                # look to the reciprocal
                if scope.reciprocal.mapping is None:
                    # there is no reciprocal mapping
                    reciprocal_val = self._get(scope.reciprocal.name)
                    if reciprocal_val is None:
                        raise AttributeError(
                            'A default for {}/{} has not been set'.format(
                                scope.name, scope.reciprocal.name
                            )
                        )
                return 1.0 / getattr(self, scope.reciprocal.name)

            mapping = getattr(self, scope.mapping.name)
            if mapping is None:
                raise AttributeError(
                    'A default `{}` or mapping `{}` has not been set.'.format(
                        scope.name,
                        scope.mapping.name
                    )
                )
            if self.model is None:
                raise AttributeError(
                    'A `model` is required for physical property {}'.format(
                        scope.name
                    )
                )
            return mapping * self.model

        def fset(self, value):
            value = scope.validate(self, value)
            self._set(scope.name, value)
            scope.clear_mappings(self)

        return property(fget=fget, fset=fset, doc=scope.doc)

    def as_pickle(self, instance):
        return instance._get(self.name)


class Derivative(properties.GettableProperty):

    physical_property = None

    @property
    def mapping(self):
        """The mapping looks through to the physical property map."""
        if self.physical_property is None:
            return None
        return self.physical_property.mapping

    def get_property(self):

        scope = self

        def fget(self):
            if scope.physical_property is None:
                return Utils.Zero()
            if scope.mapping is None:
                return Utils.Zero()
            mapping = getattr(self, scope.mapping.name)
            if mapping is None:
                return Utils.Zero()

            return mapping.deriv(self.model)

        return property(fget=fget, doc=scope.doc)


def Invertible(help, default=None):

    mapping = Mapping(
        "Mapping of {} to the inversion model.".format(help)
    )

    physical_property = PhysicalProperty(
        help,
        mapping=mapping,
        default=default
    )

    property_derivative = Derivative(
        "Derivative of {} wrt the model.".format(help),
        physical_property=physical_property
    )

    return physical_property, mapping, property_derivative


def Reciprocal(prop1, prop2):
    prop1.reciprocal = prop2
    prop2.reciprocal = prop1


class BaseSimPEG(properties.HasProperties):

    _exclusive_kwargs = False
