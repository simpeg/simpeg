from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import properties
import numpy as np
import warnings

from . import Maps
from . import Utils


class SphinxProp(object):
    """
    Update the auto-documenter from properties
    https://github.com/3ptscience/properties/issues/153
    """
    def sphinx_class(self):
        return ':class:`{cls} <{ref}>`'.format(
            cls=self.__class__.__name__,
            ref='SimPEG.Props.{}'.format(self.__class__.__name__)
        )


class Array(SphinxProp, properties.Array):

    class_info = 'a numpy, Zero or Identity array'

    def validate(self, instance, value):
        if isinstance(value, (Utils.Zero, Utils.Identity)):
            return value
        return super(Array, self).validate(instance, value)


class Float(SphinxProp, properties.Float):

    class_info = 'a float, Zero or Identity'

    def validate(self, instance, value):
        if isinstance(value, (Utils.Zero, Utils.Identity)):
            return value
        return super(Float, self).validate(instance, value)


class Integer(SphinxProp, properties.Integer):

    class_info = 'an Integer or *'

    def validate(self, instance, value):
        if isinstance(value, str):
            assert value == '*', 'value must be an integer or *, not {}'.format(
                value
            )
            return value
        return super(Integer, self).validate(instance, value)


class Model(SphinxProp, properties.Array):

    class_info = 'a numpy array'
    _required = False


class Mapping(SphinxProp, properties.Property):

    class_info = 'a SimPEG Map'
    _required = False

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
        for prop in (self.prop, self.reciprocal_prop, self.reciprocal):
            if prop is not None:
                if prop.name in instance._props:
                    delattr(instance, prop.name)
                else:
                    setattr(instance, prop.name, None)

    def validate(self, instance, value):
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
            if value is not properties.utils.undefined:
                value = scope.validate(self, value)
            self._set(scope.name, value)
            if value is not properties.utils.undefined:
                scope.clear_props(self)

        def fdel(self):
            self._set(scope.name, properties.utils.undefined)

        return property(fget=fget, fset=fset, fdel=fdel, doc=scope.doc)

    def as_pickle(self, instance):
        return instance._get(self.name)


class PhysicalProperty(SphinxProp, properties.Property):

    class_info = 'a physical property'
    reciprocal = None
    _required = False

    @property
    def mapping(self):
        return getattr(self, '_mapping', None)

    @mapping.setter
    def mapping(self, value):
        assert isinstance(value, Mapping)
        value._prop = self  # Skip the setter
        self._mapping = value

    def clear_mappings(self, instance):
        if self.mapping is not None:
            if self.mapping.name in instance._props:
                delattr(instance, self.mapping.name)
            else:
                setattr(instance, self.mapping.name, None)
        if self.reciprocal is not None:
            if self.reciprocal.mapping is not None:
                if self.reciprocal.mapping.name in instance._props:
                    delattr(instance, self.reciprocal.mapping.name)
                else:
                    setattr(instance, self.reciprocal.mapping.name, None)

    def validate(self, instance, value):
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
                    # set by default reciprocal
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
                # Set by mapped reciprocal
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
            if value is not properties.utils.undefined:
                value = scope.validate(self, value)
                if scope.reciprocal:
                    delattr(self, scope.reciprocal.name)
            self._set(scope.name, value)
            if value is not properties.utils.undefined:
                scope.clear_mappings(self)

        def fdel(self):
            self._set(scope.name, properties.utils.undefined)

        return property(fget=fget, fset=fset, fdel=fdel, doc=scope.doc)

    def as_pickle(self, instance):
        return instance._get(self.name)

    def summary(self, instance):
        default = instance._get(self.name)
        if default is not None:
            return '[*] {}: set by default value'.format(
                self.name
            )
        if self.reciprocal:
            default = instance._get(self.reciprocal.name)
            if default is not None:
                return '[*] {}: set by default reciprocal: 1.0 / {}'.format(
                    self.name,
                    self.reciprocal.name
                )
        if self.mapping is None and self.reciprocal is None:
            return '[ ] {}: property not set'.format(
                self.name,
                self.reciprocal.name
            )
        if self.mapping is None:
            if self.reciprocal.mapping is None:
                # there is no reciprocal mapping
                reciprocal_val = instance._get(self.reciprocal.name)
                if reciprocal_val is None:
                    return '[ ] {}: default for {}/{} not set'.format(
                        self.name, self.name, self.reciprocal.name
                    )
            return '[*] {}: set by mapped reciprocal 1.0 / ({} * {})'.format(
                self.name, self.reciprocal.mapping.name, self.reciprocal.name
            )

        mapping = getattr(instance, self.mapping.name)

        if mapping is None:
            return '[ ] {}: default `{}` or mapping `{}` not set'.format(
                self.name, self.name, self.mapping.name
            )
        if instance.model is None:
            return (
                '[ ] {}: model({}) required, from active `{}` mapping: {}'
            ).format(
                self.name,
                mapping.shape[1],
                self.mapping.name,
                str(mapping)
            )

        correct_shape = (
            mapping.shape[1] == '*' or
            mapping.shape[1] == len(instance.model)
        )

        if correct_shape:
            return '[*] {}: set by the `{}` mapping: {} * model({})'.format(
                self.name, self.mapping.name, str(mapping), len(instance.model)
            )

        return '[ ] {}: incorrect mapping/model shape: {} * model({})'.format(
            self.name, str(mapping), len(instance.model)
        )


class Derivative(SphinxProp, properties.GettableProperty):

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
            if self.model is None:
                return Utils.Zero()

            return mapping.deriv(self.model)

        return property(fget=fget, doc=scope.doc)


def Invertible(help, default=None):

    mapping = Mapping(
        "Mapping of {} to the inversion model.".format(help)
    )

    physical_property = PhysicalProperty(
        help,
        mapping=mapping
    )
    if default is not None:
        physical_property.default = default

    property_derivative = Derivative(
        "Derivative of {} wrt the model.".format(help),
        physical_property=physical_property
    )

    return physical_property, mapping, property_derivative


def Reciprocal(prop1, prop2):
    prop1.reciprocal = prop2
    prop2.reciprocal = prop1


class BaseSimPEG(properties.HasProperties):
    """"""


class HasModel(BaseSimPEG):

    model = Model("Inversion model.")

    @property
    def _all_map_names(self):
        """Returns all Mapping properties"""
        return sorted([
            k for k in self._props
            if isinstance(self._props[k], Mapping)
        ])

    @property
    def _act_map_names(self):
        """Returns all active Mapping properties"""
        return sorted([
            k for k in self._all_map_names
            if getattr(self, k) is not None
        ])

    @property
    def needs_model(self):
        """True if a model is necessary"""
        return len(self._act_map_names) > 0

    @property
    def _has_nested_models(self):
        for k in self._props:
            if (
                    isinstance(self._props[k], properties.Instance) and
                    issubclass(self._props[k].instance_class, HasModel)
               ):
                return True
        return False

    @properties.validator('model')
    def _check_model_valid(self, change):
        """Checks the model length and necessity"""
        if change['value'] is properties.utils.undefined:
            return True

        if not self.needs_model and not self._has_nested_models:
            warnings.warn(
                "Cannot add model as there are no active mappings"
                ", choose from: ['{}']".format(
                    "', '".join(self._all_map_names)
                )
            )
            return

        errors = []

        for name in self._act_map_names:
            mapping = getattr(self, name)
            correct_shape = (
                mapping.shape[1] == '*' or
                mapping.shape[1] == len(change['value'])
            )
            if not correct_shape:
                errors += [
                    '{}: expected model of len({}) for {}'.format(
                        name,
                        mapping.shape[1],
                        str(mapping)
                    )
                ]
        if len(errors) == 0:
            return True

        warnings.warn(
            'Model of len({}) incorrect shape for mappings: \n    {}'.format(
                len(change['value']),
                '\n    '.join(errors)
            )
        )

    @properties.validator
    def _check_valid(self):
        errors = []

        # Check if the model is necessary
        if self.needs_model and self.model is None:
            errors += ['model must not be None']
        if not self.needs_model and self.model is not None:
            errors += ['there are no active maps, but a model is provided']

        # Check each map is the same size
        shapes = []
        for name in self._act_map_names:
            shape = getattr(self, name).shape[1]
            if shape == '*':
                continue
            shapes += [shape]
        if not all(x == shapes[0] for x in shapes):
            errors += ['the mappings are not the same shape']

        # Check that the model is the same size as the mappings
        if len(shapes) > 0 and self.model is not None:
            if not len(self.model) == shapes[0]:
                errors += ['the model must be len({})'.format(shapes[0])]

        # Check each physical property
        check_boxes = sorted([
            self._props[k].summary(self) for k in self._props
            if isinstance(self._props[k], PhysicalProperty)
        ])
        for line in check_boxes:
            if line[:3] == '[ ]':
                errors += [line[4:]]

        if len(errors) == 0:
            return True

        raise ValueError(
            'The {} instance has the following errors: \n - {}'.format(
                self.__class__.__name__,
                '\n - '.join(errors)
            )
        )

    def summary(self):
        prop_names = sorted([
            k for k in self._props
            if isinstance(self._props[k], PhysicalProperty)
        ])

        out = ['Physical Properties:']

        # Grab the physical property summaries
        for prop in prop_names:
            out += [' ' + self._props[prop].summary(self)]

        # Grab the validation errors
        try:
            self.validate()
            out += ['', 'All checks pass!']
        except ValueError as e:
            out += ['', str(e)]

        return '\n'.join(out)
