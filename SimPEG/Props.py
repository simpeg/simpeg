from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import properties
import numpy as np

from . import Maps
from . import Utils


class Mapping(properties.Property):

    info_text = 'a SimPEG Map'

    def validate(self, instance, value):
        if value is None:
            return None
        if not isinstance(value, Maps.IdentityMap):
            self.error(instance, value)
        return value


class PhysicalProperty(properties.Property):

    info_text = 'a physical property'

    mapping = None

    def validate(self, instance, value):
        if value is None:
            return None
        assert isinstance(value, np.ndarray), (
            "Physical properties must be numpy arrays."
        )
        return value

    def get_property(self):

        scope = self

        def fget(self):
            default = self._get(scope.name)
            if default is not None:
                return default
            mapping = getattr(self, scope.mapping.name)
            return mapping * self.model

        def fset(self, value):
            # clear the mapping
            setattr(self, scope.mapping.name, None)
            value = scope.validate(self, value)
            self._set(scope.name, value)

        return property(fget=fget, fset=fset, doc=scope.help)


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

        return property(fget=fget, doc=scope.help)


def Invertible(help):

    mapping = Mapping(
        "Mapping of {} to the inversion model.".format(help)
    )

    physical_property = PhysicalProperty(
        help,
        mapping=mapping
    )

    property_derivative = Derivative(
        "Derivative of {} wrt the model.".format(help),
        physical_property=physical_property
    )

    return physical_property, mapping, property_derivative


class BaseSimPEG(properties.HasProperties()):
    pass
