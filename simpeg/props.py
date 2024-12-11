import functools
import warnings

import numpy as np

from simpeg.utils import deprecate_property
from . import maps
from .utils import Zero, validate_type, validate_ndarray_with_shape


class _Void:
    """A class used to mark no default value.

    Note that you can't use `None` as a marker, since `None` could be a default value itself.
    """

    pass


class PhysicalProperty(property):
    reciprocal = None

    def __init__(
        self,
        short_description,
        shape=None,
        default=_Void,
        dtype=None,
        invertible=True,
    ):
        self.default = default
        self.name = None
        self.cached_name = None
        self.invertible = invertible

        self.shape = shape
        self.dtype = dtype

        if shape is None:
            shape_str = ""
        else:
            shape_str = f"{shape} "
        if self.optional:
            shape_str = f"None or {shape_str}"
        dtype_str = f" of {dtype}"
        if dtype is None:
            dtype_str = ""

        doc = f"""{short_description}

        Returns
        -------
        {shape_str}numpy.ndarray{dtype_str}
        """

        super().__init__(fget=self.fget, fset=self.fset, fdel=self.fdel, doc=doc)

    @property
    def optional(self):
        return self.default is not _Void

    def set_name(self, name):
        self.name = name
        self.cached_name = f"_physical_property_{name}"

    def get_cls_attr_name(self, scope):
        return f"{type(scope).__name__}.{self.name}"

    def is_mapped(self, scope):
        is_self_mapped = isinstance(
            getattr(scope, self.cached_name, None), maps.IdentityMap
        )
        if not is_self_mapped and (recip := self.reciprocal):
            return isinstance(getattr(scope, recip.cached_name, None), maps.IdentityMap)
        return is_self_mapped

    def mapping(self, scope):
        stashed = getattr(scope, self.cached_name, None)
        if isinstance(stashed, maps.IdentityMap):
            return stashed
        if stashed is None:
            if recip := self.reciprocal:
                stashed = getattr(scope, recip.cached_name, None)
                if isinstance(stashed, maps.IdentityMap):
                    return maps.ReciprocalMap() * stashed
            return None

    def fget(self, scope):
        value = getattr(scope, self.cached_name, None)
        if value is None:
            if recip := self.reciprocal:
                try:
                    return 1.0 / getattr(scope, recip.name)
                except AttributeError:
                    if recip.is_mapped(scope):
                        raise AttributeError(
                            f"Reciprocal property '{recip.get_cls_attr_name(scope)}' was set as a map, "
                            f"but `{type(scope).__name__}.model` is not set"
                        )
                    else:
                        raise AttributeError(
                            f"'{type(scope).__name__}' has no attribute '{self.name}', "
                            f"nor its reciprocal '{recip.name}'"
                        )
            if self.optional:
                return self.default
            raise AttributeError(
                f"'{type(scope).__name__}' has no attribute '{self.name}'"
            )
        if isinstance(value, maps.IdentityMap):
            # if I was set as a mapping:
            if (model := scope.model) is None:
                raise AttributeError(
                    f"'{self.get_cls_attr_name(scope)}' was set as a map, but `{type(scope).__name__}.model` is not set"
                )
            return value @ model
        # otherwise I was good, so return the value
        return value

    def fset(self, scope, value):
        if value is not None:
            if isinstance(value, maps.IdentityMap):
                if not self.invertible:
                    raise ValueError(
                        f"Cannot assign a map to '{self.get_cls_attr_name(scope)}', "
                        f"because it is not an invertible property"
                    )
            else:
                value = validate_ndarray_with_shape(
                    self.name, value, shape=self.shape, dtype=self.dtype
                )
            if self.reciprocal:
                delattr(scope, self.reciprocal.name)
        setattr(scope, self.cached_name, value)

    def fdel(self, scope):
        if hasattr(scope, self.cached_name):
            delattr(scope, self.cached_name)

    def set_reciprocal(self, other):
        self.reciprocal = other
        other.reciprocal = self

    def deriv(self, scope, v=None):
        if not self.invertible:
            raise NotImplementedError(
                f"'{self.get_cls_attr_name(scope)}' has no derivative because it is not invertible"
            )
        if (mapping := self.mapping(scope)) is None:
            return Zero()
        return mapping.deriv(scope.model, v=v)

    def shallow_copy(self):
        new_prop = PhysicalProperty(
            "",
            shape=self.shape,
            default=self.default,
            dtype=self.dtype,
            invertible=self.invertible,
        )
        new_prop.__doc__ = self.__doc__
        return new_prop

    def setter(self, setter_func):
        new_prop = self.shallow_copy()
        new_prop.fset = setter_func
        return new_prop

    def deleter(self, deleter_func):
        new_prop = self.shallow_copy()
        new_prop.fdel = deleter_func
        return new_prop

    def set_invertible(self, invertible):
        new_prop = self.shallow_copy()
        new_prop.invertible = invertible
        return new_prop


class NestedModeler:
    def __init__(self, modeler_type, short_details=None):
        self.modeler_type = modeler_type
        self.short_details = short_details

    def get_property(scope):
        doc = f"""{scope.short_details}

        Returns
        -------
        {scope.modeler_type.__name__}
        """

        def fget(self):
            return getattr(self, f"_{scope.name}")

        def fset(self, value):
            if value is not None:
                value = validate_type(scope.name, value, scope.modeler_type, cast=False)
            setattr(self, f"_{scope.name}", value)

        def fdel(self):
            setattr(self, f"_{scope.name}", None)

        return property(fget=fget, fset=fset, fdel=fdel, doc=doc)


class BaseSimPEG:
    """"""


class PhysicalPropertyMetaclass(type):
    def __new__(mcs, name, bases, classdict):
        # remember the physical properties on class dicts.

        physical_properties = {}
        nested_modelers = {}
        for key, value in classdict.items():
            if isinstance(value, PhysicalProperty):
                value.set_name(key)
                physical_properties[key] = value
            elif isinstance(value, NestedModeler):
                nested_modelers[key] = value

        for key, value in nested_modelers.items():
            classdict[key] = value.get_property()

        newcls = super().__new__(mcs, name, bases, classdict)

        for parent in newcls.__mro__:
            physical_properties = (
                getattr(parent, "_physical_properties", {}) | physical_properties
            )
            nested_modelers = getattr(parent, "_nested_modelers", {}) | nested_modelers

        newcls._physical_properties = physical_properties
        newcls._nested_modelers = nested_modelers

        return newcls


class HasModel(BaseSimPEG, metaclass=PhysicalPropertyMetaclass):
    def __init__(self, model=None, **kwargs):
        self.model = model
        super().__init__(**kwargs)

    @property
    def _invertible_properties(self):
        """Returns a dictionary of string, property pairs of invertible properties."""
        return {
            name: prop
            for name, prop in self._physical_properties.items()
            if prop.invertible
        }

    @property
    def _mapped_properties(self):
        """Returns a dictionary of string, property pairs of mapped properties."""
        return {
            name: prop
            for name, prop in self._physical_properties.items()
            if prop.is_mapped(self)
        }

    @property
    def needs_model(self):
        """True if a model is necessary"""
        return len(self._mapped_properties) > 0

    def _prop_map(self, name):
        return self._physical_properties[name].mapping(self)

    def _prop_deriv(self, name, v=None):
        # TODO Add support for adjoints here and on mapping derivatives
        return self._physical_properties[name].deriv(self, v=v)

    @property
    def _has_nested_models(self):
        return len(self._nested_modelers) > 0

    # TODO: rename to _delete_on_model_update
    @property
    def _delete_on_model_update(self):
        """A list of properties stored on this object to delete when the model is updated

        Returns
        -------
        list of str
            For example `['_MeSigma', '_MeSigmaI']`.
        """
        return []

    deleteTheseOnModelUpdate = deprecate_property(
        _delete_on_model_update,
        "deleteTheseOnModelUpdate",
        removal_version="0.25.0",
        future_warn=True,
    )

    #: List of matrix names to have their factors cleared on a model update
    @property
    def clean_on_model_update(self):
        """A list of solver objects to clean when the model is updated

        Returns
        -------
        list of str
        """
        warnings.warn(
            "clean_on_model_update has been deprecated due to repeated functionality encompassed"
            " by the _delete_on_model_update method",
            FutureWarning,
            stacklevel=2,
        )
        return []

    @property
    def model(self):
        """The inversion model.

        Returns
        -------
        numpy.ndarray
        """
        return self._model

    @model.setter
    def model(self, value):
        if value is not None:
            # check if I need a model
            if not self.needs_model and not self._has_nested_models:
                raise AttributeError(
                    "Cannot set model if no properties have been set as maps. "
                    f"Choose from: {', '.join(self._invertible_properties.keys())}"
                )

            # coerce to a numpy array
            value = validate_ndarray_with_shape(
                "model", value, shape=("*",), dtype=float
            )

            # Check the model is a good shape
            errors = []
            for name, prop in self._mapped_properties.items():
                mapping = prop.mapping(self)
                correct_shape = mapping.shape[1] == "*" or mapping.shape[1] == len(
                    value
                )
                if not correct_shape:
                    errors.append(
                        "{}: expected model of len({}) for {}".format(
                            name, mapping.shape[1], str(mapping)
                        )
                    )
            if len(errors) > 0:
                raise ValueError(
                    "Model of len({}) incorrect shape for mappings: \n    {}".format(
                        len(value), "\n    ".join(errors)
                    )
                )
            previous = getattr(self, "_model", None)
            try:
                for modeler_name in self._nested_modelers:
                    modeler = getattr(self, modeler_name)
                    if modeler.needs_model:
                        modeler.model = value
            except Exception as err:
                # reset the nested_modelers and then throw the error
                for modeler_name in self._nested_modelers:
                    modeler = getattr(self, modeler_name)
                    if modeler.needs_model:
                        modeler.model = previous
                raise err

        # trigger model update function.
        previous_value = getattr(self, "_model", None)
        updated = False
        if previous_value is not value:
            if not (
                isinstance(previous_value, np.ndarray)
                and isinstance(value, np.ndarray)
                and np.allclose(previous_value, value)
            ):
                # cached items to delete
                for item in self._delete_on_model_update:
                    if hasattr(self, item):
                        delattr(self, item)

                updated = True

        self._model = value
        # Most of the time this return value is completely ignored
        # However if you need to know if the model was updated in
        # and child class, you can always access the method:
        # HasModel.model.fset
        return updated

    @model.deleter
    def model(self):
        self._model = (None,)
        # cached items to delete
        for item in self._delete_on_model_update:
            if hasattr(self, item):
                delattr(self, item)


def _add_deprecated_physical_property_functions(new_name, old_name=None):

    if old_name is None:
        old_name = new_name
    map_name = f"{old_name}Map"
    deriv_name = f"{old_name}Deriv"

    @property
    def prop_map(self):
        warnings.warn(
            f"Accessing {map_name} directly is no longer supported. If this is still necessary "
            f"use _prop_map('{new_name}') instead",
            UserWarning,
            stacklevel=2,
        )
        return self._prop_map(new_name)

    @prop_map.setter
    def prop_map(self, value):
        warnings.warn(
            f"Setting {map_name} directly is deprecated. Instead directly assign a mapping to {new_name}",
            UserWarning,
            stacklevel=2,
        )
        setattr(self, new_name, value)

    @property
    def prop_deriv(self):
        # derivatives are mostly used internally, and should not be publicly exposed to end users.
        warnings.warn(
            f"Accessing {deriv_name} is deprecated, use _prop_deriv('{new_name}') instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._prop_deriv(new_name)

    def decorator(cls):
        __init__ = cls.__init__

        @functools.wraps(__init__)
        def __new_init__(self, *args, **kwargs):
            old_map = kwargs.pop(map_name, None)
            __init__(self, *args, **kwargs)
            if old_map is not None:
                setattr(self, map_name, old_map)

        cls.__init__ = __new_init__
        setattr(cls, map_name, prop_map)
        setattr(cls, deriv_name, prop_deriv)

        return cls

    return decorator
