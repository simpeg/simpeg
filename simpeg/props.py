import functools
import warnings
from typing import Union, Optional

import numpy as np
import numpy.typing as npt

from simpeg.utils import deprecate_property
from . import maps
from .utils import Zero, validate_type, validate_ndarray_with_shape


class _Void:
    """A class used to mark no default value.

    Note that you can't use `None` as a marker, since `None` could be a default value itself.
    """

    pass


class PhysicalProperty:
    """
    Physical properties as implemented as descriptors.

    Parameters
    ----------
    short_description
    shape : tuple of int or '*'
        The shape the expected property array should have.
    default, optional
        The default value the parameter should take if it was never assigned.
    dtype : np.dtype
    invertible : bool
    reciprocal, optional
    """

    def __init__(
        self,
        short_description,
        shape=None,
        default=_Void,
        dtype=None,
        invertible=True,
        reciprocal=None,
    ):
        self.default = default
        self.cached_name = None
        self.invertible = invertible
        self.reciprocal = None
        self.set_reciprocal(reciprocal)

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

        Parameters
        ----------
        prop : {shape_str}array_like{dtype_str} or maps.IdentityMap
            If set as a mapping, the physical property will be calculated from the mapping and the `model`
            when the physical property is retrieved. Setting a physical property signals to the simulation
            that you intend to invert for this parameter.

        Returns
        -------
        {shape_str}numpy.ndarray{dtype_str}
        """
        self.__doc__ = doc

    @property
    def optional(self):
        return self.default is not _Void

    def __set_name__(self, owner, name):
        self.__name__ = name
        self.cached_name = f"_physical_property_{name}"

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        if self.fget is None:
            raise AttributeError
        return self.fget(obj)

    def __set__(self, obj, value):
        if self.fset is None:
            raise AttributeError
        self.fset(obj, value)

    def __delete__(self, obj):
        if self.fdel is None:
            raise AttributeError
        self.fdel(obj)

    def get_cls_attr_name(self, scope):
        return f"{type(scope).__name__}.{self.__name__}"

    def is_mapped(self, scope):
        im_mapped = self.__name__ in scope._mapped_properties
        if not im_mapped and (recip := self.reciprocal):
            return recip.__name__ in scope._mapped_properties
        return im_mapped

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

    def fget(self, scope) -> npt.NDArray:
        value = getattr(scope, self.cached_name, None)
        if isinstance(value, maps.IdentityMap):
            # if I was set as a mapping:
            if (model := scope.model) is None:
                raise AttributeError(
                    f"'{self.get_cls_attr_name(scope)}' was set as a map, but `{type(scope).__name__}.model` is not set"
                )
            return value @ model
        elif value is not None:
            return value
        # Means value was None
        # Then check my reciprocal for a return value
        if recip := self.reciprocal:
            recip_value = getattr(scope, recip.cached_name, None)
            if isinstance(recip_value, maps.IdentityMap):
                if (model := scope.model) is None:
                    raise AttributeError(
                        f"Reciprocal property '{recip.get_cls_attr_name(scope)}' was set as a map, "
                        f"but `{type(scope).__name__}.model` is not set"
                    )
                recip_value = recip_value @ scope.model
            elif recip_value is None:
                if self.optional:
                    return self.default
                if not recip.optional:
                    raise AttributeError(
                        f"'{type(scope).__name__}' has no attribute '{self.__name__}', "
                        f"nor its reciprocal '{recip.__name__}'"
                    )
                else:
                    recip_value = recip.default
            if recip_value is not None:
                return 1.0 / recip_value
            return recip_value

        if self.optional:
            return self.default

        raise AttributeError(
            f"'{type(scope).__name__}' has no attribute '{self.__name__}'"
        )

    def fset(self, scope, value: Optional[Union[npt.NDArray, maps.IdentityMap]]):
        is_map = isinstance(value, maps.IdentityMap)
        if is_map:
            if not self.invertible:
                raise ValueError(
                    f"Cannot assign a map to '{self.get_cls_attr_name(scope)}', "
                    f"because it is not an invertible property"
                )
        if value is not None:
            if not is_map:
                value = validate_ndarray_with_shape(
                    self.__name__, value, shape=self.shape, dtype=self.dtype
                )
                if value.ndim == 0:
                    value = value.item()
            if self.reciprocal:
                delattr(scope, self.reciprocal.__name__)
        if is_map:
            scope._mapped_properties[self.__name__] = self
        else:
            scope._mapped_properties.pop(self.__name__, None)
        setattr(scope, self.cached_name, value)

    def fdel(self, scope):
        if hasattr(scope, self.cached_name):
            delattr(scope, self.cached_name)
        scope._mapped_properties.pop(self.__name__, None)

    def set_reciprocal(self, other: "PhysicalProperty"):
        self.reciprocal = other
        if other is not None:
            other.reciprocal = self

    def deriv(self, scope, v=None):
        if not self.invertible:
            # check if the reciprocal is invertible...
            recip = self.reciprocal
            # if I don't have a reciprocal, or it is also not invertible...
            if not recip or not recip.invertible:
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
        new_prop.fget = self.fget
        new_prop.fset = self.fset
        new_prop.fdel = self.fdel
        new_prop.__doc__ = self.__doc__
        return new_prop

    def getter(self, fget):
        new_prop = self.shallow_copy()
        new_prop.fget = fget
        return new_prop

    def setter(self, fset):
        new_prop = self.shallow_copy()
        new_prop.fset = fset
        return new_prop

    def deleter(self, fdel):
        new_prop = self.shallow_copy()
        new_prop.fdel = fdel
        return new_prop

    def update_invertible(self, invertible):
        new_prop = self.shallow_copy()
        new_prop.invertible = invertible
        return new_prop


class NestedModeler:
    def __init__(self, modeler_type, short_details=None):
        self.modeler_type = modeler_type
        self.short_details = short_details

    def get_property(scope, property_name):
        doc = f"""{scope.short_details}

        Returns
        -------
        {scope.modeler_type.__name__}
        """
        cached_name = f"_{property_name}"

        def fget(self):
            value = getattr(self, cached_name, None)
            if value is None:
                raise AttributeError(
                    f"'{type(self).__name__}' has no attribute '{property_name}'"
                )
            return value

        def fset(self, value):
            if value is not None:
                value = validate_type(
                    property_name, value, scope.modeler_type, cast=False
                )
                self._active_nested_modelers[property_name] = value
            else:
                self._active_nested_modelers.pop(property_name, None)
            setattr(self, cached_name, value)

        def fdel(self):
            if hasattr(self, cached_name):
                delattr(self, cached_name)
            self._active_nested_modelers.pop(property_name, None)

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
                physical_properties[key] = value
            elif isinstance(value, NestedModeler):
                nested_modelers[key] = value

        for key, value in nested_modelers.items():
            classdict[key] = value.get_property(key)

        newcls = super().__new__(mcs, name, bases, classdict)

        for parent in newcls.__mro__:
            physical_properties = (
                getattr(parent, "_physical_properties", {}) | physical_properties
            )
            nested_modelers = getattr(parent, "_nested_modelers", {}) | nested_modelers

        newcls._physical_properties = physical_properties
        newcls._invertible_properties = {
            name: prop for name, prop in physical_properties.items() if prop.invertible
        }
        newcls._nested_modelers = nested_modelers
        newcls._has_nested_models = len(nested_modelers) > 0

        return newcls


class HasModel(BaseSimPEG, metaclass=PhysicalPropertyMetaclass):

    def __init__(self, model=None, **kwargs):
        self.model = model
        super().__init__(**kwargs)

    @property
    def needs_model(self):
        """True if a model is necessary"""
        has_mapped_props = len(self._mapped_properties) > 0
        if not has_mapped_props:
            # check if my nested modelers need models.
            for modeler_name in self._active_nested_modelers:
                modeler = getattr(self, modeler_name)
                has_mapped_props |= modeler.needs_model
        return has_mapped_props

    @property
    def _mapped_properties(self):
        if (mapped := getattr(self, "_mapped_props", None)) is None:
            mapped = {}
            self._mapped_props = mapped
        return mapped

    @property
    def _active_nested_modelers(self):
        if (actv := getattr(self, "_act_nest_modelers", None)) is None:
            actv = {}
            self._act_nest_modelers = actv
        return actv

    def _prop_map(self, name):
        return self._physical_properties[name].mapping(self)

    def _prop_deriv(self, name, v=None):
        # TODO Add support for adjoints here and on mapping derivatives
        return self._physical_properties[name].deriv(self, v=v)

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
            if not self.needs_model:
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

    prop_map.__doc__ = f"""
    Mapping from the model to {old_name}

    .. deprecated:: 0.24.0
        The method of interacting with the physical property is deprecated, instead
        directly assign a mapping to {new_name}.

    Returns
    -------
    maps.IdentityMap
    """

    @property
    def prop_deriv(self):
        # derivatives are mostly used internally, and should not be publicly exposed to end users.
        warnings.warn(
            f"Accessing {deriv_name} is deprecated, use _prop_deriv('{new_name}') instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._prop_deriv(new_name)

    prop_deriv.__doc__ = f"""
    Derivative of {old_name} w.r.t. the model

    .. deprecated:: 0.24.0
        The method of interacting with the physical property derivative is deprecated. If access is still necessary
        it can be retrieved with `_get_deriv('{old_name}')`.

    Returns
    -------
    maps.IdentityMap
    """

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


class Mapping:
    # This should only really have been called by developers/ internally to simpeg,
    # Make this throw an error alerting developers to the new behavior.
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "'Mapping' is no longer necessary. You can now directly assign a map to a 'PhysicalProperty'. "
            "If you need to access to the mapping, you can do so by using the 'HasModel._prop_map' method"
        )


class Derivative:
    # This should only really have been called by developers/ internally to simpeg,
    # Make this throw an error alerting developers to the new behavior.
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "'Derivative' is no longer necessary, because you can now directly assign a map to a 'PhysicalProperty'. "
            "If you do need to access the derivative, you can do so by using the 'HasModel._prop_deriv' method"
        )


def Invertible(property_name, optional=False):
    raise NotImplementedError(
        "You no longer need to specifically create an 'Invertible' property, instead just create a 'PhysicalProperty'"
    )


def Reciprocal(prop1, prop2):
    raise NotImplementedError(
        "Use 'prop1.set_reciprocal(prop2)' within the class to set two 'PhysicalProperty' as being related by "
        "a reciprocal"
    )
