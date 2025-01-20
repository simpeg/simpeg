import warnings
from collections import namedtuple

import numpy as np
from numba import short

from simpeg.utils import deprecate_property
from . import maps
from .maps import IdentityMap, ReciprocalMap
from .utils import Zero, validate_type, validate_ndarray_with_shape


class _Void:
    # A class to mark no default value in a PhysicalProperty
    pass


class MissingModelError(AttributeError):
    pass


class PhysicalProperty:

    def __init__(
        self,
        short_details=None,
        shape=None,
        default=_Void,  # use this as a marker for not having a default value
        dtype=None,
        reciprocal=None,
        invertible=True,
    ):
        self.short_details = short_details
        self.default = default

        self.shape = shape
        self.dtype = dtype
        self.reciprocal = reciprocal
        if reciprocal is not None:
            reciprocal.reciprocal = self
        self.invertible = invertible

    def build_doc(self):
        # buildup my doc string
        if self.shape is None:
            shape_str = ""
        else:
            shape_str = f"{self.shape} "
        if self.optional:
            shape_str = f"None or {shape_str}"
        dtype_str = f" of {self.dtype}"
        if self.dtype is None:
            dtype_str = ""

        doc = f"""{self.short_details}

        Returns
        -------
        {shape_str}numpy.ndarray{dtype_str}
        """
        if self.invertible:
            doc += f"""
            
        Notes
        -----
        `{self.name}` is an invertible property.
        """

        return doc

    # Descriptor protocol: __set_name__, __get__, __set__, and __delete__
    def __set_name__(self, owner, name):
        # This gets called on type's __new__ method
        if not issubclass(owner, HasModel):
            raise SyntaxError("PhysicalProperty must exist on a subclass of HasModel")
        self.name = name
        self.owner = owner
        self.private_name = "_" + name
        if self.__doc__ is None:
            self.__doc__ = self.build_doc()

    def __get__(self, obj, objtype=None):
        if obj is None:
            # happens on a class (not instance)
            return self

        # basic logic is:
        # Get a value from my fget, return it if not None
        # then get a value from my reciprocal.fget, return its inverse if not None
        # then if I'm optional, return my default value
        # then if my reciprocal is optional, return the inverse of its default value (if not None), otherwise return None
        # finally issue an AttributeError if these all fail.
        my_value = self.fget(obj)
        if my_value is not None:
            return my_value
        # If my fget returned None, try the reciprocal
        if recip := self.get_class_reciprocal(objtype):
            try:
                r_value = recip.fget(self)
            # Catch this error to re-issue it with a more relevant message
            except MissingModelError:
                objname = objtype.__name__
                raise MissingModelError(
                    f"{objname}.model is required for physical property {objname}.{self.name}'s parameterized reciprocal {objname}.{self.reciprocal.name}"
                ) from None
            if r_value is not None:
                return 1.0 / r_value
        # This point in the code would mean:
        # * my fget successfully returned None
        # * recip successfully returned None (if I had one).

        # If I'm optional, return my default value
        if self.optional:
            return self.default
        # If I had an optional reciprocal
        if recip and recip.optional:
            val = recip.default
            # If it wasn't None, try to invert it.
            if val is not None:
                val = 1 / val
            return val

        # This point would be all errors
        # 1) I am a required physical property on the class
        # I am a required physical property
        inst_name = objtype.__name__
        error_message = (
            f"Required physical property {inst_name}.{self.name} has not been set"
        )
        if self.invertible:
            error_message += " nor parametrized"
        if recip:
            error_message += (
                f", and neither has its reciprocal {inst_name}.{recip.name}"
            )
        raise AttributeError(error_message)

    def __set__(self, obj, value):
        value = self.fvalidate(obj, value)
        self.fset(obj, value)

    def __delete__(self, obj):
        self.fdel(obj)

    def get_class_reciprocal(self, objtype):
        """Return the reciprocal property defined on the class

        Use this function to get the reciprocal that is defined on the instance,
        not necessarily the exact reciprocal this was defined with. This will
        account for inheritance and re-defined getters, setters, deleters, etc...

        Parameters
        ----------
        instance

        Returns
        -------
        PhysicalProperty or None

        """
        if recip := self.reciprocal:
            return getattr(objtype, recip.name)
        return None

    @property
    def optional(self):
        """Whether the PhysicalProperty has a default value."""
        return self.default is not _Void

    def fget(self, instance):
        """Return my value (or calculate it from a model if I was parametrized) from an object.

        If overwriting this function, you should not need to make a call to get its
        reciprocal value, as that should be handled by `PhysicalProperty.__get__`.

        If this attribute hasn't been set, and is not parametrized, this should return `None`
        and then `__get__` will return the default value if this is optional, otherwise it will
        error.

        Parameters
        ----------
        instance
            The object to access my value from.

        Returns
        -------
        value
            The value of this property. Or `None` if no value was set.
        """
        # If I was set with a value, get it.
        if (value := getattr(instance, self.private_name, None)) is not None:
            return value
        # If I was parametrized, compute myself
        elif paramer := getattr(instance.parametrizations, self.name, None):
            if model := instance.model is None:
                inst_name = type(instance).__name__
                raise MissingModelError(
                    f"{inst_name}.model is required for parametrized physical property {inst_name}.{self.name}"
                )
            return paramer * model
        else:
            return None

    def fvalidate(self, instance, value):
        """Validate the input value to be set on an object

        By default, this validates a physical property to be array_like with any dimension,
        and also allows `None` if the physical property is optional.

        You can overwrite this method using the `PhysicalProperty.validator` decorator.

        This is called prior to `PhysicalProperty.fset`.

        Parameters
        ----------
        value

        Returns
        -------
        value
            The validated value to be set on the class.
        """
        if value is None:
            if self.optional:
                raise TypeError(
                    f"Cannot set required physical property {type(instance).__name__}.{self.name} to None"
                )
        else:
            value = validate_ndarray_with_shape(
                f"{type(instance).__name__}.{self.name}",
                value,
                shape=self.shape,
                dtype=self.dtype,
            )
        return value

    def fset(self, instance, valid_value):
        """Set the PhysicalProperty attribute on an object with a valid value."""
        if valid_value is None:
            # Should only be validated for optional properties
            # in which case, delete my private stashed name from instance (if there was one)
            if hasattr(instance, self.private_name):
                delattr(instance, self.private_name)
        setattr(instance, self.private_name, valid_value)
        # clear any parametrization
        instance._remove_parametrization(self.name)
        # and cleanup my reciprocal
        if recip := self.reciprocal:
            delattr(instance, recip.name)

    def fdel(self, instance):
        """Delete this PhysicalProperty on an object."""
        if hasattr(instance, self.private_name):
            delattr(instance, self.private_name)
        instance._remove_parametrization(self.name)

    def shallow_copy(self):
        """Make a shallow copy of this PhysicalProperty."""
        copy = type(self)(
            short_details=self.short_details,
            shape=self.shape,
            default=self.default,
            dtype=self.dtype,
            reciprocal=self.reciprocal,
            invertible=self.invertible,
        )
        copy.fget = self.fget
        copy.fset = self.fset
        copy.fdel = self.fdel
        copy.fvalidate = self.fvalidate
        copy.__doc__ = self.__doc__
        return copy

    def getter(self, fget):
        """Decorate a function used to get the value of a PhysicalProperty."""
        new_prop = self.shallow_copy()
        new_prop.fget = fget
        return new_prop

    def setter(self, fset):
        """Decorate a function used to set a PhysicalProperty."""
        new_prop = self.shallow_copy()
        new_prop.fset = fset
        return new_prop

    def deleter(self, fdel):
        """Decorate a function used to delete a PhysicalProperty."""
        new_prop = self.shallow_copy()
        new_prop.fdel = fdel
        return new_prop

    def validator(self, fvalidate):
        """Decorate a function used to validate the input value to a PhysicalProperty."""
        new_prop = self.shallow_copy()
        new_prop.fvalidate = fvalidate
        return new_prop

    def set_default(self, value):
        new_prop = self.shallow_copy()
        new_prop.default = value
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
        nested_dict = {
            key: value
            for key, value in classdict.items()
            if isinstance(value, NestedModeler)
        }

        # set the nested_modelers as @properties
        nested_modelers = set()
        for key, value in nested_dict.items():
            value.name = key
            classdict[key] = value.get_property()
            nested_modelers.add(key)

        newcls = super().__new__(mcs, name, bases, classdict)

        for parent in reversed(newcls.__mro__):
            nested_modelers.update(getattr(parent, "_nested_modelers", set()))

        newcls._nested_modelers = nested_modelers

        return newcls


class ParametrizationList:
    __slots__ = ("_fields",)

    def __init__(self, **fields):
        self._fields = fields

    def __getitem__(self, key):
        return self._fields[key]

    def __setattr__(self, key, value):
        if key in self.__slots__:
            super().__setattr__(key, value)
        elif key in self._fields:
            raise AttributeError(f"'Cannot set attribute '{key}'") from None
        else:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{key}'"
            ) from None

    def __getattr__(self, key):
        try:
            return self._fields[key]
        except KeyError:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{key}'"
            ) from None

    def __contains__(self, key):
        return key in self._fields

    def __bool__(self):
        return bool(self._fields)

    def __len__(self):
        return len(self._fields)

    def __iter__(self):
        return iter(self._fields)


PhysicalPropertyInfo = namedtuple("PhysicalPropertyInfo", ("invertible", "optional"))


class HasModel(BaseSimPEG, metaclass=PhysicalPropertyMetaclass):
    def __init__(self, model=None, **kwargs):
        self.model = model

        self.__paramers = ParametrizationList()
        super().__init__(**kwargs)

    @classmethod
    def physical_properties(cls) -> dict[str, PhysicalPropertyInfo]:
        """Which physical properties are defined on this class.

        The dictionary keys are the physical property names, and their respective values
        are a `PhysicalPropertyInfo` namedtuple with `invertible` and `optional` attributes.

        Returns
        -------
        dict[str, PhysicalPropertyInfo]
        """
        props = {}
        for attr_name in dir(cls):
            attr = getattr(cls, attr_name)
            if isinstance(attr, PhysicalProperty):
                props[attr_name] = PhysicalPropertyInfo(attr.invertible, attr.optional)
        return props

    @property
    def parametrizations(self):
        """A list of parametrizations of physical properties.

        The attributes of this object, named by the physical property, return the object used
        to parametrize that physical property

        Returns
        -------
        tuple of simpeg.maps.IdentityMap

        """
        return self.__paramers

    def parametrize(self, attr, parametrization):
        """Parametrize a physical property, so that its value is dynamically calculated from the model.

        Parameters
        ----------
        parametrization : simpeg.maps.IdentityMap
        """
        if not isinstance(parametrization, maps.IdentityMap):
            raise TypeError(
                f"simpeg currently only supports using a mapping as a parametrizer, not a {type(parametrization).__name__}"
            )

        # Let this throw an attribute error on its own
        prop = getattr(type(self), attr)
        if not isinstance(prop, PhysicalProperty):
            raise TypeError(f"{type(self).__name__}.{attr} is not a PhysicalProperty")
        if not prop.invertible:
            raise TypeError(
                f"{type(self).__name__}.{attr} is not an invertible PhysicalProperty and cannot be parametrized"
            )

        self.parametrizations._fields[attr] = parametrization

    def _remove_parametrization(self, attr):
        """Remove attr's parametrization"""
        # just silently succeed here as it is an internal method.
        self.parametrizations._fields.pop(attr, None)

    @property
    def needs_model(self):
        """True if a model is necessary"""
        return bool(self.parametrizations)

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
            paramers = self.parametrizations
            if not paramers and not self._has_nested_models:
                raise ValueError(
                    "Cannot add model as there are no active mappings"
                    ", choose from: ['{}']".format("', '".join(self._all_map_names))
                )

            # coerce to a numpy array
            value = validate_ndarray_with_shape(
                "model", value, shape=("*",), dtype=float
            )

            # Check the model is a good shape
            errors = []
            for name, mapping in paramers:
                correct_shape = mapping.shape[1] == "*" or mapping.shape[1] == len(
                    value
                )
                if not correct_shape:
                    errors.append(
                        f"The parametrization, {mapping}, for '{type(self).__name__}.{name}' expected a model "
                        f"of length {mapping.shape[1]}"
                    )
            if len(errors) > 0:
                raise ValueError(
                    f"'{type(self).__name__}.model' had a length of {len(value)} but expected a different "
                    f"length for mappings : \n    " + "\n    ".join(errors)
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
                # cached properties to delete
                for prop in self._delete_on_model_update:
                    if hasattr(self, prop):
                        delattr(self, prop)

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
        # cached properties to delete
        for prop in self._delete_on_model_update:
            if hasattr(self, prop):
                delattr(self, prop)


def _add_deprecated_physical_property_functions(
    new_name, old_name=None, old_map=None, old_deriv=None
):

    if old_name is None:
        old_name = new_name

    if old_map is None:
        old_map = f"{old_name}Map"

    if old_deriv is None:
        old_deriv = f"{old_name}Deriv"

    @property
    def prop_map(self):
        cls_name = type(self).__name__
        warnings.warn(
            f"Getting `{cls_name}.{old_map}` directly is no longer supported. If this is still necessary "
            f"use `{cls_name}.parametrizations.{new_name}` instead",
            UserWarning,
            stacklevel=2,
        )
        return getattr(self.parametrizations, new_name)

    @prop_map.setter
    def prop_map(self, value):
        cls_name = type(self).__name__
        warnings.warn(
            f"Setting `{cls_name}.{old_map}` directly is deprecated. Instead register a parametrization with `{cls_name}.parametrize('{new_name}')`",
            UserWarning,
            stacklevel=2,
        )
        setattr(self, new_name, value)

    prop_map.__doc__ = f"""
    Mapping from the model to {old_name}

    .. deprecated:: 0.24.0
        The method of interacting with the physical property is deprecated, instead
        register a mapping with the `parametrize()` method, and access it using the
        `parametrizations` property.

    Returns
    -------
    maps.IdentityMap
    """

    @property
    def prop_deriv(self):
        cls_name = type(self).__name__
        warnings.warn(
            f"Getting `{cls_name}.{old_deriv}` is deprecated, use `{cls_name}._prop_deriv('{new_name}')` instead.",
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
            mapping = kwargs.pop(old_map, None)
            __init__(self, *args, **kwargs)
            if mapping is not None:
                setattr(self, old_map, mapping)

        cls.__init__ = __new_init__
        setattr(cls, old_map, prop_map)
        setattr(cls, old_deriv, prop_deriv)

        return cls

    return decorator


class Mapping:
    # This should only really have been called by developers/ internally to simpeg,
    # Make this throw an error alerting developers to the new behavior.
    def __init__(self, *args, **kwargs):
        raise SyntaxError(
            "'Mapping' is no longer necessary. You should interact with mappings using the `HasModel.parametrize' and "
            "'HasModel.parametrizations' methods."
        )


class Derivative:
    # This should only really have been called by developers/ internally to simpeg,
    # Make this throw an error alerting developers to the new behavior.
    def __init__(self, *args, **kwargs):
        raise SyntaxError(
            "'Derivative' is no longer necessary. You should interact with mappings using the `HasModel.parametrize' and "
            "'HasModel.parametrizations' methods."
        )


def Invertible(property_name, optional=False):
    raise SyntaxError(
        "You no longer need to specifically create an 'Invertible' property, instead just create a 'PhysicalProperty'"
    )


def Reciprocal(prop1, prop2):
    raise SyntaxError(
        "To assign reciprocal relationships for physical properties, you must pass the first physical property"
        "to the second physical properties `reciprocal` argument on initialization."
    )
