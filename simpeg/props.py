import inspect
import warnings

import numpy as np

from simpeg.utils import deprecate_property
from . import maps
import functools
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
        dtype=float,
        reciprocal=None,
        invertible=True,
        fget=None,
        fset=None,
        fdel=None,
        doc=None,
    ):
        self.short_details = short_details
        self.default = default

        self.shape = shape
        self.dtype = dtype
        if reciprocal is not None and not isinstance(reciprocal, str):
            raise TypeError("reciprocal must be a string, or None")
        self._reciprocal = reciprocal
        self.invertible = invertible

        self.fget = fget
        self.fset = fset
        self.fdel = fdel
        self.__doc__ = doc

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

        # Try to evaluate myself
        if self.fget is not None:
            my_value = self.fget(obj)
        else:
            my_value = self._fget(obj)

        # return a successful value
        if my_value is not None:
            return my_value
        # Else try getting the reciprocal's value
        if recip := self.get_reciprocal(objtype):
            if recip.fget is not None:
                r_value = recip.fget(obj)
            else:
                r_value = recip._fget(obj, from_reciprocal=self.name)
            if r_value is not None:
                return 1.0 / r_value
        # This point in the code would mean:
        # * my getter successfully returned None
        # and
        # * recip successfully returned None (if I had a reciprocal).

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

    def __set__(self, obj: "HasModel", value):
        if self.fset is not None:
            self.fset(obj, value)
        else:
            self._fset(obj, value)
        obj._remove_parametrization(self.name)
        if recip := self.get_reciprocal(obj):
            recip.__delete__(obj)

    def __delete__(self, obj: "HasModel"):
        if self.fdel is not None:
            self.fdel(obj)
        else:
            if hasattr(obj, self.private_name):
                delattr(obj, self.private_name)
        obj._remove_parametrization(self.name)

    def _fget(self, obj, from_reciprocal=""):
        """Return my value (or calculate it from a model if I was parametrized) from an object.

        If this attribute hasn't been set, and is not parametrized, this should return `None`
        and then `__get__` will return the default value if this is optional, otherwise it will
        error.

        Parameters
        ----------
        obj
            The object to access my value from.
        from_reciprocal : str, optional
            The name of the reciprocal class calling this function

        Returns
        -------
        value
            The value of this property. Or `None` if no value was set.
        """
        if (value := getattr(obj, self.private_name, None)) is not None:
            return value
        # If I was parametrized, compute myself
        elif paramer := getattr(obj.parametrizations, self.name, None):
            if (model := obj.model) is None:
                inst_name = type(obj).__name__
                if not from_reciprocal:
                    raise AttributeError(
                        f"{inst_name}.model is required for parametrized physical property {inst_name}.{self.name}"
                    )
                else:
                    raise MissingModelError(
                        f"{inst_name}.model is required for physical property {inst_name}.{from_reciprocal}'s parameterized reciprocal {inst_name}.{self.name}"
                    )
            return paramer * model
        return None

    def _fset(self, obj, value):
        if value is None:
            if not self.optional:
                warnings.warn(
                    f"Setting a required physical property {type(obj).__name__}.{self.name} to None is deprecated behavior. "
                    f"This will change to an error in simpeg X.X",
                    FutureWarning,
                    stacklevel=4,
                )
        else:
            value = validate_ndarray_with_shape(
                f"{type(obj).__name__}.{self.name}",
                value,
                shape=self.shape,
                dtype=self.dtype,
            )
        setattr(obj, self.private_name, value)

    def get_reciprocal(self, objtype):
        """Return the reciprocal property defined on the object's class

        Use this function to get the reciprocal that is defined on the instance,
        not necessarily the exact reciprocal this was defined with. This will
        account for inheritance and re-defined getters, setters, deleters, etc...

        Parameters
        ----------
        obj

        Returns
        -------
        PhysicalProperty or None

        """
        if (recip_name := self._reciprocal) is not None:
            if not inspect.isclass(objtype):
                objtype = type(objtype)
            return getattr(objtype, recip_name)
        return None

    @property
    def has_reciprocal(self):
        return self._reciprocal is not None

    @property
    def optional(self):
        """Whether the PhysicalProperty has a default value."""
        return self.default is not _Void

    def shallow_copy(self):
        """Make a shallow copy of this PhysicalProperty."""
        copy = type(self)(
            short_details=self.short_details,
            shape=self.shape,
            default=self.default,
            dtype=self.dtype,
            reciprocal=self._reciprocal,
            invertible=self.invertible,
            fget=self.fget,
            fset=self.fset,
            fdel=self.fdel,
            doc=self.__doc__,
        )
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

    def set_feature(self, **features):
        new_prop = self.shallow_copy()
        for attr, value in features.items():
            if attr not in dir(new_prop):
                raise AttributeError(
                    f"{attr} is not a valid attribute of PhysicalProperty."
                )
            setattr(new_prop, attr, value)
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
            if (ret_val := getattr(self, f"_{scope.name}", None)) is None:
                raise AttributeError(f"NestedModeler `{scope.name}` has not been set.")
            return ret_val

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
        newcls._has_nested_models = len(nested_modelers) > 0

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

    def items(self):
        return self._fields.items()

    def keys(self):
        return self._fields.keys()

    def values(self):
        return self._fields.values()


class HasModel(BaseSimPEG, metaclass=PhysicalPropertyMetaclass):
    def __init__(self, model=None, **kwargs):
        self.model = model
        # A helper for initializing leftover property keyword arguments.
        if kwargs:
            props = self.physical_properties()
            prop_kwargs = {}
            other_kwargs = {}
            for key, value in kwargs.items():
                if key in props:
                    prop_kwargs[key] = value
                else:
                    other_kwargs[key] = value
            self._init_property(**prop_kwargs)
            kwargs = other_kwargs
        super().__init__(**kwargs)

    def _init_property(self, **kwargs):
        """Initialize physical properties, or a pair of reciprocal properties."""
        for attr, value in kwargs.items():
            if isinstance(value, maps.IdentityMap):
                self.parametrize(attr, value)
            else:
                setattr(self, attr, value)

    def _init_recip_properties(self, **kwargs):
        """Initialize a pair of reciprocal properties."""
        if len(kwargs) != 2:
            raise ValueError("Must give two reciprocal properties")
        prop1, prop2 = kwargs.keys()
        inp1, inp2 = kwargs.values()
        prop1 = getattr(type(self), prop1)
        prop2 = getattr(type(self), prop2)
        if inp1 is not None and inp2 is not None:
            raise TypeError(
                f"Can only specify one of `{prop1.name}` or `{prop2.name}` for `{type(self).__name__}`"
            )
        required = not prop1.optional and not prop2.optional
        if required and inp1 is None and inp2 is None:
            warnings.warn(
                f"Setting both `{prop1.name}` and `{prop2.name}` to None for `{type(self).__name__}`'s required "
                f"physical properties is deprecated behavior. This message will be changed to an error in simpeg "
                f"version X.X",
                FutureWarning,
                stacklevel=3,
                # f"`{type(self).__name__}` requires one of `{prop1.name}` or `{prop2.name}`"
            )
        if inp2 is not None:
            inp = {prop2.name: inp2}
        else:
            inp = {prop1.name: inp1}

        self._init_property(**inp)

    @classmethod
    def physical_properties(cls) -> dict[str, PhysicalProperty]:
        """Which physical properties are defined on this class.

        The dictionary keys are the physical property names, and their respective values
        are a `PhysicalPropertyInfo` namedtuple with `invertible` and `optional` attributes.

        Returns
        -------
        dict[str, PhysicalProperty]
        """

        props = {}
        for attr_name in dir(cls):
            attr = getattr(cls, attr_name)
            if isinstance(attr, PhysicalProperty):
                props[attr_name] = attr

        return props

    @classmethod
    def invertible_properties(cls) -> dict[str, PhysicalProperty]:
        all_props = cls.physical_properties()
        inv_props = {}
        for name, prop in all_props.items():
            if prop.invertible:
                inv_props[name] = prop
        return inv_props

    @property
    def parametrizations(self):
        """A list of parametrizations of physical properties.

        The attributes of this object, named by the physical property, return the object used
        to parametrize that physical property

        Returns
        -------
        tuple of simpeg.maps.IdentityMap

        """
        if getattr(self, "_parametrizations", None) is None:
            self._parametrizations = ParametrizationList()
        return self._parametrizations

    def parametrize(self, attr, parametrization):
        """Parametrize a physical property, so that its value is dynamically calculated from the model.

        Parameters
        ----------
        attr : str
            PhysicalProperty attribute to parametrize
        parametrization : simpeg.maps.IdentityMap
        """
        if not isinstance(parametrization, maps.IdentityMap):
            raise TypeError(
                f"simpeg currently only supports using a mapping as a PhysicalProperty parametrizer, not a {type(parametrization).__name__}"
            )

        # Let this throw an attribute error on its own
        prop = getattr(type(self), attr)
        if not isinstance(prop, PhysicalProperty):
            raise TypeError(f"{type(self).__name__}.{attr} is not a PhysicalProperty")
        if not prop.invertible:
            raise TypeError(
                f"{type(self).__name__}.{attr} is not an invertible PhysicalProperty and cannot be parametrized"
            )

        # cleanup myself and my reciprocal
        prop.__delete__(self)
        if recip := prop.get_reciprocal(self):
            recip.__delete__(self)

        self.parametrizations._fields[attr] = parametrization

    def is_parametrized(self, attr):
        """Determine if a physical property has been parametrized.

        Parameters
        ----------
        attr : str
            PhysicalProperty attribute name.

        Returns
        -------
        bool
        """
        prop = getattr(type(self), attr)
        if not isinstance(prop, PhysicalProperty):
            raise TypeError(f"{type(self).__name__}.{attr} is not a PhysicalProperty")
        if attr in self.parametrizations:
            return True
        return prop._reciprocal in self.parametrizations

    def _remove_parametrization(self, attr):
        """Remove attr's parametrization"""
        # just silently succeed here as it is an internal method.
        self.parametrizations._fields.pop(attr, None)

    def _prop_deriv(self, attr):
        # TODO Add support for adjoints here and on mapping derivatives
        # TODO Add support for passing v to the maps
        paramers = self.parametrizations
        if attr not in paramers:
            my_class = type(self)
            recip = getattr(my_class, attr).get_reciprocal(my_class)
            if recip and recip.name in paramers:
                paramer = maps.ReciprocalMap() @ paramers[recip.name]
            else:
                return Zero()
        else:
            paramer = paramers[attr]
        if self.model is not None:
            return paramer.deriv(self.model)
        else:
            raise AttributeError(
                f"{type(self).__name__}.model, required for a derivative, is not set"
            )

    @property
    def needs_model(self):
        """True if a model is necessary"""
        return bool(self.parametrizations)

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
        try:
            return self._model
        except AttributeError:
            return None

    @model.setter
    def model(self, value):
        if value is not None:
            # check if I need a model
            paramers = self.parametrizations
            if not paramers and not self._has_nested_models:
                raise AttributeError(
                    "Cannot add model as there are no parametrized properties"
                    ", choose from: ['{}']".format(
                        "', '".join(self.invertible_properties().keys())
                    )
                )

            # coerce to a numpy array
            value = validate_ndarray_with_shape(
                "model", value, shape=("*",), dtype=float
            )

            # Check the model is a good shape
            errors = []
            for name, mapping in paramers.items():
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
        cls = type(self)
        cls_name = cls.__name__
        warnings.warn(
            f"Getting `{cls_name}.{old_map}` directly is deprecated. If this is still necessary "
            f"use `{cls_name}.parametrizations.{new_name}` instead",
            FutureWarning,
            stacklevel=2,
        )
        if (my_map := getattr(self.parametrizations, new_name, None)) is not None:
            return my_map
        if recip_name := getattr(cls, new_name)._reciprocal:
            if (
                recip_map := getattr(self.parametrizations, recip_name, None)
            ) is not None:
                return maps.ReciprocalMap() @ recip_map
        return None

    @prop_map.setter
    def prop_map(self, value):
        cls_name = type(self).__name__
        warnings.warn(
            f"Setting `{cls_name}.{old_map}` directly is deprecated. Instead register a parametrization with `{cls_name}.parametrize('{new_name}')`",
            FutureWarning,
            stacklevel=2,
        )
        self.parametrize(new_name, value)

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
            FutureWarning,
            stacklevel=2,
        )
        return self._prop_deriv(new_name)

    prop_deriv.__doc__ = f"""
    Derivative of {old_name} w.r.t. the model

    .. deprecated:: 0.24.0
        The method of interacting with the physical property derivative is deprecated. If access is still necessary
        it can be retrieved with `_prop_deriv('{old_name}')`.

    Returns
    -------
    maps.IdentityMap
    """

    def decorator(cls):
        __init__ = cls.__init__

        @functools.wraps(__init__)
        def __new_init__(self, *args, **kwargs):
            mapping = kwargs.pop(old_map, None)
            if mapping is not None:
                warnings.warn(
                    f"Passing argument {old_map} to {type(self).__name__} is deprecated. Instead "
                    f"use the {new_name} argument.",
                    FutureWarning,
                    stacklevel=2,
                )
                kwargs[new_name] = mapping
            __init__(self, *args, **kwargs)

        cls.__init__ = __new_init__
        setattr(cls, old_map, prop_map)
        prop_map.__set_name__(old_map, cls)
        setattr(cls, old_deriv, prop_deriv)
        prop_deriv.__set_name__(old_deriv, cls)

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
