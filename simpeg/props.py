import enum
import inspect
import warnings
from types import MappingProxyType

import numpy as np

from discretize.utils import inverse_property_tensor
from simpeg.utils import deprecate_property
from . import maps
import functools
from .utils import Zero, validate_type, validate_ndarray_with_shape
from .typing import MapLike


class _Void:
    # A class to mark no default value in a PhysicalProperty
    pass


class Location(enum.Enum):
    """Where a `PhysicalProperty` lives relative to a mesh."""

    CELL_CENTERS = "cell_centers"
    FACES = "faces"
    EDGES = "edges"


class AnisotropyLevel(enum.IntEnum):
    """The level of anisotropy a `PhysicalProperty` supports.

    Values align with discretize's ``TensorType`` classification
    (isotropic=1, diagonal-anisotropic=2, full-tensor-anisotropic=3).
    """

    ISOTROPIC = 1
    DIAGONAL = 2
    FULL = 3


_LOCATION_SIZE_ATTR = {
    Location.CELL_CENTERS: "n_cells",
    Location.FACES: "n_faces",
    Location.EDGES: "n_edges",
}


def _mesh_shape_resolver(location, anisotropy):
    """Build an `fshape` function for a property living at `location` on `obj.mesh`.

    This is a convenience default derived from `location`/`anisotropy`; it has
    no special status and can always be replaced with a custom function via
    `PhysicalProperty.shapes`.
    """

    def fshape(obj):
        mesh = obj.mesh
        if mesh is None:
            raise AttributeError(
                f"`{type(obj).__name__}.mesh` is required to validate this property."
            )
        size = getattr(mesh, _LOCATION_SIZE_ATTR[location])
        shapes = [(), (1,), (size,)]
        if anisotropy is not AnisotropyLevel.ISOTROPIC and mesh.dim > 1:
            shapes.append((size, mesh.dim))
            if anisotropy is AnisotropyLevel.FULL:
                shapes.append((size, 3 if mesh.dim == 2 else 6))
        return shapes

    return fshape


def _mesh_tensor_invert(obj, value):
    """Default `invert` for a cell-centered, anisotropy-capable property."""

    return inverse_property_tensor(obj.mesh, value)


def _reciprocal_output_is_full_tensor(prop, obj, recip_map):
    """Best-effort check: does `recip_map`'s flattened output size match a
    full-tensor-per-location array for `prop`'s declared `location`, given
    `obj.mesh`?

    Used to guard against `maps.ReciprocalMap`, whose elementwise transform/
    derivative is only correct for scalar, isotropic, or diagonal-anisotropic
    values -- not a genuine full tensor (with off-diagonal terms).

    If `recip_map`'s own declared output shape is ambiguous (`"*"`, i.e. no
    `mesh`/`nP` given), falls back to comparing the model's length instead: a
    map with no declared shape has no way to change the length of what it's
    given, since it isn't wired to know it should reshape to a specific size --
    it is, in effect, elementwise/size-preserving. So a model already shaped
    for a full tensor is a reasonable proxy for the mapping's output shape too.
    Returns `False` (assume safe) only when neither can be determined.
    """
    if prop.location is not Location.CELL_CENTERS:
        return False
    mesh = getattr(obj, "mesh", None)
    if mesh is None or mesh.dim <= 1:
        return False
    size = getattr(mesh, _LOCATION_SIZE_ATTR[prop.location])
    if not size:
        return False
    n_full = 3 if mesh.dim == 2 else 6

    output_size = recip_map.shape[0]
    if isinstance(output_size, (int, np.integer)):
        return output_size == size * n_full

    model = getattr(obj, "model", None)
    if model is None:
        return False
    return len(model) == size * n_full


class PhysicalProperty:

    def __init__(
        self,
        short_details=None,
        shape=None,
        default=_Void,  # use this as a marker for not having a default value
        dtype=float,
        reciprocal=None,
        invertible=True,
        location=None,
        anisotropy=AnisotropyLevel.ISOTROPIC,
        invert=None,
        fget=None,
        fset=None,
        fdel=None,
        fshape=None,
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

        if location is not None and shape is not None:
            raise ValueError(
                "Cannot specify `location` together with `shape`; `location` "
                "builds a default shape resolver automatically. Pass an explicit "
                "`fshape` alongside `location` instead if a custom resolver is needed."
            )
        if location is None and anisotropy is not AnisotropyLevel.ISOTROPIC:
            raise ValueError("`anisotropy` can only be set when `location` is given.")
        if (
            location in (Location.FACES, Location.EDGES)
            and anisotropy is not AnisotropyLevel.ISOTROPIC
        ):
            raise ValueError(
                "Anisotropy beyond ISOTROPIC is only supported at "
                f"`Location.CELL_CENTERS`, got location={location}, "
                f"anisotropy={anisotropy}."
            )
        self.location = location
        self.anisotropy = anisotropy

        # `fshape` is the generic override hook: PhysicalProperty itself has no
        # built-in notion of a mesh/location beyond this. `location` is only a
        # convenience that builds a default `fshape` (below) when one isn't
        # already given; it can always be overridden via `.shapes()`.
        self.fshape = fshape
        self.invert = invert
        self._derive_shape_and_invert()

        self.fget = fget
        self.fset = fset
        self.fdel = fdel
        self.__doc__ = doc

    def _derive_shape_and_invert(self):
        """Populate `fshape`/`invert` from `location`/`anisotropy` if not already set.

        Only fills in a default when the corresponding hook is still `None` —
        an explicitly given (or previously overridden) `fshape`/`invert` is
        never replaced. Called from `__init__`, and from `set_feature` after
        resetting `fshape`/`invert` to `None` when `location`/`anisotropy`
        are among the features being changed (see `set_feature`).
        """
        if self.location is not None and self.fshape is None:
            self.fshape = _mesh_shape_resolver(self.location, self.anisotropy)

        if (
            self.location is Location.CELL_CENTERS
            and self.anisotropy is not AnisotropyLevel.ISOTROPIC
            and self.invert is None
        ):
            self.invert = _mesh_tensor_invert

    def build_doc(self):
        # buildup my doc string
        if self.location is not None:
            loc_str = {
                Location.CELL_CENTERS: "cell-centered",
                Location.FACES: "face-defined",
                Location.EDGES: "edge-defined",
            }[self.location]
            aniso_str = {
                AnisotropyLevel.ISOTROPIC: "isotropic",
                AnisotropyLevel.DIAGONAL: "isotropic or diagonally-anisotropic",
                AnisotropyLevel.FULL: (
                    "isotropic, diagonally-anisotropic, or fully-anisotropic"
                ),
            }[self.anisotropy]
            shape_str = f"scalar, or {loc_str} ({aniso_str}) "
        elif self.shape is None:
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
                return self._invert(obj, r_value)
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
                val = self._invert(obj, val)
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
        recip = self.get_reciprocal(obj)
        if self.name in obj.parametrizations:
            removed = self.name
        elif recip is not None and recip.name in obj.parametrizations:
            removed = recip.name
        else:
            removed = None

        if self.fset is not None:
            self.fset(obj, value)
        else:
            self._fset(obj, value)

        if removed is not None:
            cls_name = type(obj).__name__
            warnings.warn(
                f"Assigning a value directly to `{cls_name}.{self.name}` removes the "
                f"existing parametrization of `{cls_name}.{removed}`; it will no "
                f"longer be invertible through that mapping.",
                UserWarning,
                stacklevel=3,
            )

        obj._remove_parametrization(self.name)
        if recip is not None:
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
                    raise AttributeError(
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
            shape = self.fshape(obj) if self.fshape is not None else self.shape
            try:
                value = validate_ndarray_with_shape(
                    f"{type(obj).__name__}.{self.name}",
                    value,
                    shape=shape,
                    dtype=self.dtype,
                )
            except TypeError:
                if isinstance(value, maps.IdentityMap):
                    cls_name = type(obj).__name__
                    raise TypeError(
                        f"Cannot assign a mapping directly to `{cls_name}.{self.name}`. Instead "
                        f"pass `{self.name}=mapping` to the constructor, or call "
                        f"`{cls_name}.parametrize({self.name}=mapping)`."
                    ) from None
                raise
        setattr(obj, self.private_name, value)

    def _invert(self, obj, value):
        """Invert `value` (my reciprocal's value) to produce my own value."""
        if self.invert is not None:
            return self.invert(obj, value)
        return 1.0 / value

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
            location=self.location,
            anisotropy=self.anisotropy,
            invert=self.invert,
            fget=self.fget,
            fset=self.fset,
            fdel=self.fdel,
            fshape=self.fshape,
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

    def shapes(self, fshape):
        """Decorate a function used to compute this PhysicalProperty's valid shape(s) dynamically.

        The function should accept the `HasModel` instance and return a shape
        (or list of shapes) suitable for `simpeg.utils.validate_ndarray_with_shape`.
        """
        new_prop = self.shallow_copy()
        new_prop.fshape = fshape
        return new_prop

    def set_feature(self, **features):
        new_prop = self.shallow_copy()
        for attr, value in features.items():
            if attr not in dir(new_prop):
                raise AttributeError(
                    f"{attr} is not a valid attribute of PhysicalProperty."
                )
            setattr(new_prop, attr, value)

        # `fshape`/`invert` may have been derived from the *original*
        # `location`/`anisotropy` at construction time. If either changed here
        # (without also explicitly overriding `fshape`/`invert` in this same
        # call), the stale derived hooks must be rebuilt for the new values --
        # otherwise they'd silently keep validating/inverting against the old
        # anisotropy level.
        if "location" in features or "anisotropy" in features:
            if "fshape" not in features:
                new_prop.fshape = None
            if "invert" not in features:
                new_prop.invert = None
            new_prop._derive_shape_and_invert()

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

        # collect all `PhysicalProperty` descriptors visible on this class, walking
        # the MRO from base to derived so subclass overrides win. A subclass that
        # shadows an inherited PhysicalProperty with a non-PhysicalProperty attribute
        # must also drop the stale inherited entry.
        physical_properties = {}
        for parent in reversed(newcls.__mro__):
            for key, value in vars(parent).items():
                if isinstance(value, PhysicalProperty):
                    physical_properties[key] = value
                elif key in physical_properties:
                    del physical_properties[key]

        newcls.physical_properties = MappingProxyType(physical_properties)
        newcls.invertible_properties = MappingProxyType(
            {
                key: value
                for key, value in physical_properties.items()
                if value.invertible
            }
        )

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
            raise AttributeError(
                f"Cannot set attribute '{key}': '{type(self).__name__}' is read-only. "
                f"Use `HasModel.parametrize({key}=mapping)` instead."
            ) from None
        else:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{key}'"
            ) from None

    def __getattr__(self, key):
        if key != "_fields":
            try:
                return self._fields[key]
            except KeyError:
                raise AttributeError(
                    f"'{type(self).__name__}' object has no attribute '{key}'"
                ) from None
        raise AttributeError

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

    def __str__(self):
        inner = ", ".join(f"{key}={value!r}" for key, value in self._fields.items())
        return f"{type(self).__name__}({inner})"

    __repr__ = __str__


class HasModel(BaseSimPEG, metaclass=PhysicalPropertyMetaclass):
    """Base class giving a class `PhysicalProperty` descriptors and a `model`.

    Attributes
    ----------
    physical_properties : dict[str, PhysicalProperty]
        Which physical properties are defined on this class, keyed by attribute
        name. Set by `PhysicalPropertyMetaclass` from the `PhysicalProperty`
        descriptors visible on the class (including inherited ones).
    invertible_properties : dict[str, PhysicalProperty]
        The subset of `physical_properties` whose `PhysicalProperty.invertible`
        is `True`.
    """

    # Placeholder values so the attribute (with its type and docstring below) is
    # statically visible to IDEs, type checkers, and Sphinx. `PhysicalPropertyMetaclass`
    # overwrites these with the real, per-class computed values right after the class
    # is created; see `PhysicalPropertyMetaclass.__new__`.
    physical_properties: dict[str, "PhysicalProperty"] = MappingProxyType({})
    """Which physical properties are defined on this class, keyed by attribute name."""

    invertible_properties: dict[str, "PhysicalProperty"] = MappingProxyType({})
    """The subset of `physical_properties` that are invertible."""

    def __init__(self, model=None, **kwargs):
        self.model = model
        # A helper for initializing leftover property keyword arguments.
        if kwargs:
            props = self.physical_properties
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
                self.parametrize(**{attr: value})
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

    def parametrize(self, **kwargs: MapLike):
        """Parametrize a physical property, so that its value is dynamically calculated from the model.

        Parameters
        ----------
        **kwargs : dict[str, simpeg.maps.IdentityMap]
            Each attribute keyword is parametrized by the given relationship.
            Accepts any `simpeg.maps.IdentityMap` instance (or, informally,
            any :class:`~simpeg.typing.MapLike`-shaped object, though only
            `IdentityMap` subclasses are currently accepted at runtime; see
            `simpeg.typing.MapLike`'s docstring for why).

        Examples
        --------

        >>> sim.parametrize(sigma=maps.ExpMap())

        """
        for attr, parametrization in kwargs.items():
            if not isinstance(parametrization, maps.IdentityMap):
                raise TypeError(
                    f"simpeg currently only supports using a mapping as a PhysicalProperty parametrizer, not a {type(parametrization).__name__}"
                )

            # Let this throw an attribute error on its own
            prop = getattr(type(self), attr)
            if not isinstance(prop, PhysicalProperty):
                raise TypeError(
                    f"{type(self).__name__}.{attr} is not a PhysicalProperty"
                )
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
        """Determine if a physical property (or its reciprocal) has been parametrized.

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
        # TODO Add support for passing v to the parametrization
        parameters = self.parametrizations
        if attr not in parameters:
            my_class = type(self)
            prop = getattr(my_class, attr)
            recip = prop.get_reciprocal(my_class)
            if recip and recip.name in parameters:
                recip_map = parameters[recip.name]
                if prop.anisotropy is AnisotropyLevel.FULL and (
                    _reciprocal_output_is_full_tensor(prop, self, recip_map)
                ):
                    raise NotImplementedError(
                        f"Cannot compute the derivative of `{my_class.__name__}.{attr}` "
                        f"through its parametrized reciprocal `{recip.name}`, which is "
                        f"currently set as a full-tensor anisotropic property: "
                        f"`maps.ReciprocalMap`'s derivative is only correct for "
                        f"scalar, isotropic, or diagonal-anisotropic values. "
                        f"Parametrize `{attr}` directly instead of `{recip.name}` "
                        f"to get a correct derivative."
                    )
                paramer = maps.ReciprocalMap() @ recip_map
            else:
                return Zero()
        else:
            paramer = parameters[attr]
        if self.model is not None:
            return paramer.deriv(self.model)
        else:
            raise AttributeError(
                f"{type(self).__name__}.model, required for a derivative, is not set"
            )

    @property
    def needs_model(self):
        """True if a model is necessary"""

        needs_model = bool(self.parametrizations)

        if not needs_model and self._has_nested_models:
            needs_model = any(
                getattr(self, modeler_name).needs_model
                for modeler_name in self._nested_modelers
            )

        return needs_model

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
        error=True,
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
            parameters = self.parametrizations
            if not self.needs_model:
                raise AttributeError(
                    "Cannot add model as there are no parametrized properties"
                    ", choose from: ['{}']".format(
                        "', '".join(
                            set(self.invertible_properties.keys())
                            | self._nested_modelers
                        )
                    )
                )

            # coerce to a numpy array
            value = validate_ndarray_with_shape(
                "model", value, shape=("*",), dtype=float
            )

            # Check the model is a good shape
            errors = []
            for name, mapping in parameters.items():
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

        self._model = value

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
                prop = getattr(cls, new_name)
                if prop.anisotropy is AnisotropyLevel.FULL and (
                    _reciprocal_output_is_full_tensor(prop, self, recip_map)
                ):
                    raise NotImplementedError(
                        f"Cannot compute `{cls_name}.{old_map}` from the "
                        f"parametrized reciprocal `{recip_name}`, which is "
                        f"currently set as a full-tensor anisotropic property: "
                        f"`maps.ReciprocalMap` is only correct for scalar, "
                        f"isotropic, or diagonal-anisotropic values. Parametrize "
                        f"`{new_name}` directly instead."
                    )
                return maps.ReciprocalMap() @ recip_map
        return None

    @prop_map.setter
    def prop_map(self, value):
        cls_name = type(self).__name__
        warnings.warn(
            f"Setting `{cls_name}.{old_map}` directly is deprecated. Instead pass "
            f"`{new_name}=mapping` to the constructor, or call "
            f"`{cls_name}.parametrize({new_name}=mapping)`.",
            FutureWarning,
            stacklevel=2,
        )
        self.parametrize(**{new_name: value})

    prop_map.__doc__ = f"""
    Mapping from the model to {old_name}

    .. deprecated:: X.Y
        The method of interacting with the physical property is deprecated. Instead
        pass `{new_name}=mapping` directly to the constructor (the simplest
        replacement for most use cases), or call the `parametrize()` method and
        access it using the `parametrizations` property.

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

    .. deprecated:: X.Y
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
                    f"Passing mapping argument {old_map} to {type(self).__name__} is deprecated. Instead "
                    f"you can now use the {new_name} argument.",
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
