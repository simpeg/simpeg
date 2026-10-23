import enum
import inspect
import math
import os
import warnings
from types import MappingProxyType

import numpy as np

from discretize.utils import inverse_property_tensor
from simpeg.utils import deprecate_property
from . import maps
from .utils import Zero, validate_type, validate_ndarray_with_shape
from .typing import MapLike

# Warnings about how the user called into simpeg should be reported at the user's
# code, no matter how many simpeg frames (constructors, setters, ...) are in between.
_SIMPEG_DIR = (os.path.dirname(os.path.abspath(__file__)),)


def _delete_attrs(obj, names):
    """Delete each attribute in ``names`` that is set on ``obj``."""
    for name in names:
        if hasattr(obj, name):
            delattr(obj, name)


# Used for the deprecated interface that is called directly by (user or simpeg) code, so
# a warning about simpeg using its own deprecated interface is attributed to simpeg.
# (Without the extension: a prefix that is exactly the file name does not skip that file.)
_PROPS_FILE = (os.path.splitext(os.path.abspath(__file__))[0],)


def _warn_user(message, category, skip_file_prefixes=_SIMPEG_DIR):
    warnings.warn(  # noqa: B028
        message, category, skip_file_prefixes=skip_file_prefixes
    )


_REMOVAL_VERSION = "0.29.0"


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


_LOCATION_DESCRIPTION = {
    Location.CELL_CENTERS: "cell-centered",
    Location.FACES: "face-defined",
    Location.EDGES: "edge-defined",
}

_ANISOTROPY_DESCRIPTION = {
    AnisotropyLevel.ISOTROPIC: "isotropic",
    AnisotropyLevel.DIAGONAL: "isotropic or diagonally-anisotropic",
    AnisotropyLevel.FULL: "isotropic, diagonally-anisotropic, or fully-anisotropic",
}

_LOCATION_SIZE_ATTR = {
    Location.CELL_CENTERS: "n_cells",
    Location.FACES: "n_faces",
    Location.EDGES: "n_edges",
}


def _is_int(value):
    return isinstance(value, (int, np.integer))


def _n_full_tensor(dim):
    """Number of components of a symmetric full tensor in ``dim`` dimensions (as discretize uses)."""
    return 3 if dim == 2 else 6


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
            # also accept the flattened (Fortran ordered) layouts that discretize
            # uses, and that a mapping's flat output produces.
            shapes.append((size * mesh.dim,))
            if anisotropy is AnisotropyLevel.FULL:
                n_full = _n_full_tensor(mesh.dim)
                shapes.append((size, n_full))
                shapes.append((size * n_full,))
        return shapes

    return fshape


def _mesh_tensor_invert(obj, value):
    """Default `invert` for a cell-centered, anisotropy-capable property."""

    return inverse_property_tensor(obj.mesh, value)


def _shape_size(shape):
    """Total element count implied by a single shape tuple, or `None` if any
    dimension is `"*"` (unresolvable)."""
    if "*" in shape:
        return None
    return math.prod(shape)


class PhysicalProperty:
    """A physical property of a :class:`HasModel` class.

    A physical property is a descriptor. On an instance of the owning class it holds a
    value (an array or scalar, validated against the property's shape and dtype), or
    it can be parametrized by a mapping (see :meth:`HasModel.parametrize`), in which
    case its value is calculated from the instance's ``model``.

    Parameters
    ----------
    short_details : str, optional
        One line description of the property (with units), used to build its docstring.
    shape : tuple or list of tuple, optional
        The valid shape(s) of the property's value. A dimension of ``"*"`` matches any
        size. Cannot be combined with ``location``.
    default : optional
        The value of the property when it was never set. A property with a default is
        optional, otherwise it is required.
    dtype : type or tuple of type, optional
        The valid data type(s) of the value.
    reciprocal : str, optional
        The name of the property on the same class that is the reciprocal of this one
        (for example ``"rho"`` for ``"sigma"``). Either one of the pair can be set,
        and the other is calculated from it.
    invertible : bool, optional
        Whether the property can be parametrized by a mapping.
    location : Location, optional
        Where the property lives on the object's ``mesh``. When given, ``shape`` is
        resolved from the mesh.
    anisotropy : AnisotropyLevel, optional
        How anisotropic the value is allowed to be. Requires ``location``.
    invert : callable, optional
        Function, ``invert(obj, value)``, used to compute this property's value from the
        value of its reciprocal. Defaults to ``1 / value`` (or the tensor inverse for
        anisotropic properties at cell centers).
    fget, fset, fdel : callable, optional
        Custom getter, setter and deleter. Usually these are given by using the
        :meth:`getter`, :meth:`setter` and :meth:`deleter` decorators.
    fshape : callable, optional
        Function, ``fshape(obj)``, that returns the valid shape(s) for an instance.
        Usually this is given by using the :meth:`shapes` decorator.
    doc : str, optional
        The docstring of the property. Built from the other arguments if it is not given.

    See Also
    --------
    HasModel : The base class of classes that have physical properties.

    Examples
    --------
    >>> from simpeg import props, maps
    >>> class Example(props.HasModel):
    ...     sigma = props.PhysicalProperty("Electrical conductivity (S/m)")
    >>> fixed = Example(sigma=1e-2)
    >>> mapped = Example(sigma=maps.ExpMap(nP=3))
    >>> mapped.model = [-1.0, -2.0, -3.0]
    >>> mapped.sigma
    array([0.36787944, 0.13533528, 0.04978707])
    """

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
        **removed,
    ):
        for old in ("mapping", "optional"):
            if old in removed:
                raise TypeError(
                    f"`PhysicalProperty` no longer takes a `{old}` argument. "
                    f"Mappings are attached to a property with `HasModel.parametrize` "
                    f"(or by passing a mapping for the property to the constructor), "
                    f"and a property is optional when it is given a `default`."
                )
        if removed:
            raise TypeError(
                f"PhysicalProperty got unexpected keyword arguments: {sorted(removed)}"
            )
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

    def expected_output_sizes(self, obj):
        """The set of valid flattened output lengths for this property on `obj`, given its
        declared `fshape`/`shape`. Returns an empty set if this property declares no
        shape constraint at all, or if every declared shape contains a wildcard.
        """
        shape = self.fshape(obj) if self.fshape is not None else self.shape
        if shape is None:
            return set()
        shapes = [shape] if isinstance(shape, tuple) else shape
        return {size for size in (_shape_size(s) for s in shapes) if size is not None}

    def _reciprocal_output_is_full_tensor(self, obj, recip_map):
        """Best-effort check: does `recip_map`'s flattened output size match a
        full-tensor-per-location array for this property's declared `location`, given
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
        if self.location is not Location.CELL_CENTERS:
            return False
        mesh = getattr(obj, "mesh", None)
        if mesh is None or mesh.dim <= 1:
            return False
        size = getattr(mesh, _LOCATION_SIZE_ATTR[self.location])
        if not size:
            return False
        n_full = _n_full_tensor(mesh.dim)

        output_size = recip_map.shape[0]
        if _is_int(output_size):
            return output_size == size * n_full

        model = getattr(obj, "model", None)
        if model is None:
            return False
        return len(model) == size * n_full

    def build_doc(self):
        # buildup my doc string
        if self.location is not None:
            loc_str = _LOCATION_DESCRIPTION[self.location]
            aniso_str = _ANISOTROPY_DESCRIPTION[self.anisotropy]
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
        if objtype is None:
            objtype = type(obj)

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
            _warn_user(
                f"Assigning a value directly to `{cls_name}.{self.name}` removes the "
                f"existing parametrization of `{cls_name}.{removed}`; it will no "
                f"longer be invertible through that mapping.",
                UserWarning,
            )

        obj._remove_parametrization(self.name)
        if recip is not None:
            recip.__delete__(obj)
        obj._property_changed(self.name)

    def _delete_with_reciprocal(self, obj: "HasModel"):
        """Delete my value and parametrization, and my reciprocal's."""
        self.__delete__(obj)
        if recip := self.get_reciprocal(obj):
            recip.__delete__(obj)

    def __delete__(self, obj: "HasModel"):
        if self.fdel is not None:
            self.fdel(obj)
        else:
            if hasattr(obj, self.private_name):
                delattr(obj, self.private_name)
        obj._remove_parametrization(self.name)
        obj._property_changed(self.name)

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
        elif paramer := obj.parametrizations.get(self.name):
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
                _warn_user(
                    f"Setting a required physical property {type(obj).__name__}.{self.name} to None is deprecated behavior. "
                    f"This will change to an error in simpeg {_REMOVAL_VERSION}",
                    FutureWarning,
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
            if isinstance(value, np.ndarray) and value.ndim == 0:
                # store scalars as python scalars
                value = value.item()
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
    def __call__(cls, *args, **kwargs):
        # Translate the deprecated `xMap=mapping` arguments to `x=mapping`, so
        # `__init__` methods only ever see the new arguments.
        for new_name, mapping in cls._pop_deprecated_map_kwargs(
            kwargs, skip_file_prefixes=_PROPS_FILE
        ).items():
            if kwargs.get(new_name) is not None:
                raise TypeError(
                    f"Cannot pass both `{new_name}` and its deprecated mapping argument "
                    f"to {cls.__name__}."
                )
            kwargs[new_name] = mapping

        obj = super().__call__(*args, **kwargs)
        obj._construction_done = True

        # The parametrizations and the model can only be set once every `__init__` has
        # run (this makes `Sim(sigma=mapping, model=m)` work).
        # See `HasModel.__init__`.
        if pending := obj.__dict__.pop("_init_parametrizations", None):
            obj.parametrize(**pending)
        if (model := obj.__dict__.pop("_init_model", None)) is not None:
            obj.model = model
        return obj

    def _pop_deprecated_map_kwargs(cls, kwargs, skip_file_prefixes=_SIMPEG_DIR):
        """Remove the deprecated ``xMap`` arguments from ``kwargs``.

        Returns the mappings that were given (not `None`) keyed by the property
        they now parametrize, and warns about them.
        """
        translated = {}
        for old_map, new_name in cls._all_deprecated_map_kwargs().items():
            mapping = kwargs.pop(old_map, None)
            if mapping is not None:
                _warn_user(
                    f"Passing mapping argument {old_map} to {cls.__name__} is deprecated. Instead "
                    f"you can now use the {new_name} argument. It will be removed in simpeg {_REMOVAL_VERSION}.",
                    FutureWarning,
                    skip_file_prefixes,
                )
                translated[new_name] = mapping
        return translated

    def _all_deprecated_map_kwargs(cls):
        """The deprecated ``xMap`` arguments from this class and all of its parents."""
        aliases = {}
        for parent in reversed(cls.__mro__):
            aliases.update(vars(parent).get("_deprecated_map_kwargs", {}))
        return aliases

    @property
    def __signature__(cls):
        # `__call__` above would otherwise hide the signature of `__init__` from `inspect`
        sig = inspect.signature(cls.__init__)
        return sig.replace(parameters=list(sig.parameters.values())[1:])

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

        for key, value in physical_properties.items():
            if (recip_name := value._reciprocal) is None:
                continue
            recip = physical_properties.get(recip_name)
            if recip is None:
                # A mixin may define just one half of a pair (the other half comes
                # with the rest of the MRO), but a class cannot replace one that a
                # parent defined with something that is not a physical property.
                if any(
                    isinstance(vars(parent).get(recip_name), PhysicalProperty)
                    for parent in newcls.__mro__[1:]
                ):
                    raise TypeError(
                        f"`{name}` replaces `{recip_name}`, the reciprocal of the "
                        f"physical property `{key}`, with something that is not a "
                        f"PhysicalProperty. Override both properties of a reciprocal pair."
                    )
            elif recip._reciprocal != key:
                raise TypeError(
                    f"`{name}.{recip_name}` is not the reciprocal of `{name}.{key}`. "
                    f"Its `reciprocal` is {recip._reciprocal!r}. Override both properties "
                    f"of a reciprocal pair with matching reciprocals."
                )

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

    def get(self, key, default=None):
        return self._fields.get(key, default)

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

    #: Set once the constructor has finished (see `PhysicalPropertyMetaclass.__call__`).
    _construction_done = False

    #: The deprecated ``xMap`` constructor arguments, and the property they now map to.
    _deprecated_map_kwargs: dict[str, str] = MappingProxyType({})

    def __init__(self, model=None, **kwargs):
        self.model = None
        if model is not None:
            # Subclasses may set their physical properties after calling this, so the
            # model is set last (see `PhysicalPropertyMetaclass.__call__`).
            self._init_model = model
        # Deprecated mapping arguments forwarded by a subclass are applied last too.
        if translated := type(self)._pop_deprecated_map_kwargs(kwargs):
            self._init_parametrizations = translated
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
            _warn_user(
                f"Setting both `{prop1.name}` and `{prop2.name}` to None for `{type(self).__name__}`'s required "
                f"physical properties is deprecated behavior. This message will be changed to an error in simpeg "
                f"version {_REMOVAL_VERSION}",
                FutureWarning,
            )
        if inp1 is None and inp2 is None:
            # nothing to set (and the warning above already covers a missing value).
            return
        if inp2 is not None:
            self._init_property(**{prop2.name: inp2})
        else:
            self._init_property(**{prop1.name: inp1})

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
        # Validate everything before changing any state, so a failure leaves
        # the existing values and parametrizations untouched.
        replaced = set()
        checked = []
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
            if attr in replaced:
                raise TypeError(
                    f"Cannot parametrize both `{attr}` and its reciprocal in a single call."
                )
            replaced.add(attr)
            if recip := prop.get_reciprocal(self):
                replaced.add(recip.name)
            checked.append((attr, prop, parametrization))

        for attr, prop, parametrization in checked:
            others = {
                name: other_map
                for name, other_map in self.parametrizations.items()
                if name not in replaced
            }
            others.update({a: m for a, _, m in checked if a != attr})
            self._check_parametrization_sizes(attr, prop, parametrization, others)

        for attr, prop, parametrization in checked:
            # cleanup myself and my reciprocal first
            prop._delete_with_reciprocal(self)
            self.parametrizations._fields[attr] = parametrization

        self._clear_model_caches()

    def _clear_model_caches(self):
        """Delete everything cached from the previous parametrizations/model."""
        if not self._construction_done:
            # still being constructed, there is nothing cached yet.
            return
        _delete_attrs(self, self._delete_on_model_update)

    def _property_changed(self, name):
        """Called after the physical property ``name`` was set or deleted."""

    def _check_parametrization_sizes(self, attr, prop, parametrization, others):
        """Check the shape of a new ``parametrization`` of ``attr``.

        Its output length must be valid for the property, and its input length must match
        the (``others``) parametrizations that are staying, as they all share one model.
        """
        cls_name = type(self).__name__
        valid_sizes = prop.expected_output_sizes(self)
        output_size = parametrization.shape[0]
        if valid_sizes and _is_int(output_size) and output_size not in valid_sizes:
            raise ValueError(
                f"The mapping assigned to `{cls_name}.{attr}` has "
                f"an output length of {output_size}, but `{attr}` expects a "
                f"length of one of {sorted(valid_sizes)}."
            )

        input_size = parametrization.shape[1]
        if not _is_int(input_size):
            return
        for other_attr, other_map in others.items():
            other_input = other_map.shape[1]
            if _is_int(other_input) and other_input != input_size:
                raise ValueError(
                    f"Cannot parametrize `{cls_name}.{attr}` "
                    f"with a mapping expecting a model of length "
                    f"{input_size}: `{other_attr}` is already "
                    f"parametrized with a mapping expecting a model of "
                    f"length {other_input}. All parametrizations on a "
                    f"`{cls_name}` instance must accept the "
                    f"same model length."
                )

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

    def _parametrization_of(self, attr):
        """The mapping from the model to the value of ``attr``, or `None`.

        If only the reciprocal of ``attr`` is parametrized, the result is the
        reciprocal composed with that parametrization.
        """
        cls = type(self)
        parameters = self.parametrizations
        if attr in parameters:
            return parameters[attr]
        prop = getattr(cls, attr)
        recip = prop.get_reciprocal(cls)
        if recip and recip.name in parameters:
            recip_map = parameters[recip.name]
            if prop.anisotropy is AnisotropyLevel.FULL and (
                prop._reciprocal_output_is_full_tensor(self, recip_map)
            ):
                raise NotImplementedError(
                    f"Cannot compute the mapping of `{cls.__name__}.{attr}` "
                    f"through its parametrized reciprocal `{recip.name}`, which is "
                    f"currently set as a full-tensor anisotropic property: "
                    f"`maps.ReciprocalMap`'s derivative is only correct for "
                    f"scalar, isotropic, or diagonal-anisotropic values. "
                    f"Parametrize `{attr}` directly instead of `{recip.name}` "
                    f"to get a correct derivative."
                )
            return maps.ReciprocalMap() @ recip_map
        return None

    def _prop_deriv(self, attr):
        # TODO Add support for adjoints here and on mapping derivatives
        # TODO Add support for passing v to the parametrization
        paramer = self._parametrization_of(attr)
        if paramer is None:
            return Zero()
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
        self._set_model(value)

    def _validate_model(self, value):
        """Check ``value`` can be this object's model, returning it as a numpy array."""
        parameters = self.parametrizations
        if not self.needs_model:
            raise AttributeError(
                "Cannot add model as there are no parametrized properties"
                ", choose from: ['{}']".format(
                    "', '".join(
                        set(self.invertible_properties.keys()) | self._nested_modelers
                    )
                )
            )

        # coerce to a numpy array
        value = validate_ndarray_with_shape("model", value, shape=("*",), dtype=float)
        cls_name = type(self).__name__

        # Check the model is a good shape
        errors = [
            f"The parametrization, {mapping}, for '{cls_name}.{name}' expected a model "
            f"of length {mapping.shape[1]}"
            for name, mapping in parameters.items()
            if mapping.shape[1] not in ("*", len(value))
        ]
        if errors:
            raise ValueError(
                f"'{cls_name}.model' had a length of {len(value)} but expected a different "
                f"length for mappings : \n    " + "\n    ".join(errors)
            )

        # For mappings whose output shape couldn't be resolved statically
        # (and so weren't checked against the property's expected shape at
        # `parametrize()` time, see `PhysicalProperty.expected_output_sizes`), assume the
        # output length matches the model's length: a map with no declared
        # shape has no way to signal a size change, so it's effectively
        # elementwise/size-preserving -- no need to evaluate the mapping
        # just to check this.
        errors = []
        for name, mapping in parameters.items():
            if mapping.shape[0] != "*":
                continue
            valid_sizes = getattr(type(self), name).expected_output_sizes(self)
            if valid_sizes and len(value) not in valid_sizes:
                errors.append(
                    f"The parametrization, {mapping}, for '{cls_name}.{name}' "
                    f"has an unresolved output shape; the model's length ({len(value)}) "
                    f"is not one of `{name}`'s expected sizes: {sorted(valid_sizes)}"
                )
        if errors:
            raise ValueError(
                f"'{cls_name}.model' had a length of {len(value)} but is not "
                f"valid for the following parametrizations with unresolved output shapes"
                f":\n    " + "\n    ".join(errors)
            )
        return value

    def _set_nested_models(self, value):
        """Pass the model on to the nested modelers, restoring them all if one fails."""
        previous = {}
        try:
            for modeler_name in self._nested_modelers:
                modeler = getattr(self, modeler_name)
                if modeler.needs_model:
                    previous[modeler_name] = modeler.model
                    modeler.model = value
        except Exception:
            for modeler_name, previous_model in previous.items():
                getattr(self, modeler_name).model = previous_model
            raise

    def _set_model(self, value):
        """Set the model, returning `True` if it changed (and cached values were cleared)."""
        if value is not None:
            value = self._validate_model(value)
            self._set_nested_models(value)

        # trigger model update function.
        previous_value = getattr(self, "_model", None)
        updated = False
        if previous_value is not value:
            if not (
                isinstance(previous_value, np.ndarray)
                and isinstance(value, np.ndarray)
                and np.allclose(previous_value, value)
            ):
                updated = True
                # cached properties to delete
                _delete_attrs(self, self._delete_on_model_update)

        self._model = value
        return updated

    @model.deleter
    def model(self):
        self._model = None
        # cached properties to delete
        _delete_attrs(self, self._delete_on_model_update)


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
            f"use `{cls_name}.parametrizations.{new_name}` instead. It will be removed in simpeg {_REMOVAL_VERSION}.",
            FutureWarning,
            stacklevel=2,
        )
        return self._parametrization_of(new_name)

    @prop_map.setter
    def prop_map(self, value):
        cls_name = type(self).__name__
        _warn_user(
            f"Setting `{cls_name}.{old_map}` directly is deprecated. Instead pass "
            f"`{new_name}=mapping` to the constructor, or call "
            f"`{cls_name}.parametrize({new_name}=mapping)`. It will be removed in simpeg {_REMOVAL_VERSION}.",
            FutureWarning,
            skip_file_prefixes=_PROPS_FILE,
        )
        if value is None:
            # Setting a mapping to None used to remove it.
            if new_name in self.parametrizations:
                getattr(type(self), new_name).__delete__(self)
            self._clear_model_caches()
            return
        self.parametrize(**{new_name: value})

    prop_map.__doc__ = f"""
    Mapping from the model to {old_name}

    .. deprecated:: 0.27.0
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
            f"Getting `{cls_name}.{old_deriv}` is deprecated, use `{cls_name}._prop_deriv('{new_name}')` instead. It will be removed in simpeg {_REMOVAL_VERSION}.",
            FutureWarning,
            stacklevel=2,
        )
        return self._prop_deriv(new_name)

    prop_deriv.__doc__ = f"""
    Derivative of {old_name} w.r.t. the model

    .. deprecated:: 0.27.0
        The method of interacting with the physical property derivative is deprecated. If access is still necessary
        it can be retrieved with `_prop_deriv('{old_name}')`.

    Returns
    -------
    maps.IdentityMap
    """

    def decorator(cls):
        cls._deprecated_map_kwargs = MappingProxyType(
            {**vars(cls).get("_deprecated_map_kwargs", {}), old_map: new_name}
        )
        setattr(cls, old_map, prop_map)
        prop_map.__set_name__(cls, old_map)
        setattr(cls, old_deriv, prop_deriv)
        prop_deriv.__set_name__(cls, old_deriv)

        return cls

    return decorator


def _removed(name, message):
    raise NotImplementedError(f"'{name}' was removed. {message}")


class Mapping:
    # This should only really have been called by developers/ internally to simpeg,
    # Make this throw an error alerting developers to the new behavior.
    def __init__(self, *args, **kwargs):
        _removed(
            "Mapping",
            "'Mapping' is no longer necessary. You should interact with mappings using the "
            "`HasModel.parametrize` and `HasModel.parametrizations` methods.",
        )


class Derivative:
    # This should only really have been called by developers/ internally to simpeg,
    # Make this throw an error alerting developers to the new behavior.
    def __init__(self, *args, **kwargs):
        _removed(
            "Derivative",
            "'Derivative' is no longer necessary. You should interact with mappings using the "
            "`HasModel.parametrize` and `HasModel.parametrizations` methods.",
        )


def Invertible(property_name, optional=False):
    _removed(
        "Invertible",
        "You no longer need to specifically create an 'Invertible' property, instead just "
        "create a 'PhysicalProperty'.",
    )


def Reciprocal(prop1, prop2):
    _removed(
        "Reciprocal",
        "To assign reciprocal relationships for physical properties, you must pass the "
        "first physical property to the second physical property's `reciprocal` argument "
        "on initialization.",
    )
