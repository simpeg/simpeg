import numpy as np

from .maps import IdentityMap, ReciprocalMap
from .utils import Zero, validate_type, validate_ndarray_with_shape


class Mapping:
    def __init__(self, short_details=None):
        self.short_details = short_details

    @property
    def prop(self):
        return getattr(self, "_prop", None)

    @prop.setter
    def prop(self, value):
        value = validate_type("prop", value, PhysicalProperty, cast=False)
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
                delattr(instance, prop.name)

    def get_property(scope):
        doc = f"""{scope.short_details}

        Returns
        -------
        SimPEG.maps.IdentityMap
        """

        def fget(self):
            value = getattr(self, f"_{scope.name}", None)
            if value is not None:
                return value
            if scope.reciprocal is None:
                return None
            reciprocal = getattr(self, f"_{scope.reciprocal.name}", None)
            if reciprocal is None:
                return None
            return ReciprocalMap() * reciprocal

        def fset(self, value):
            if value is not None:
                value = validate_type(scope.name, value, IdentityMap, cast=False)
                scope.clear_props(self)
            setattr(self, f"_{scope.name}", value)

        def fdel(self):
            setattr(self, f"_{scope.name}", None)

        return property(fget=fget, fset=fset, fdel=fdel, doc=doc)


class PhysicalProperty:
    reciprocal = None

    def __init__(
        self, short_details, mapping=None, shape=None, default=None, dtype=None
    ):
        self.short_details = short_details
        if mapping is not None:
            mapping.prop = self

        self._mapping = mapping

        self.shape = shape
        self.dtype = None

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    @property
    def mapping(self):
        return self._mapping

    @mapping.setter
    def mapping(self, value):
        value = validate_type("mapping", value, Mapping, cast=False)
        value._prop = self  # Skip the setter
        self._mapping = value

    def clear_mappings(self, instance):
        if self.mapping is not None:
            delattr(instance, self.mapping.name)
        if self.reciprocal is not None:
            if self.reciprocal.mapping is not None:
                delattr(instance, self.reciprocal.mapping.name)

    def get_property(scope):
        if scope.shape is None:
            shape_str = ""
        else:
            shape_str = f"{scope.shape} "
        dtype_str = f" of {scope.dtype}"
        if scope.dtype is None:
            dtype_str = ""

        doc = f"""{scope.short_details}

        Returns
        -------
        {shape_str}numpy.ndarray{dtype_str}
        """

        def fget(self):
            value = getattr(self, f"_{scope.name}", None)
            if value is not None:
                return value
            if scope.reciprocal:
                value = getattr(self, f"_{scope.reciprocal.name}", None)
                if value is not None:
                    return 1.0 / value
            # If I don't have a mapping
            if scope.mapping is None:
                # I done have a reciprocal, or it doesn't have a mapping
                if scope.reciprocal is None:
                    return None
                if scope.reciprocal.mapping is None:
                    reciprocal_val = getattr(self, f"_{scope.reciprocal.name}", None)
                    if reciprocal_val is None:
                        raise AttributeError(
                            "Neither a value nor mapping for {}/{} has been set".format(
                                scope.name, scope.reciprocal.name
                            )
                        )
                # Set by mapped reciprocal
                print("returning this thing?")
                return 1.0 / getattr(self, scope.reciprocal.name)

            mapping = getattr(self, scope.mapping.name, None)
            if mapping is None:
                raise AttributeError(
                    f"Neither a value for `{scope.name}` or mapping for `{scope.mapping.name}` has not been set."
                )
            if self.model is None:
                raise AttributeError(
                    f"A `model` is required for physical property {scope.name}"
                )
            return mapping * self.model

        def fset(self, value):
            if value is not None:
                value = validate_ndarray_with_shape(
                    scope.name, value, shape=scope.shape, dtype=scope.dtype
                )
                if value.ndim == 0:
                    value = value.item()

                if scope.reciprocal:
                    delattr(self, scope.reciprocal.name)
                scope.clear_mappings(self)
            setattr(self, f"_{scope.name}", value)

        def fdel(self):
            setattr(self, f"_{scope.name}", None)

        return property(fget=fget, fset=fset, fdel=fdel, doc=doc)


class Derivative:
    def __init__(self, short_details=None, physical_property=None):
        self.short_details = short_details
        self.physical_property = physical_property

    @property
    def mapping(self):
        """The mapping looks through to the physical property map."""
        if self.physical_property is None:
            return None
        return self.physical_property.mapping

    def get_property(scope):
        doc = f"""{scope.short_details}

        Returns
        -------
        scipy.sparse.spmatrix, discretize.Zero, or discretize.Identity
        """

        def fget(self):
            if scope.physical_property is None:
                return Zero()
            if scope.mapping is None:
                return Zero()
            mapping = getattr(self, scope.mapping.name)
            if mapping is None:
                return Zero()
            if self.model is None:
                return Zero()

            return mapping.deriv(self.model)

        return property(fget=fget, doc=doc)


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


def Invertible(property_name):
    mapping = Mapping(f"Mapping of the inversion model to {property_name}.")

    physical_property = PhysicalProperty(
        f"{property_name.capitalize()} physical property model.",
        mapping=mapping,
    )

    property_derivative = Derivative(
        f"Derivative of {property_name} wrt the model.",
        physical_property=physical_property,
    )

    return physical_property, mapping, property_derivative


def Reciprocal(prop1, prop2):
    prop1.reciprocal = prop2
    prop2.reciprocal = prop1


class BaseSimPEG:
    """"""


class PhysicalPropertyMetaclass(type):
    def __new__(mcs, name, bases, classdict):
        # set the phyiscal properties list.

        property_dict = {
            key: value
            for key, value in classdict.items()
            if isinstance(value, PhysicalProperty)
        }
        map_dict = {
            key: value for key, value in classdict.items() if isinstance(value, Mapping)
        }
        deriv_dict = {
            key: value
            for key, value in classdict.items()
            if isinstance(value, Derivative)
        }
        nested_dict = {
            key: value
            for key, value in classdict.items()
            if isinstance(value, NestedModeler)
        }

        # set the physical properties as @properties
        for key, value in property_dict.items():
            value.name = key
            classdict[key] = value.get_property()

        map_names = classdict.get("_all_map_names", set())
        # set the mappings as @properties
        for key, value in map_dict.items():
            value.name = key
            classdict[key] = value.get_property()
            map_names.add(key)

        # set the derivatives as @properties
        for key, value in deriv_dict.items():
            value.name = key
            classdict[key] = value.get_property()

        # set the nested_modelers as @properties
        nested_modelers = set()
        for key, value in nested_dict.items():
            value.name = key
            classdict[key] = value.get_property()
            nested_modelers.add(key)

        newcls = super().__new__(mcs, name, bases, classdict)

        for parent in reversed(newcls.__mro__):
            map_names.update(getattr(parent, "_all_map_names", set()))
            nested_modelers.update(getattr(parent, "_nested_modelers", set()))

        newcls._all_map_names = map_names
        newcls._nested_modelers = nested_modelers

        return newcls


class HasModel(BaseSimPEG, metaclass=PhysicalPropertyMetaclass):
    def __init__(self, model=None, **kwargs):
        self.model = model
        super().__init__(**kwargs)

    @property
    def _act_map_names(self):
        return set(
            name
            for name in self._all_map_names
            if getattr(self, f"_{name}", None) is not None
        )

    @property
    def needs_model(self):
        """True if a model is necessary"""
        return len(self._act_map_names) > 0

    @property
    def _has_nested_models(self):
        return len(self._nested_modelers) > 0

    # TODO: rename to _delete_on_model_update
    @property
    def deleteTheseOnModelUpdate(self):
        """A list of properties stored on this object to delete when the model is updated

        Returns
        -------
        list of str
            For example `['_MeSigma', '_MeSigmaI']`.
        """
        return []

    #: List of matrix names to have their factors cleared on a model update
    @property
    def clean_on_model_update(self):
        """A list of solver objects to clean when the model is updated

        Returns
        -------
        list of str
        """
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
            for name in self._act_map_names:
                mapping = getattr(self, name)
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
                # cached properties to delete
                for prop in self.deleteTheseOnModelUpdate:
                    if hasattr(self, prop):
                        delattr(self, prop)

                # matrix factors to clear
                for mat in self.clean_on_model_update:
                    if getattr(self, mat, None) is not None:
                        getattr(self, mat).clean()  # clean factors
                        setattr(self, mat, None)  # set to none
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
        for prop in self.deleteTheseOnModelUpdate:
            if hasattr(self, prop):
                delattr(self, prop)

        # matrix factors to clear
        for mat in self.clean_on_model_update:
            if getattr(self, mat, None) is not None:
                getattr(self, mat).clean()  # clean factors
                setattr(self, mat, None)  # set to none
