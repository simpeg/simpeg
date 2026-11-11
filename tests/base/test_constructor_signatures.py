"""Tests of the signatures of the classes with physical properties."""

import importlib
import inspect
import pkgutil

import numpy as np
import pytest

import simpeg
from simpeg import props

# (the class, the first argument that has to be given as a keyword). Each of these
# classes used to take a mapping argument (``xMap``) right before that argument.
KEYWORD_ONLY = [
    ("simpeg.base.BaseElectricalPDESimulation", "rho"),
    ("simpeg.base.BaseMagneticPDESimulation", "mui"),
    ("simpeg.electromagnetics.base_1d.BaseEM1DSimulation", "rho"),
    (
        "simpeg.electromagnetics.natural_source.simulation_1d.Simulation1DRecursive",
        "rho",
    ),
    (
        "simpeg.electromagnetics.static.resistivity.simulation_1d.Simulation1DLayers",
        "rho",
    ),
    (
        "simpeg.electromagnetics.static.induced_polarization.simulation.BaseIPSimulation",
        "Ainv",
    ),
    (
        "simpeg.electromagnetics.static.spectral_induced_polarization.simulation.BaseSIPSimulation",
        "taui",
    ),
    ("simpeg.flow.richards.empirical.Haverkamp_k", "A"),
    ("simpeg.flow.richards.empirical.Haverkamp_theta", "theta_s"),
    ("simpeg.flow.richards.empirical.Vangenuchten_k", "I"),
    ("simpeg.flow.richards.empirical.Vangenuchten_theta", "theta_s"),
    ("simpeg.potential_fields.gravity.simulation.Simulation3DIntegral", "engine"),
    ("simpeg.potential_fields.magnetics.simulation.Simulation3DIntegral", "model_type"),
    ("simpeg.potential_fields.magnetics.simulation.Simulation3DDifferential", "rem"),
    (
        "simpeg.seismic.straight_ray_tomography.simulation.Simulation2DIntegral",
        "velocity",
    ),
    ("simpeg.simulation.LinearSimulation", "G"),
]


def _import_class(path):
    module, name = path.rsplit(".", 1)
    return getattr(importlib.import_module(module), name)


@pytest.mark.parametrize("path, name", KEYWORD_ONLY)
def test_arguments_after_removed_mapping_arguments_are_keyword_only(path, name):
    parameter = inspect.signature(_import_class(path)).parameters[name]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY


def _module_names(path, prefix):
    for info in pkgutil.iter_modules(path, prefix):
        if info.name.startswith("simpeg.dask"):
            # importing these patches the simulations in the rest of simpeg
            continue
        yield info.name
        if info.ispkg:
            try:
                package = importlib.import_module(info.name)
            except ImportError:
                continue
            yield from _module_names(package.__path__, info.name + ".")


def _has_model_classes():
    classes = {}
    for module_name in _module_names(simpeg.__path__, "simpeg."):
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        for cls in vars(module).values():
            if isinstance(cls, type) and issubclass(cls, props.HasModel):
                classes[f"{cls.__module__}.{cls.__qualname__}"] = cls
    return [classes[key] for key in sorted(classes)]


@pytest.mark.parametrize("cls", _has_model_classes(), ids=lambda cls: cls.__qualname__)
def test_property_defaults_match_constructor_defaults(cls):
    parameters = inspect.signature(cls).parameters
    for name, prop in cls.physical_properties.items():
        parameter = parameters.get(name)
        if parameter is None or not prop.optional:
            continue
        if parameter.default is inspect.Parameter.empty or parameter.default is None:
            continue
        np.testing.assert_equal(prop.default, parameter.default)
