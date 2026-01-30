import inspect
import re
import warnings

import numpy as np
import numpy.testing as npt
import pytest
from scipy.constants import mu_0
from discretize.tests import check_derivative

from simpeg import maps
from simpeg.base import (
    ElectricalConductivity,
    MagneticPermeability,
    DielectricPermittivity,
    MassDensity,
    LayerThickness,
    ElectricalChargeability,
    MagneticSusceptibility,
    AcousticVelocity,
    HydraulicConductivity,
    WaterRetention,
    ColeCole,
    ViscousMagneticSusceptibility,
    AmalgamatedViscousMagneticSusceptibility,
)

RECIPROCALS = {
    ElectricalConductivity,
    MagneticPermeability,
    AcousticVelocity,
    ColeCole,
}

DEFAULT_VALUED = {
    MagneticPermeability,
    LayerThickness,
    DielectricPermittivity,
}

NOT_INVERTIBLE = {
    DielectricPermittivity,
    ViscousMagneticSusceptibility,
}

INVERTIBLE = {
    ElectricalConductivity,
    MagneticPermeability,
    MassDensity,
    LayerThickness,
    ElectricalChargeability,
    MagneticSusceptibility,
    AcousticVelocity,
    HydraulicConductivity,
    WaterRetention,
    ColeCole,
    AmalgamatedViscousMagneticSusceptibility,
}
ALL_CLASSES = NOT_INVERTIBLE | INVERTIBLE


def _param_prop_filter(param):
    name = param.name
    return not ("Map" in name or "Deriv" in name or name == "kwargs")


@pytest.mark.parametrize("prop_class", ALL_CLASSES - DEFAULT_VALUED)
def test_unassigned(prop_class):
    modeler = prop_class()
    params = inspect.signature(prop_class).parameters
    props = [param.name for param in params.values() if _param_prop_filter(param)]
    for prop in props:
        print(prop)
        with pytest.raises(AttributeError):
            out = getattr(modeler, prop)
            print(out)


@pytest.mark.parametrize("prop_class", DEFAULT_VALUED)
def test_defaults(prop_class):
    modeler = prop_class()
    if prop_class == MagneticPermeability:
        assert modeler.mu == mu_0
        assert modeler.mui == 1.0 / mu_0
    elif prop_class == LayerThickness:
        npt.assert_equal(modeler.thicknesses, [])
    else:  # prop_class == Dielectric
        assert modeler.permittivity is None


@pytest.mark.parametrize("prop_class", ALL_CLASSES)
@pytest.mark.parametrize("inp_type", ["scalar", "array"])
def test_assignment(prop_class, inp_type):
    modeler = prop_class()
    params = inspect.signature(prop_class).parameters
    props = [param.name for param in params.values() if _param_prop_filter(param)][:2]
    for prop in props:
        if inp_type == "scalar":
            inp = np.pi
        else:  # inp_type == array
            inp = np.linspace(10, 12)

        setattr(modeler, prop, inp)
        calc = getattr(modeler, prop)
        npt.assert_equal(calc, inp)


@pytest.mark.parametrize("prop_class", INVERTIBLE)
def test_invertible_map_assign(prop_class):
    modeler = prop_class()
    params = inspect.signature(prop_class).parameters
    props = [param.name for param in params.values() if _param_prop_filter(param)]
    for prop in props:
        inp = np.linspace(10, 12)
        setattr(modeler, prop + "Map", maps.IdentityMap())

        modeler.model = inp
        calc = getattr(modeler, prop)

        npt.assert_equal(calc, inp)


@pytest.mark.parametrize("prop_class", INVERTIBLE)
def test_derivatives(prop_class):
    modeler = prop_class()
    prop_map = maps.ExpMap()
    params = inspect.signature(prop_class).parameters
    props = [param.name for param in params.values() if _param_prop_filter(param)]
    for prop in props:
        inp = np.linspace(10, 12)
        setattr(modeler, prop + "Map", prop_map)

        def deriv_test(x, attr=prop):
            modeler.model = x
            out = getattr(modeler, attr)

            map_deriv = getattr(modeler, attr + "Deriv")
            return out, lambda u: map_deriv @ u

        check_derivative(deriv_test, x0=inp, plotIt=False, random_seed=664)


@pytest.mark.parametrize("prop_class", RECIPROCALS)
@pytest.mark.parametrize("direction", [1, -1])
@pytest.mark.parametrize("inp_type", ["scalar", "array"])
def test_recip_assigned(prop_class, direction, inp_type):
    modeler = prop_class()
    params = inspect.signature(prop_class).parameters
    props = [param.name for param in params.values() if _param_prop_filter(param)][:2]

    if direction == 1:
        prop1, prop2 = props
    else:
        prop2, prop1 = props

    if inp_type == "scalar":
        inp = np.pi
        expected = 1 / np.pi
    else:  # inp_type == array
        inp = np.linspace(10, 12)
        expected = 1.0 / inp

    setattr(modeler, prop1, inp)
    calc = getattr(modeler, prop2)
    npt.assert_equal(calc, expected)


@pytest.mark.parametrize("prop_class", RECIPROCALS & INVERTIBLE)
@pytest.mark.parametrize("direction", [1, -1])
def test_recip_map_assign(prop_class, direction):
    modeler = prop_class()
    params = inspect.signature(prop_class).parameters
    props = [param.name for param in params.values() if _param_prop_filter(param)][:2]

    if direction == 1:
        prop1, prop2 = props
    else:
        prop2, prop1 = props

    inp = np.linspace(10, 12)
    expected = 1.0 / inp

    setattr(modeler, prop1 + "Map", maps.IdentityMap())
    modeler.model = inp
    calc = getattr(modeler, prop2)
    npt.assert_equal(calc, expected)


@pytest.mark.parametrize("prop_class", INVERTIBLE & RECIPROCALS)
@pytest.mark.parametrize("direction", [1, -1])
def test_recip_derivatives(prop_class, direction):
    modeler = prop_class()
    params = inspect.signature(prop_class).parameters
    props = [param.name for param in params.values() if _param_prop_filter(param)][:2]
    prop_map = maps.ExpMap()

    if direction == 1:
        prop1, prop2 = props
    else:
        prop2, prop1 = props
    inp = np.linspace(10, 12)

    setattr(modeler, prop1 + "Map", prop_map)

    def deriv_test(x):
        modeler.model = x
        out = getattr(modeler, prop2)

        map_deriv = getattr(modeler, prop2 + "Deriv")
        return out, lambda u: map_deriv @ u

    check_derivative(deriv_test, x0=inp, plotIt=False, random_seed=5523)


def test_permittivity_warning():
    msg = re.escape(
        "Simulations using permittivity have not yet been thoroughly tested and derivatives are not implemented. Contributions welcome!"
    )
    # assert it warns if not None
    with pytest.warns(UserWarning, match=msg):
        DielectricPermittivity(permittivity=0.1)

    # assert it doesn't warn if None
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        DielectricPermittivity(permittivity=None)
