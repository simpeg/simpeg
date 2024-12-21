from simpeg import maps
from simpeg.base import BaseElectricalPDESimulation, BaseMagneticPDESimulation
import pytest
import numpy as np
import discretize


MASSMATS_FORMAT_STR = [
    ("Mcc{arg}", "Mcc{arg}Deriv"),
    ("Mn{arg}", "Mn{arg}Deriv"),
    ("Mf{arg}", "Mf{arg}Deriv"),
    ("Me{arg}", "Me{arg}Deriv"),
    ("Mcc{arg}I", "Mcc{arg}IDeriv"),
    ("Mn{arg}I", "Mn{arg}IDeriv"),
    ("Mf{arg}I", "Mf{arg}IDeriv"),
    ("Me{arg}I", "Me{arg}IDeriv"),
]


@pytest.mark.parametrize(
    "base_class", [BaseMagneticPDESimulation, BaseElectricalPDESimulation]
)
@pytest.mark.parametrize("mm_and_func_fmt_str", MASSMATS_FORMAT_STR)
@pytest.mark.parametrize("map_prop", [0, 1])
@pytest.mark.parametrize("deriv_prop", [0, 1])
@pytest.mark.parametrize("dimension", [1, 2, 3])
@pytest.mark.parametrize("deprecated", [True, False])
def test_mass_matrix_derivatives(
    base_class, mm_and_func_fmt_str, map_prop, deriv_prop, dimension, deprecated
):
    prop_names = list(base_class._physical_properties.keys())
    map_prop = prop_names[map_prop]
    hx = np.full(13, 10.0)
    hy = np.full(12, 10.0)
    hz = np.full(11, 10.0)
    h = [hx, hy, hz][:dimension]
    mesh = discretize.TensorMesh(h)
    sim = base_class(mesh)
    if deprecated:
        setattr(sim, map_prop + "Map", maps.ExpMap())
    else:
        setattr(sim, map_prop, maps.ExpMap())

    x0 = np.linspace(-1, 1, mesh.n_cells)
    sim.model = x0

    deriv_prop = prop_names[deriv_prop]
    deriv_prop = deriv_prop[0].upper() + deriv_prop[1:]
    mm_attr, mm_func = mm_and_func_fmt_str
    mm_attr = mm_attr.format(arg=deriv_prop)
    deriv_func = getattr(sim, mm_func.format(arg=deriv_prop))

    M = getattr(sim, mm_attr)
    u = np.linspace(-12, 31, M.shape[0])

    def mm_test_func(mod):
        sim.model = mod
        M = getattr(sim, mm_attr)

        def Jvec(v):
            return deriv_func(u, v)

        return M @ u, Jvec

    discretize.tests.check_derivative(mm_test_func, x0, num=4, random_seed=412)


@pytest.mark.parametrize(
    "base_class", [BaseMagneticPDESimulation, BaseElectricalPDESimulation]
)
@pytest.mark.parametrize("mm_and_func_fmt_str", MASSMATS_FORMAT_STR)
@pytest.mark.parametrize("prop", [0, 1])
@pytest.mark.parametrize("dimension", [1, 2, 3])
def test_mass_matrix_adjoints(base_class, mm_and_func_fmt_str, prop, dimension):
    prop_names = list(base_class._physical_properties.keys())
    map_prop = prop_names[prop]
    hx = np.full(13, 10.0)
    hy = np.full(12, 10.0)
    hz = np.full(11, 10.0)
    h = [hx, hy, hz][:dimension]
    mesh = discretize.TensorMesh(h)
    sim = base_class(mesh)
    setattr(sim, map_prop, maps.ExpMap())

    x0 = np.linspace(-1, 1, mesh.n_cells)
    sim.model = x0

    deriv_prop = map_prop[0].upper() + map_prop[1:]
    mm_attr, mm_func = mm_and_func_fmt_str
    mm_attr = mm_attr.format(arg=deriv_prop)
    deriv_func = getattr(sim, mm_func.format(arg=deriv_prop))

    M = getattr(sim, mm_attr)
    y = np.linspace(-12, 31, M.shape[0])

    def mm_func(u):
        return deriv_func(y, u, adjoint=False)

    def mm_func_adj(v):
        return deriv_func(y, v, adjoint=True)

    shape_u = (mesh.n_cells,)
    shape_v = (M.shape[0],)

    discretize.tests.assert_isadjoint(
        mm_func, mm_func_adj, shape_u, shape_v, random_seed=552
    )
