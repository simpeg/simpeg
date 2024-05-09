import pytest
import numpy as np
import simpeg.electromagnetics.static.self_potential as sp
import simpeg.electromagnetics.static.resistivity as dc
import discretize
from simpeg import utils
from simpeg import maps
from discretize.tests import check_derivative, assert_isadjoint


# setup simulation
mesh = discretize.TensorMesh([10, 11, 12], "CCN")
conductivity = 0.01
base_elec = [-0.4, -0.4, -0.3]
xyz_roving = utils.ndgrid(
    mesh.cell_centers_x[2:-2], mesh.cell_centers_y[2:-2], np.r_[-0.3]
)
xyz_base = np.tile([base_elec], (xyz_roving.shape[0], 1))
rx_dipole = dc.receivers.Dipole(locations_m=xyz_roving, locations_n=xyz_base)
rx_pole = dc.receivers.Pole(locations=np.r_[xyz_roving, [base_elec]])
src = sp.sources.StreamingCurrents([rx_dipole, rx_pole])
survey = sp.Survey([src])
sim = sp.Simulation3DCellCentered(mesh=mesh, survey=survey, sigma=conductivity)


def test_forward():
    # double check qMap is maps.IdentityMap()
    sim.qMap = maps.IdentityMap()
    # We can setup a dc simulation with a dipole source at these
    # two locations to double check everything evaluated correctly.
    q = np.zeros(mesh.nC)
    a_loc = np.r_[-0.5, 0.0, -0.8]
    b_loc = np.r_[0.5, 0.0, -0.8]
    inda = mesh.closest_points_index(a_loc)
    indb = mesh.closest_points_index(b_loc)
    q[inda] = 1.0
    q[indb] = -1.0
    q /= mesh.cell_volumes

    dc_tx = dc.sources.Dipole([rx_dipole, rx_pole], location_a=a_loc, location_b=b_loc)
    dc_survey = dc.Survey([dc_tx])
    sim_dc = dc.Simulation3DCellCentered(
        mesh=mesh, survey=dc_survey, sigma=conductivity
    )

    dc_dpred = sim_dc.make_synthetic_data(None, add_noise=False)
    sp_dpred = sim.make_synthetic_data(q, add_noise=False)

    np.testing.assert_allclose(dc_dpred.dobs, sp_dpred.dobs)


@pytest.mark.parametrize(
    "q_map",
    [
        maps.IdentityMap(mesh),
        sp.CurrentDensityMap(mesh),
        sp.CurrentDensityMap(mesh, active_cells=mesh.cell_centers[:, -1] < 0.85),
        sp.HydraulicHeadMap(mesh, L=1.0),
    ],
)
def test_deriv(q_map):
    sim.model = None
    sim.qMap = q_map

    def func(m):
        f = sim.fields(m)
        d = sim.dpred(m, f=f)

        def Jvec(v):
            return sim.Jvec(m, v, f=f)

        return d, Jvec

    m0 = np.random.randn(q_map.shape[1])
    check_derivative(func, m0, plotIt=False)


@pytest.mark.parametrize(
    "q_map",
    [
        maps.IdentityMap(mesh),
        sp.CurrentDensityMap(mesh),
        sp.HydraulicHeadMap(mesh, L=1.0),
    ],
)
def test_adjoint(q_map):
    sim.model = None
    sim.qMap = q_map

    model = np.random.rand(q_map.shape[1])
    f = sim.fields(model)

    def Jvec(v):
        return sim.Jvec(model, v, f=f)

    def Jtvec(v):
        return sim.Jtvec(model, v, f=f)

    assert_isadjoint(Jvec, Jtvec, shape_u=(q_map.shape[1],), shape_v=(survey.nD))


def test_errors():
    with pytest.raises(ValueError):
        sp.Simulation3DCellCentered(mesh=mesh, survey=survey, sigma=None, rho=None)
    with pytest.raises(ValueError):
        sp.Simulation3DCellCentered(mesh=mesh, survey=survey, sigma=1.0, rho=1.0)


def test_clears():
    # set qMap as a non-linear map to make sure it adds the correct
    # items to be cleared on model update
    sim.qMap = maps.IdentityMap()
    assert sim.deleteTheseOnModelUpdate == []
    assert sim.clean_on_model_update == []

    sim.storeJ = True
    sim.qMap = maps.ExpMap()
    assert sim.deleteTheseOnModelUpdate == ["_Jmatrix", "_gtgdiag"]
    assert sim.clean_on_model_update == []


def test_deprecations():
    """
    Test warning after importing deprecated `spontaneous_potential` module
    """
    msg = (
        "The 'spontaneous_potential' module has been renamed to 'self_potential'. "
        "Please use the 'self_potential' module instead. "
        "The 'spontaneous_potential' module will be removed in SimPEG 0.23."
    )
    with pytest.warns(FutureWarning, match=msg):
        import simpeg.electromagnetics.static.spontaneous_potential  # noqa: F401


def test_imported_objects_on_deprecated_module():
    """
    Test if the new `self_potential` module and the deprecated `spontaneous
    potential` have the same members.
    """
    import simpeg.electromagnetics.static.spontaneous_potential as spontaneous

    members_self = set([m for m in dir(sp) if not m.startswith("_")])
    members_spontaneous = set([m for m in dir(spontaneous) if not m.startswith("_")])
    difference = members_self - members_spontaneous
    assert not difference
