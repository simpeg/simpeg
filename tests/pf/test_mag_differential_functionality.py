import pytest
import discretize
import simpeg.potential_fields as PF
from simpeg import utils, maps
from discretize.utils import mkvc, refine_tree_xyz
import numpy as np
from tests.utils.shared_helpers import ProlateEllispse


@pytest.fixture
def mesh():

    dhx, dhy, dhz = 75.0, 75.0, 75.0  # minimum cell width (base mesh cell width)
    nbcx = 512  # number of base mesh cells in x
    nbcy = 512
    nbcz = 512

    # Define base mesh (domain and finest discretization)
    hx = dhx * np.ones(nbcx)
    hy = dhy * np.ones(nbcy)
    hz = dhz * np.ones(nbcz)
    _mesh = discretize.TreeMesh([hx, hy, hz], x0="CCC")

    xp, yp, zp = np.meshgrid([-1400.0, 1400.0], [-1400.0, 1400.0], [-1000.0, 200.0])
    xy = np.c_[mkvc(xp), mkvc(yp), mkvc(zp)]
    _mesh = refine_tree_xyz(
        _mesh,
        xy,
        method="box",
        finalize=False,
        octree_levels=[1, 1, 1, 1],
    )
    _mesh.finalize()

    return _mesh


def test_recievers(mesh):
    """
    Test that multiple point recievers with different components work.
    """

    ccx = np.linspace(-1400, 1400, num=57)
    ccy = np.linspace(-1400, 1400, num=57)
    ccx, ccy = np.meshgrid(ccx, ccy)
    ccz = 50.0 * np.ones_like(ccx)
    components_1 = ["bx", "by", "bz", "tmi"]
    components_2 = ["by", "tmi"]
    rxLoc_1 = PF.magnetics.receivers.Point(
        np.c_[utils.mkvc(ccy.T), utils.mkvc(ccx.T), utils.mkvc(ccz.T)],
        components=components_1,
    )
    rxLoc_2 = PF.magnetics.receivers.Point(
        np.c_[utils.mkvc(ccy.T), utils.mkvc(ccx.T), utils.mkvc(ccz.T + 20)],
        components=components_2,
    )
    inducing_field = [55000.0, 60.0, 90.0]

    srcField_1 = PF.magnetics.sources.UniformBackgroundField(
        [rxLoc_1], inducing_field[0], inducing_field[1], inducing_field[2]
    )
    survey_1 = PF.magnetics.survey.Survey(srcField_1)

    srcField_2 = PF.magnetics.sources.UniformBackgroundField(
        [rxLoc_2], inducing_field[0], inducing_field[1], inducing_field[2]
    )
    survey_2 = PF.magnetics.survey.Survey(srcField_2)

    srcField_all = PF.magnetics.sources.UniformBackgroundField(
        [rxLoc_1, rxLoc_2], inducing_field[0], inducing_field[1], inducing_field[2]
    )
    survey_all = PF.magnetics.survey.Survey(srcField_all)

    amplitude = survey_1.source_field.amplitude
    inclination = survey_1.source_field.inclination
    declination = survey_1.source_field.declination
    inducing_field = [amplitude, inclination, declination]

    susceptibility = 5
    MrX = 150000
    MrY = 150000
    MrZ = 150000

    center = np.array([00, 0, -400.0])
    axes = [600.0, 200.0]
    strike_dip_rake = [0, 0, 90]

    ellipsoid = ProlateEllispse(
        center,
        axes,
        strike_dip_rake,
        susceptibility=susceptibility,
        Mr=(MrX, MrY, MrZ),
        inducing_field=inducing_field,
    )

    ind_ellipsoid = ellipsoid.get_indices(mesh.cell_centers)

    sus_model = np.zeros(mesh.n_cells)
    sus_model[ind_ellipsoid] = susceptibility

    Rx = np.zeros(mesh.n_cells)
    Ry = np.zeros(mesh.n_cells)
    Rz = np.zeros(mesh.n_cells)

    Rx[ind_ellipsoid] = MrX / 55000
    Ry[ind_ellipsoid] = MrY / 55000
    Rz[ind_ellipsoid] = MrZ / 55000

    EsusRem = mkvc(np.array([Rx, Ry, Rz]).T)

    chimap = maps.ChiMap(mesh)
    eff_sus_map = maps.EffectiveSusceptibilityMap(
        nP=mesh.n_cells * 3, ambient_field_magnitude=survey_1.source_field.amplitude
    )

    wire_map = maps.Wires(("mu", mesh.n_cells), ("rem", mesh.n_cells * 3))
    mu_map = chimap * wire_map.mu
    rem_map = eff_sus_map * wire_map.rem
    m = np.r_[sus_model, EsusRem]

    simulation_1 = PF.magnetics.simulation.Simulation3DDifferential(
        mesh=mesh,
        survey=survey_1,
        muMap=mu_map,
        remMap=rem_map,
    )

    simulation_2 = PF.magnetics.simulation.Simulation3DDifferential(
        survey=survey_2,
        mesh=mesh,
        muMap=mu_map,
        remMap=rem_map,
    )

    simulation_all = PF.magnetics.simulation.Simulation3DDifferential(
        survey=survey_all,
        mesh=mesh,
        muMap=mu_map,
        remMap=rem_map,
    )
    dpred_numeric_all = simulation_all.dpred(m)
    dpred_numeric_1 = simulation_1.dpred(m)
    dpred_numeric_2 = simulation_2.dpred(m)
    dpred_stack = np.hstack((dpred_numeric_1, dpred_numeric_2))

    rvec = np.random.randn(mesh.n_cells * 4) * 0.001

    jv_all = simulation_all.Jvec(m, v=rvec)
    jv_1 = simulation_1.Jvec(m, v=rvec)
    jv_2 = simulation_2.Jvec(m, v=rvec)
    jv_stack = np.hstack((jv_1, jv_2))

    assert np.allclose(dpred_numeric_all, dpred_stack, atol=1e-8)
    assert np.allclose(jv_all, jv_stack, atol=1e-8)


def test_component_validation(mesh):
    """
    Test that invalid reciever types raise an error.
    """

    ccx = np.linspace(-1400, 1400, num=57)
    ccy = np.linspace(-1400, 1400, num=57)
    ccx, ccy = np.meshgrid(ccx, ccy)
    ccz = 50.0 * np.ones_like(ccx)

    components_invalid = ["bx", "by", "bz", "bxx", "byy", "bzz", "tmi"]

    rxLoc = PF.magnetics.receivers.Point(
        np.c_[utils.mkvc(ccy.T), utils.mkvc(ccx.T), utils.mkvc(ccz.T)],
        components=components_invalid,
    )

    inducing_field = [55000.0, 60.0, 90.0]

    srcField = PF.magnetics.sources.UniformBackgroundField(
        [rxLoc], inducing_field[0], inducing_field[1], inducing_field[2]
    )
    survey = PF.magnetics.survey.Survey(srcField)

    chimap = maps.ChiMap(mesh)
    eff_sus_map = maps.EffectiveSusceptibilityMap(
        nP=mesh.n_cells * 3, ambient_field_magnitude=survey.source_field.amplitude
    )

    wire_map = maps.Wires(("mu", mesh.n_cells), ("rem", mesh.n_cells * 3))
    mu_map = chimap * wire_map.mu
    rem_map = eff_sus_map * wire_map.rem

    with pytest.raises(
        ValueError, match="does not currently support the following components"
    ):
        _ = PF.magnetics.simulation.Simulation3DDifferential(
            mesh=mesh,
            survey=survey,
            muMap=mu_map,
            remMap=rem_map,
        )
