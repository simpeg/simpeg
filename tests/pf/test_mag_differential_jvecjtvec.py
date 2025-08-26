from discretize.tests import check_derivative, assert_isadjoint
import numpy as np
import pytest
from simpeg import maps, utils
from discretize.utils import mkvc, refine_tree_xyz
import discretize
import simpeg.potential_fields as PF


@pytest.fixture
def mesh():
    dhx, dhy, dhz = 400.0, 400.0, 400.0  # minimum cell width (base mesh cell width)
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


@pytest.fixture
def survey():
    ccx = np.linspace(-1400, 1400, num=57)
    ccy = np.copy(ccx)

    ccx, ccy = np.meshgrid(ccx, ccy)

    ccz = 50.0 * np.ones_like(ccx)

    components = ["bx", "by", "bz", "tmi"]
    rxLoc = PF.magnetics.receivers.Point(
        np.c_[utils.mkvc(ccy.T), utils.mkvc(ccx.T), utils.mkvc(ccz.T)],
        components=components,
    )
    inducing_field = [55000.0, 60.0, 90.0]
    srcField = PF.magnetics.sources.UniformBackgroundField(
        [rxLoc], inducing_field[0], inducing_field[1], inducing_field[2]
    )
    _survey = PF.magnetics.survey.Survey(srcField)

    return _survey


@pytest.mark.parametrize(
    "deriv_type", ("mu", "rem", "mu_fix_rem", "rem_fix_mu", "both")
)
def test_derivative(deriv_type, mesh, survey):
    np.random.seed(40)

    chimap = maps.ChiMap(mesh)
    eff_sus_map = maps.EffectiveSusceptibilityMap(
        ambient_field_magnitude=survey.source_field.amplitude, nP=mesh.n_cells * 3
    )

    sus_model = np.abs(np.random.randn(mesh.n_cells))
    mu_model = chimap * sus_model

    Rx = np.random.randn(mesh.n_cells)
    Ry = np.random.randn(mesh.n_cells)
    Rz = np.random.randn(mesh.n_cells)
    EsusRem = mkvc(np.array([Rx, Ry, Rz]).T)

    u0_Mr_model = eff_sus_map * EsusRem

    if deriv_type == "mu":
        mu_map = chimap
        mu = None
        rem_map = None
        rem = None
        m = sus_model
    if deriv_type == "rem":
        mu_map = None
        mu = None
        rem_map = eff_sus_map
        rem = None
        m = EsusRem
    if deriv_type == "mu_fix_rem":
        mu_map = chimap
        mu = None
        rem_map = None
        rem = u0_Mr_model
        m = sus_model
    if deriv_type == "rem_fix_mu":
        mu_map = None
        mu = mu_model
        rem_map = eff_sus_map
        rem = None
        m = EsusRem
    if deriv_type == "both":
        wire_map = maps.Wires(("mu", mesh.n_cells), ("rem", mesh.n_cells * 3))
        mu_map = chimap * wire_map.mu
        rem_map = eff_sus_map * wire_map.rem
        m = np.r_[sus_model, EsusRem]
        mu = None
        rem = None

    simulation = PF.magnetics.simulation.Simulation3DDifferential(
        survey=survey, mesh=mesh, mu=mu, rem=rem, muMap=mu_map, remMap=rem_map
    )

    def sim_func(m):
        d = simulation.dpred(m)

        def J(v):
            return simulation.Jvec(m, v)

        return d, J

    assert check_derivative(sim_func, m, plotIt=False, num=6, eps=1e-8, random_seed=40)


@pytest.mark.parametrize(
    "deriv_type", ("mu", "rem", "mu_fix_rem", "rem_fix_mu", "both")
)
def test_adjoint(deriv_type, mesh, survey):
    np.random.seed(40)

    chimap = maps.ChiMap(mesh)
    eff_sus_map = maps.EffectiveSusceptibilityMap(
        ambient_field_magnitude=survey.source_field.amplitude, nP=mesh.n_cells * 3
    )

    sus_model = np.abs(np.random.randn(mesh.n_cells))
    mu_model = chimap * sus_model

    Rx = np.random.randn(mesh.n_cells)
    Ry = np.random.randn(mesh.n_cells)
    Rz = np.random.randn(mesh.n_cells)
    EsusRem = mkvc(np.array([Rx, Ry, Rz]).T)

    u0_Mr_model = eff_sus_map * EsusRem

    if deriv_type == "mu":
        mu_map = chimap
        mu = None
        rem_map = None
        rem = None
        m = sus_model
    if deriv_type == "rem":
        mu_map = None
        mu = None
        rem_map = eff_sus_map
        rem = None
        m = EsusRem
    if deriv_type == "mu_fix_rem":
        mu_map = chimap
        mu = None
        rem_map = None
        rem = u0_Mr_model
        m = sus_model
    if deriv_type == "rem_fix_mu":
        mu_map = None
        mu = mu_model
        rem_map = eff_sus_map
        rem = None
        m = EsusRem
    if deriv_type == "both":
        wire_map = maps.Wires(("mu", mesh.n_cells), ("rem", mesh.n_cells * 3))
        mu_map = chimap * wire_map.mu
        rem_map = eff_sus_map * wire_map.rem
        m = np.r_[sus_model, EsusRem]
        mu = None
        rem = None

    simulation = PF.magnetics.simulation.Simulation3DDifferential(
        survey=survey, mesh=mesh, mu=mu, rem=rem, muMap=mu_map, remMap=rem_map
    )

    def J(v):
        return simulation.Jvec(m, v)

    def JT(v):
        return simulation.Jtvec(m, v)

    assert_isadjoint(J, JT, len(m), survey.nD, random_seed=40)
