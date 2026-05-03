import numpy as np
import pytest
import discretize
from simpeg import (
    maps,
    data_misfit,
    regularization,
    inversion,
    optimization,
    inverse_problem,
    tests,
    utils,
)
from simpeg.utils import mkvc
from simpeg.electromagnetics import resistivity as dc
from simpeg.electromagnetics.static import utils as static_utils
import shutil


REL_TOL = 1e-5
ABS_TOL = 1e-20


# =========================
# Fixtures
# =========================


@pytest.fixture
def base_mesh():
    aSpacing = 2.5
    nElecs = 5
    surveySize = nElecs * aSpacing - aSpacing
    cs = surveySize / nElecs / 4

    mesh = discretize.TensorMesh(
        [
            [(cs, 10, -1.3), (cs, surveySize / cs), (cs, 10, 1.3)],
            [(cs, 3, -1.3), (cs, 3, 1.3)],
        ],
        "CN",
    )
    return mesh, aSpacing, nElecs, surveySize


@pytest.fixture(params=["dipole-pole", "pole-dipole", "pole-pole"])
def survey_type(request):
    return request.param


@pytest.fixture(params=["volt", "apparent_resistivity"])
def data_type(request):
    return request.param


@pytest.fixture
def survey(base_mesh, survey_type, data_type):
    mesh, aSpacing, nElecs, surveySize = base_mesh

    source_list = dc.utils.WennerSrcList(nElecs, aSpacing, in2D=True)
    assert len(source_list) > 0

    next_list = static_utils.generate_dcip_sources_line(
        survey_type=survey_type,
        data_type=data_type,
        dimension_type="2D",
        end_points=[-surveySize / 2, surveySize / 2],
        num_rx_per_src=5,
        station_spacing=aSpacing,
        topo=0.0,
    )
    assert len(next_list) > 0

    source_list.extend(next_list)

    survey = dc.survey.Survey(source_list)
    survey.set_geometric_factor()
    return survey


@pytest.fixture(params=["cell_centered", "nodal"])
def simulation(request, base_mesh, survey):
    mesh, *_ = base_mesh

    if request.param == "cell_centered":
        sim = dc.simulation.Simulation3DCellCentered(
            mesh=mesh, survey=survey, rhoMap=maps.IdentityMap(mesh)
        )
    else:
        sim = dc.simulation.Simulation3DNodal(
            mesh=mesh, survey=survey, rhoMap=maps.IdentityMap(mesh)
        )

    return sim


@pytest.fixture
def inverse_setup(simulation, base_mesh):
    mesh, *_ = base_mesh

    m0 = np.ones(mesh.nC)
    dobs = simulation.make_synthetic_data(m0, add_noise=True, random_seed=39)

    dmis = data_misfit.L2DataMisfit(simulation=simulation, data=dobs)
    reg = regularization.WeightedLeastSquares(mesh)

    opt = optimization.InexactGaussNewton(maxIterLS=20, maxIter=10, cg_maxiter=6)

    invProb = inverse_problem.BaseInvProblem(dmis, reg, opt, beta=1e4)
    inv = inversion.BaseInversion(invProb)

    return {
        "simulation": simulation,
        "mesh": mesh,
        "m0": m0,
        "dobs": dobs,
        "dmis": dmis,
        "inv": inv,
    }


# =========================
# Core tests
# =========================


def test_misfit(inverse_setup):
    sim = inverse_setup["simulation"]
    m0 = inverse_setup["m0"]

    assert tests.check_derivative(
        lambda m: [sim.dpred(m), lambda mx: sim.Jvec(m0, mx)],
        m0,
        plotIt=False,
        num=3,
        random_seed=40,
    )


def test_adjoint(inverse_setup):
    sim = inverse_setup["simulation"]
    mesh = inverse_setup["mesh"]
    m0 = inverse_setup["m0"]
    dobs = inverse_setup["dobs"]

    rng = np.random.default_rng(42)
    v = rng.uniform(size=mesh.nC)
    w = rng.uniform(size=mkvc(dobs).shape[0])

    wtJv = w.dot(sim.Jvec(m0, v))
    vtJtw = v.dot(sim.Jtvec(m0, w))

    assert np.abs(wtJv - vtJtw) < 1e-8


def test_dataObj(inverse_setup):
    dmis = inverse_setup["dmis"]
    m0 = inverse_setup["m0"]

    assert tests.check_derivative(
        lambda m: [dmis(m), dmis.deriv(m)], m0, plotIt=False, num=3, random_seed=40
    )


# =========================
# Fields test (separate)
# =========================


@pytest.fixture
def fields_problem():
    cs = 10
    nc = 20
    npad = 10

    mesh = discretize.CylindricalMesh(
        [
            [(cs, nc), (cs, npad, 1.3)],
            np.r_[2 * np.pi],
            [(cs, npad, -1.3), (cs, nc), (cs, npad, 1.3)],
        ]
    )

    mesh.x0 = np.r_[0.0, 0.0, -mesh.h[2][: npad + nc].sum()]

    rx_x = np.linspace(10, 200, 20)
    rx_locs = utils.ndgrid([rx_x, np.r_[0], np.r_[-5]])

    rx_list = [dc.receivers.BaseRx(rx_locs, projField="e", orientation="x")]

    src = dc.sources.Dipole(
        rx_list,
        location_a=np.r_[0.0, 0.0, -5.0],
        location_b=np.r_[55.0, 0.0, -5.0],
    )

    survey = dc.survey.Survey([src])

    sigma_map = maps.ExpMap(mesh) * maps.InjectActiveCells(
        mesh, mesh.gridCC[:, 2] <= 0, np.log(1e-8)
    )

    prob = dc.simulation.Simulation3DCellCentered(
        mesh=mesh,
        survey=survey,
        sigmaMap=sigma_map,
        bc_type="Dirichlet",
    )

    return prob, sigma_map


def test_fields_derivative(fields_problem):
    prob, sigma_map = fields_problem

    rng = np.random.default_rng(40)
    x0 = -1 + 1e-1 * rng.uniform(size=sigma_map.nP)

    def fun(x):
        return prob.dpred(x), lambda x: prob.Jvec(x0, x)

    assert tests.check_derivative(fun, x0, num=3, plotIt=False, random_seed=40)


def test_fields_adjoint(fields_problem):
    prob, sigma_map = fields_problem

    rng = np.random.default_rng(42)
    m = -1 + 1e-1 * rng.uniform(size=sigma_map.nP)

    v = rng.uniform(size=prob.survey.nD)
    w = rng.uniform(size=sigma_map.nP)

    u = prob.fields(m)

    vJw = v.dot(prob.Jvec(m, w, u))
    wJtv = w.dot(prob.Jtvec(m, v, u))

    np.testing.assert_allclose(vJw, wJtv, atol=ABS_TOL, rtol=REL_TOL)

    # tol = np.max([REL_TOL * (10 ** int(np.log10(np.abs(vJw)))), ABS_TOL])
    # assert np.abs(vJw - wJtv) < tol


# =========================
# storeJ cleanup test
# =========================


@pytest.fixture
def storeJ_simulation(base_mesh, survey):
    mesh, *_ = base_mesh

    sim = dc.simulation.Simulation3DCellCentered(
        mesh=mesh,
        survey=survey,
        rhoMap=maps.IdentityMap(mesh),
        storeJ=True,
    )
    yield sim

    try:
        shutil.rmtree(sim.sensitivity_path)
    except FileNotFoundError:
        pass


def test_storeJ_runs(storeJ_simulation):
    mesh = storeJ_simulation.mesh
    m0 = np.ones(mesh.nC)

    dobs = storeJ_simulation.make_synthetic_data(m0, add_noise=True)
    assert dobs is not None


# =========================
# hierarchical test
# =========================


def test_hierarchical():
    aSpacing = 2.5
    nElecs = 10
    surveySize = nElecs * aSpacing - aSpacing
    cs = surveySize / nElecs / 4

    mesh = discretize.TensorMesh(
        [
            [(cs, 10, -1.3), (cs, surveySize / cs), (cs, 10, 1.3)],
            [(cs, 10, -1.3), (cs, surveySize / cs), (cs, 10, 1.3)],
            [(cs, 5, -1.3), (cs, 10)],
        ],
        "CCN",
    )

    survey = dc.survey.Survey(dc.utils.WennerSrcList(nElecs, aSpacing, in2D=False))

    wire_map = maps.Wires(
        ("log_sigma", mesh.n_cells),
        ("log_tau", mesh.n_faces),
        ("log_kappa", mesh.n_edges),
    )

    sim = dc.simulation.Simulation3DHierarchicalNodal(
        mesh=mesh,
        survey=survey,
        sigmaMap=maps.ExpMap(nP=mesh.n_cells) * wire_map.log_sigma,
        tauMap=maps.ExpMap(nP=mesh.n_faces) * wire_map.log_tau,
        kappaMap=maps.ExpMap(nP=mesh.n_edges) * wire_map.log_kappa,
        storeJ=True,
    )

    n_params = mesh.n_cells + mesh.n_faces + mesh.n_edges
    m0 = np.ones(n_params)

    dobs = sim.make_synthetic_data(m0, add_noise=True)

    rng = np.random.default_rng(42)
    v = rng.uniform(size=n_params)
    w = rng.uniform(size=mkvc(dobs).shape[0])

    wtJv = w.dot(sim.Jvec(m0, v))
    vtJtw = v.dot(sim.Jtvec(m0, w))

    assert np.abs(wtJv - vtJtw) < 1e-8

    try:
        shutil.rmtree(sim.sensitivity_path)
    except FileNotFoundError:
        pass
