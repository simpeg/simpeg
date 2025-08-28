import re
import pytest
import discretize
import simpeg.potential_fields as PF
from simpeg import utils, maps
from discretize.utils import mkvc, refine_tree_xyz
import numpy as np
from tests.utils.ellipsoid import ProlateEllipsoid


@pytest.fixture
def mesh():

    dhx, dhy, dhz = 50.0, 50.0, 50.0  # minimum cell width (base mesh cell width)
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


def get_survey(components=("bx", "by", "bz")):
    ccx = np.linspace(-1400, 1400, num=57)
    ccy = np.linspace(-1400, 1400, num=57)
    ccx, ccy = np.meshgrid(ccx, ccy)
    ccz = 50.0 * np.ones_like(ccx)
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


@pytest.mark.parametrize("model_type", ("mu_rem", "mu", "rem"))
def test_forward(model_type, mesh):
    """
    Test against the analytic solution for an ellipse with
    uniform intrinsic remanence and susceptibility in a
    uniform ambient geomagnetic field
    """
    tol = 0.1

    survey = get_survey()

    amplitude = survey.source_field.amplitude
    inclination = survey.source_field.inclination
    declination = survey.source_field.declination
    inducing_field = [amplitude, inclination, declination]

    if model_type == "mu_rem":
        susceptibility = 5
        MrX = 150000
        MrY = 150000
        MrZ = 150000
    if model_type == "mu":
        susceptibility = 5
        MrX = 0
        MrY = 0
        MrZ = 0
    if model_type == "rem":
        susceptibility = 0
        MrX = 150000
        MrY = 150000
        MrZ = 150000

    center = np.array([00, 0, -400.0])
    axes = [600.0, 200.0]
    strike_dip_rake = [0, 0, 90]

    ellipsoid = ProlateEllipsoid(
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
    mu_model = maps.ChiMap() * sus_model

    Rx = np.zeros(mesh.n_cells)
    Ry = np.zeros(mesh.n_cells)
    Rz = np.zeros(mesh.n_cells)

    Rx[ind_ellipsoid] = MrX
    Ry[ind_ellipsoid] = MrY
    Rz[ind_ellipsoid] = MrZ

    u0_Mr_model = mkvc(np.array([Rx, Ry, Rz]).T)

    if model_type == "mu":
        u0_Mr_model = None
    if model_type == "rem":
        mu_model = None

    simulation = PF.magnetics.simulation.Simulation3DDifferential(
        survey=survey, mesh=mesh, mu=mu_model, rem=u0_Mr_model, solver_dtype=np.float32
    )

    dpred_numeric = simulation.dpred()
    dpred_analytic = mkvc(ellipsoid.anomalous_bfield(survey.receiver_locations))

    assert np.allclose(
        dpred_numeric,
        dpred_analytic,
        rtol=0.1,
        atol=0.05 * np.max(np.abs(dpred_analytic)),
    )

    err = np.linalg.norm(dpred_numeric - dpred_analytic) / np.linalg.norm(
        dpred_analytic
    )

    print(
        "\n||dpred_analytic-dpred_numeric||/||dpred_analytic|| = "
        + "{:.{}f}".format(err, 2)
        + ", tol = "
        + str(tol)
    )

    assert err < tol

    u0_M_analytic = ellipsoid.Magnetization()
    u0_M_numeric = mesh.average_face_to_cell_vector * simulation.magnetic_polarization()
    u0_M_numeric = u0_M_numeric.reshape((mesh.n_cells, 3), order="F")
    u0_M_numeric = np.mean(u0_M_numeric[ind_ellipsoid, :], axis=0)

    assert np.allclose(
        u0_M_numeric,
        u0_M_analytic,
        rtol=0.1,
        atol=0.01 * np.max(np.abs(u0_M_analytic)),
    )


def test_exact_tmi(mesh):
    """
    Test against the analytic solution for an ellipse with
    uniform intrinsic remanence and susceptibility in a
    uniform ambient geomagnetic field
    """
    tol = 1e-8

    survey = get_survey(components=["bx", "by", "bz", "tmi"])

    amplitude = survey.source_field.amplitude
    inclination = survey.source_field.inclination
    declination = survey.source_field.declination
    inducing_field = [amplitude, inclination, declination]

    susceptibility = 5
    MrX = 150000
    MrY = 150000
    MrZ = 150000

    center = np.array([00, 0, -400.0])
    axes = [600.0, 200.0]
    strike_dip_rake = [0, 0, 90]

    ellipsoid = ProlateEllipsoid(
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
    mu_model = maps.ChiMap() * sus_model

    Rx = np.zeros(mesh.n_cells)
    Ry = np.zeros(mesh.n_cells)
    Rz = np.zeros(mesh.n_cells)

    Rx[ind_ellipsoid] = MrX
    Ry[ind_ellipsoid] = MrY
    Rz[ind_ellipsoid] = MrZ

    u0_Mr_model = mkvc(np.array([Rx, Ry, Rz]).T)

    simulation = PF.magnetics.simulation.Simulation3DDifferential(
        survey=survey,
        mesh=mesh,
        mu=mu_model,
        rem=u0_Mr_model,
    )

    dpred_numeric = simulation.dpred()

    dpred_fields = np.reshape(dpred_numeric[: survey.nRx * 4], (4, survey.nRx)).T

    B0 = survey.source_field.b0

    TMI_exact_analytic = np.linalg.norm(dpred_fields[:, :3] + B0, axis=1) - amplitude
    dpred_TMI_exact = dpred_fields[:, 3]

    TMI_exact_err = np.max(np.abs(dpred_TMI_exact - TMI_exact_analytic))

    assert TMI_exact_err < tol
    print(
        "max(TMI_exact_err) = "
        + "{:.{}e}".format(TMI_exact_err, 2)
        + ", tol = "
        + str(tol)
    )


def test_differential_magnetization_against_integral(mesh):

    survey = get_survey()

    amplitude = survey.source_field.amplitude
    inclination = survey.source_field.inclination
    declination = survey.source_field.declination
    inducing_field = [amplitude, inclination, declination]

    MrX = 150000
    MrY = 150000
    MrZ = 150000

    center = np.array([00, 0, -400.0])
    axes = [600.0, 200.0]
    strike_dip_rake = [0, 0, 90]

    ellipsoid = ProlateEllipsoid(
        center,
        axes,
        strike_dip_rake,
        Mr=np.array([MrX, MrY, MrZ]),
        inducing_field=inducing_field,
    )
    ind_ellipsoid = ellipsoid.get_indices(mesh.cell_centers)

    Rx = np.zeros(mesh.n_cells)
    Ry = np.zeros(mesh.n_cells)
    Rz = np.zeros(mesh.n_cells)

    Rx[ind_ellipsoid] = MrX
    Ry[ind_ellipsoid] = MrY
    Rz[ind_ellipsoid] = MrZ

    u0_Mr_model = mkvc(np.array([Rx, Ry, Rz]).T)
    eff_sus_model = (u0_Mr_model / amplitude)[
        np.hstack((ind_ellipsoid, ind_ellipsoid, ind_ellipsoid))
    ]

    simulation_differential = PF.magnetics.simulation.Simulation3DDifferential(
        survey=survey,
        mesh=mesh,
        rem=u0_Mr_model,
    )

    simulation_integral = PF.magnetics.simulation.Simulation3DIntegral(
        survey=survey,
        mesh=mesh,
        chi=eff_sus_model,
        model_type="vector",
        store_sensitivities="forward_only",
        active_cells=ind_ellipsoid,
    )

    dpred_numeric_differential = simulation_differential.dpred()
    dni = simulation_integral.dpred()
    dpred_numeric_integral = np.hstack((dni[0::3], dni[1::3], dni[2::3]))
    dpred_analytic = mkvc(ellipsoid.anomalous_bfield(survey.receiver_locations))

    diff_numeric = np.linalg.norm(
        dpred_numeric_differential - dpred_numeric_integral
    ) / np.linalg.norm(dpred_numeric_integral)
    diff_differential = np.linalg.norm(
        dpred_numeric_differential - dpred_analytic
    ) / np.linalg.norm(dpred_analytic)
    diff_integral = np.linalg.norm(
        dpred_numeric_integral - dpred_analytic
    ) / np.linalg.norm(dpred_analytic)

    # Check both discretized solutions are closer to each other than to the analytic
    assert diff_numeric < diff_differential
    assert diff_numeric < diff_integral

    print(
        "\n||dpred_integral-dpred_pde||/||dpred_integral|| = "
        + "{:.{}f}".format(diff_numeric, 2)
    )
    print(
        "||dpred_integral-dpred_analytic||/||dpred_analytic|| = "
        + "{:.{}f}".format(diff_integral, 2)
    )
    print(
        "||dpred_pde-dpred_analytic||/||dpred_analytic|| = "
        + "{:.{}f}".format(diff_differential, 2)
    )


def test_invalid_solver_dtype(mesh):
    """
    Test error upon invalid `solver_dtype`.
    """
    survey = get_survey()
    invalid_dtype = np.int64
    msg = re.escape(
        f"Invalid `solver_dtype` '{invalid_dtype}'. "
        "It must be np.float32 or np.float64."
    )
    with pytest.raises(ValueError, match=msg):
        PF.magnetics.simulation.Simulation3DDifferential(
            survey=survey, mesh=mesh, solver_dtype=invalid_dtype
        )
