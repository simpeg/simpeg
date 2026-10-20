import numpy as np
import pytest

import discretize
from simpeg.electromagnetics import time_domain as tdem
from simpeg import maps

# solver
from simpeg.utils.solver_utils import get_default_solver

Solver = get_default_solver()

# conductivity values
rho_back = 500
sigma_air = 1e-8
sigma_back = 1 / rho_back

current = 2
REL_TOL = 0.35


def setup_mesh_model(tx_halfwidth=50):

    # design a tensor mesh
    cell_size = 20
    padding_factor = 1.5

    n_cells_x = int(tx_halfwidth * 2 / cell_size)
    n_cells_z = int((tx_halfwidth) / cell_size) + 5
    n_padding_x = 11
    n_padding_z = 11

    hx = [
        (cell_size, n_padding_x, -padding_factor),
        (cell_size, n_cells_x),
        (cell_size, n_padding_z, padding_factor),
    ]

    hz = [
        (cell_size, n_padding_z, -padding_factor),
        (cell_size, n_cells_z),
        (cell_size, n_padding_z, padding_factor),
    ]

    mesh = discretize.TensorMesh([hx, hx, hz], origin="CC0")
    mesh.origin = mesh.origin - np.r_[0, 0, mesh.h[2][: n_padding_z + n_cells_z].sum()]
    mesh.n_cells

    # define model
    model = sigma_air * np.ones(mesh.n_cells)
    model[mesh.cell_centers[:, 2] < 0] = sigma_back

    return mesh, model


def setup_survey(rx_locs, rx_times, tx_halfwidth=50, tx_z=0.5, waveform=None):
    # transmitter, by default slightly above the surface
    tx_points = np.array(
        [
            [-tx_halfwidth, -tx_halfwidth, tx_z],
            [tx_halfwidth, -tx_halfwidth, tx_z],
            [tx_halfwidth, tx_halfwidth, tx_z],
            [-tx_halfwidth, tx_halfwidth, tx_z],
            [-tx_halfwidth, -tx_halfwidth, tx_z],  # close the loop
        ]
    )

    # define survey for 3D simulation
    dbdt_receivers = [
        tdem.receivers.PointMagneticFluxTimeDerivative(
            locations=rx_locs, times=rx_times, orientation="z"
        )
    ]

    b_receivers = [
        tdem.receivers.PointMagneticFluxDensity(
            locations=rx_locs, times=rx_times, orientation=orientation
        )
        for orientation in ["z"]
    ]

    if waveform is None:
        waveform = tdem.sources.StepOffWaveform()

    src = tdem.sources.LineCurrent(
        receiver_list=b_receivers + dbdt_receivers,
        location=tx_points,
        waveform=waveform,
        srcType="inductive",
        current=current,
    )

    survey = tdem.Survey([src])

    return survey


def setup_simulation(mesh, survey, simulation_type="EB", on_time_steps=None):

    nsteps = 20
    if on_time_steps is None:
        on_time_steps = []
    time_steps = on_time_steps + [
        (3e-6, nsteps),
        (1e-5, nsteps),
        (3e-5, nsteps),
        (1e-4, nsteps),
        (3e-4, nsteps + 4),
    ]

    if simulation_type == "EB":
        simulation = tdem.simulation.Simulation3DMagneticFluxDensity(
            mesh=mesh,
            survey=survey,
            time_steps=time_steps,
            solver=Solver,
            sigmaMap=maps.IdentityMap(mesh),
        )
    elif simulation_type == "HJ":
        simulation = tdem.simulation.Simulation3DMagneticField(
            mesh=mesh,
            survey=survey,
            time_steps=time_steps,
            solver=Solver,
            sigmaMap=maps.IdentityMap(mesh),
        )
    return simulation


@pytest.mark.parametrize("simulation_type", ["EB", "HJ"])
def test_large_loop(simulation_type):

    tx_halfwidth = 50

    # receiver times
    rx_times = 1e-3 * np.logspace(-1, 1, 30)

    rx_x = np.r_[20]  # np.linspace(-100, 100, 10)
    rx_y = np.r_[20]  # np.linspace(-100, 100, 10)
    rx_z = np.r_[0]

    rx_locs = discretize.utils.ndgrid(rx_x, rx_y, rx_z)

    mesh, model = setup_mesh_model(tx_halfwidth=tx_halfwidth)
    survey = setup_survey(rx_locs=rx_locs, rx_times=rx_times, tx_halfwidth=tx_halfwidth)
    simulation = setup_simulation(mesh, survey, simulation_type)

    fields = simulation.fields(model)
    dpred_numeric = simulation.dpred(model, f=fields)

    # define 1D simulation
    dbdt_receivers1d = [
        tdem.receivers.PointMagneticFluxTimeDerivative(
            locations=rx_locs, times=rx_times, orientation="z"
        )
    ]

    b_receivers1d = [
        tdem.receivers.PointMagneticFluxDensity(
            locations=rx_locs, times=rx_times, orientation="z"
        )
    ]

    waveform = tdem.sources.StepOffWaveform()
    src1d = tdem.sources.LineCurrent(
        receiver_list=b_receivers1d + dbdt_receivers1d,
        location=survey.source_list[0].location,
        waveform=waveform,
        srcType="inductive",
    )

    survey1d = tdem.Survey([src1d])

    simulation_1D = tdem.simulation_1d.Simulation1DLayered(
        survey=survey1d, sigmaMap=maps.IdentityMap()
    )

    dpred1d = simulation_1D.dpred(sigma_back) * current

    np.testing.assert_allclose(dpred_numeric, dpred1d, rtol=REL_TOL)


@pytest.mark.parametrize("simulation_type", ["EB", "HJ"])
def test_large_loop_trapezoid(simulation_type):
    """
    Test the on-time source term of the loop.

    For a step-off waveform, the HJ initial fields of an inductive loop come
    from the Biot-Savart vector potential and the source term is zero after
    t=0, so the direction of the current on the mesh never enters the
    simulation. A trapezoid waveform has no initial fields, so the fields are
    driven by the source term during the on-time.
    """

    # The mesh has cell centers at +/-40 m.
    # The HJ source term moves the loop to the nearest cell centers,
    # so use a 40 m half-width at z = 15 m (also on cell centers) so the 3D
    # and 1D simulations model the same loop.
    tx_halfwidth = 40
    tx_z = 15

    off_time = 5e-4
    waveform = tdem.sources.TrapezoidWaveform(
        ramp_on=[0, 1e-4], ramp_off=[4e-4, off_time]
    )
    on_time_steps = [(2e-5, 20), (1e-5, 10)]

    # receiver times: late times are less accurate in the EB solve, so stop at 5 ms
    rx_times = off_time + 1e-3 * np.logspace(-1, np.log10(5), 20)

    rx_locs = np.array([[20, 20, 0]])

    mesh, model = setup_mesh_model()
    survey = setup_survey(
        rx_locs=rx_locs,
        rx_times=rx_times,
        tx_halfwidth=tx_halfwidth,
        tx_z=tx_z,
        waveform=waveform,
    )
    simulation = setup_simulation(
        mesh, survey, simulation_type, on_time_steps=on_time_steps
    )
    dpred_numeric = simulation.dpred(model)

    # 1D simulation with the same source
    simulation_1D = tdem.simulation_1d.Simulation1DLayered(
        survey=survey, sigmaMap=maps.IdentityMap()
    )
    dpred1d = simulation_1D.dpred(sigma_back)

    # Make sure that the numeric and the 1D solutions have the same sign.
    # This tests the currents have consistent orientations between the EB and HJ
    # formulations, a bug fixed in #1812
    np.testing.assert_equal(np.sign(dpred_numeric), np.sign(dpred1d), strict=True)

    # Compare the numeric b and dbdt with their respective analytic solutions
    b_numeric = dpred_numeric[: dpred_numeric.size // 2]
    b_1d = dpred1d[: dpred1d.size // 2]
    atol = 1e-3 * np.max(np.abs(b_1d))
    np.testing.assert_allclose(b_numeric, b_1d, rtol=REL_TOL, atol=atol, strict=True)

    dbdt_numeric = dpred_numeric[dpred_numeric.size // 2 :]
    dbdt_1d = dpred1d[dpred1d.size // 2 :]
    atol = 1e-7 * np.max(np.abs(dbdt_1d))
    np.testing.assert_allclose(
        dbdt_numeric, dbdt_1d, rtol=REL_TOL, atol=atol, strict=True
    )
