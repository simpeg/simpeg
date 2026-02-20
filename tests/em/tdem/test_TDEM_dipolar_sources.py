from discretize import TensorMesh

from simpeg import maps
import simpeg.electromagnetics.time_domain as tdem

import numpy as np

from simpeg.utils.solver_utils import get_default_solver

Solver = get_default_solver()

TOL = 0.06  # relative tolerance

# Observation times for response (time channels)
n_times = 30
time_channels = np.logspace(-4, -1, n_times)

# Defining transmitter locations
source_locations = np.r_[0, 0, 15.5]
receiver_locations = np.atleast_2d(np.r_[0, 0, 15.5])


def create_survey(src_type="MagDipole"):

    bz_receiver = tdem.receivers.PointMagneticFluxDensity(
        receiver_locations, time_channels, "z"
    )
    dbdtz_receiver = tdem.receivers.PointMagneticFluxTimeDerivative(
        receiver_locations, time_channels, "z"
    )
    receivers_list = [bz_receiver, dbdtz_receiver]

    source_list = [
        getattr(tdem.sources, src_type)(
            receivers_list,
            location=source_locations,
            waveform=tdem.sources.StepOffWaveform(),
            moment=1.0,
            orientation="z",
        )
    ]
    survey = tdem.Survey(source_list)
    return survey


def test_BH_dipole():
    survey_b = create_survey()
    survey_h = create_survey()

    cell_size = 20
    n_core = 10
    padding_factor = 1.3
    n_padding = 15

    h = [
        (cell_size, n_padding, -padding_factor),
        (cell_size, n_core),
        (cell_size, n_padding, padding_factor),
    ]
    mesh = TensorMesh([h, h, h], origin="CCC")

    air_conductivity = 1e-8
    background_conductivity = 1e-1

    model = air_conductivity * np.ones(mesh.n_cells)
    model[mesh.cell_centers[:, 2] < 0] = background_conductivity

    nsteps = 10
    time_steps = [
        (1e-5, nsteps),
        (3e-5, nsteps),
        (1e-4, nsteps),
        (3e-4, nsteps),
        (1e-3, nsteps),
        (3e-3, nsteps),
        (1e-2, nsteps - 4),
    ]

    simulation_b = tdem.simulation.Simulation3DMagneticFluxDensity(
        mesh,
        survey=survey_b,
        sigmaMap=maps.IdentityMap(),
        solver=Solver,
        time_steps=time_steps,
    )

    simulation_h = tdem.simulation.Simulation3DMagneticField(
        mesh,
        survey=survey_h,
        sigmaMap=maps.IdentityMap(),
        solver=Solver,
        time_steps=time_steps,
    )

    fields_b = simulation_b.fields(model)
    dpred_b = simulation_b.dpred(model, f=fields_b)

    fields_h = simulation_h.fields(model)
    dpred_h = simulation_h.dpred(model, f=fields_h)

    # Check if the two predicted fields are close enough
    np.testing.assert_allclose(dpred_h, dpred_b, rtol=TOL)
