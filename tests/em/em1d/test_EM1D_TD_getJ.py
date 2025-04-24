"""
Test the getJ method of FDEM 1D simulation.
"""

import numpy as np
import simpeg.electromagnetics.time_domain as tdem
from simpeg import maps


def create_simulation_and_conductivities(identity_mapping: bool):
    # Create Survey
    # -------------
    # Source properties
    source_location = np.array([0.0, 0.0, 20.0])
    source_orientation = "z"  # "x", "y" or "z"
    source_current = 1.0  # maximum on-time current
    source_radius = 6.0  # source loop radius

    # Receiver properties
    receiver_location = np.array([0.0, 0.0, 20.0])
    receiver_orientation = "z"  # "x", "y" or "z"
    times = np.logspace(-5, -2, 31)  # time channels (s)

    # Define receiver list. In our case, we have only a single receiver for each source.
    # When simulating the response for multiple component and/or field orientations,
    # the list consists of multiple receiver objects.
    receiver_list = []
    receiver_list.append(
        tdem.receivers.PointMagneticFluxDensity(
            receiver_location, times, orientation=receiver_orientation
        )
    )

    # Define the source waveform. Here we define a unit step-off. The definition
    # of other waveform types is covered in a separate tutorial.
    waveform = tdem.sources.StepOffWaveform()

    # Define source list. In our case, we have only a single source.
    source_list = [
        tdem.sources.CircularLoop(
            receiver_list=receiver_list,
            location=source_location,
            orientation=source_orientation,
            waveform=waveform,
            current=source_current,
            radius=source_radius,
        )
    ]

    # Define the survey
    survey = tdem.Survey(source_list)

    # Defining a 1D Layered Earth Model
    # ---------------------------------
    # Physical properties
    background_conductivity = 1e-1
    layer_conductivity = 1e0

    # Layer thicknesses
    thicknesses = np.array([40.0, 40.0])
    n_layer = len(thicknesses) + 1

    # Conductivities
    conductivities = background_conductivity * np.ones(n_layer)
    conductivities[1] = layer_conductivity

    # Define a mapping
    model_mapping = (
        maps.IdentityMap(nP=n_layer) if identity_mapping else maps.ExpMap(nP=n_layer)
    )

    # Define the Forward Simulation, Predict Data and Plot
    # ----------------------------------------------------
    simulation = tdem.Simulation1DLayered(
        survey=survey,
        thicknesses=thicknesses,
        sigmaMap=model_mapping,
    )

    return simulation, conductivities


def test_getJ():
    """
    Test if getJ returns different J matrices after passing different maps.
    """
    dpreds, jacobians = [], []

    # Compute dpred and J using an identity map and an exp map
    for identity_mapping in (True, False):
        simulation, conductivities = create_simulation_and_conductivities(
            identity_mapping
        )
        model = conductivities if identity_mapping else np.log(conductivities)
        dpreds.append(simulation.dpred(model))
        jac = simulation.getJ(model)
        jacobians.append(jac)

    # The two dpreds should be equal
    assert np.allclose(*dpreds)

    # The two J matrices should not be equal
    assert not np.allclose(*jacobians, atol=0.0)
