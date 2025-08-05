"""
Test the getJ method of FDEM 1D simulation.
"""

import numpy as np
import simpeg.electromagnetics.frequency_domain as fdem
from simpeg import maps


def create_simulation_and_conductivities(identity_mapping: bool):
    # Create Survey
    # -------------
    # Source properties
    frequencies = np.r_[382, 1822, 7970, 35920, 130100]  # frequencies in Hz
    source_location = np.array([0.0, 0.0, 30.0])  # (3, ) numpy.array_like
    source_orientation = "z"  # "x", "y" or "z"
    moment = 1.0  # dipole moment in Am^2

    # Receiver properties
    receiver_locations = np.array([10.0, 0.0, 30.0])  # or (N, 3) numpy.ndarray
    receiver_orientation = "z"  # "x", "y" or "z"
    data_type = "ppm"  # "secondary", "total" or "ppm"

    source_list = []  # create empty list for source objects

    # loop over all sources
    for freq in frequencies:
        # Define receivers that measure real and imaginary component
        # magnetic field data in ppm.
        receiver_list = []
        receiver_list.append(
            fdem.receivers.PointMagneticFieldSecondary(
                receiver_locations,
                orientation=receiver_orientation,
                data_type=data_type,
                component="real",
            )
        )
        receiver_list.append(
            fdem.receivers.PointMagneticFieldSecondary(
                receiver_locations,
                orientation=receiver_orientation,
                data_type=data_type,
                component="imag",
            )
        )

        # Define a magnetic dipole source at each frequency
        source_list.append(
            fdem.sources.MagDipole(
                receiver_list=receiver_list,
                frequency=freq,
                location=source_location,
                orientation=source_orientation,
                moment=moment,
            )
        )

    # Define the survey
    survey = fdem.survey.Survey(source_list)

    # Defining a 1D Layered Earth Model
    # ---------------------------------
    # Define layer thicknesses (m)
    thicknesses = np.array([20.0, 40.0])

    # Define layer conductivities (S/m)
    conductivities = np.r_[0.1, 1.0, 0.1]

    # Define a mapping
    n_layers = len(conductivities)
    model_mapping = (
        maps.IdentityMap(nP=n_layers) if identity_mapping else maps.ExpMap(nP=n_layers)
    )

    # Define the Forward Simulation, Predict Data and Plot
    # ----------------------------------------------------
    simulation = fdem.Simulation1DLayered(
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
