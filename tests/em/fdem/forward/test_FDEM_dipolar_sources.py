from scipy.constants import mu_0
import numpy as np
import pytest

from discretize import TensorMesh
from geoana.em.static import MagneticDipoleWholeSpace
import simpeg.electromagnetics.frequency_domain as fdem
from simpeg import maps

from simpeg.utils.solver_utils import get_default_solver

Solver = get_default_solver()

TOL = 5e-2  # relative tolerance

# Defining transmitter locations
source_location = np.r_[0, 0, 0]


def create_survey(source_type="MagDipole", mu=mu_0, orientation="Z"):

    freq = 10

    # Must define the transmitter properties and associated receivers
    source_list = [
        getattr(fdem.sources, source_type)(
            [],
            location=source_location,
            frequency=freq,
            moment=1.0,
            orientation=orientation,
            mu=mu,
        )
    ]

    survey = fdem.Survey(source_list)
    return survey


def create_mesh_model():
    cell_size = 20
    n_core = 10
    padding_factor = 1.3
    n_padding = 10

    h = [
        (cell_size, n_padding, -padding_factor),
        (cell_size, n_core),
        (cell_size, n_padding, padding_factor),
    ]
    mesh = TensorMesh([h, h, h], origin="CCC")

    # Conductivity in S/m
    air_conductivity = 1e-8
    background_conductivity = 1e-1

    model = air_conductivity * np.ones(mesh.n_cells)
    model[mesh.cell_centers[:, 2] < 0] = background_conductivity

    return mesh, model


@pytest.mark.parametrize("simulation_type", ["e", "b", "h", "j"])
@pytest.mark.parametrize("field_test", ["bPrimary", "hPrimary"])
@pytest.mark.parametrize("mur", [1, 50])
def test_dipolar_fields(simulation_type, field_test, mur, orientation="Z"):

    mesh, model = create_mesh_model()
    survey = create_survey("MagDipole", mu=mur * mu_0, orientation="Z")

    if simulation_type in ["e", "b"]:
        grid = mesh.faces
        projection = mesh.project_face_vector
        if simulation_type == "e":
            sim = fdem.simulation.Simulation3DElectricField(
                mesh, survey=survey, sigmaMap=maps.IdentityMap(), solver=Solver
            )
        elif simulation_type == "b":
            sim = fdem.simulation.Simulation3DMagneticFluxDensity(
                mesh, survey=survey, sigmaMap=maps.IdentityMap(), solver=Solver
            )

    elif simulation_type in ["h", "j"]:
        grid = mesh.edges
        projection = mesh.project_edge_vector
        if simulation_type == "h":
            sim = fdem.simulation.Simulation3DMagneticField(
                mesh, survey=survey, sigmaMap=maps.IdentityMap(), solver=Solver
            )
        elif simulation_type == "j":
            sim = fdem.simulation.Simulation3DCurrentDensity(
                mesh, survey=survey, sigmaMap=maps.IdentityMap(), solver=Solver
            )

    # get numeric solution
    src = survey.source_list[0]
    numeric = getattr(src, field_test)(sim)

    # get analytic
    dipole = MagneticDipoleWholeSpace(orientation=orientation, mu=mur * mu_0)

    if field_test == "bPrimary":
        analytic = projection(dipole.magnetic_flux_density(grid))
    elif field_test == "hPrimary":
        analytic = projection(dipole.magnetic_field(grid))

    assert np.abs(np.mean((numeric / analytic)) - 1) < TOL
