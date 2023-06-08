import pytest

import numpy as np
from scipy.constants import epsilon_0

import geoana
import discretize
from discretize import utils
from SimPEG.electromagnetics import frequency_domain as fdem
from SimPEG.electromagnetics import resistivity as dc
from SimPEG import utils, maps, Report
from pymatsolver import Pardiso


# set up the mesh
hx = 1
hz = 1
nx = 50
nz = int(2 * nx) + 1
npad = 20
pf = 1.3

mesh = discretize.CylindricalMesh(
    [[(hx, nx), (hx, npad, pf)], 1, [(hz, npad, -pf), (hz, nz), (hz, npad, pf)]],
    x0="00C",
)

sigma = 1e-2
conductivity = sigma * np.ones(mesh.n_cells)
epsilon_r_list = [0, 1, 1e3, 1e4, 1e5, 1e6]
epsilon_list = [epsilon_0 * e_r for e_r in epsilon_r_list]
frequency_list = [50, 100]

threshold = 1e-1


def get_inds(val, x, z):
    if len(val) == mesh.n_faces:
        grid = mesh.faces
    elif len(val) == mesh.n_edges:
        grid = mesh.edges

    return (
        (grid[:, 0] > x.min())
        & (grid[:, 0] < x.max())
        & (grid[:, 2] > z.min())
        & (grid[:, 2] < z.max())
    )


def print_comparison(
    numeric, analytic, x=np.r_[50, 100], z=np.r_[-100, 100], threshold=threshold
):
    inds = get_inds(numeric, x, z)
    results = []
    for component in ["real", "imag"]:
        numeric_norm = np.linalg.norm(np.abs(getattr(numeric[inds], component)))
        analytic_norm = np.linalg.norm(np.abs(getattr(analytic[inds], component)))
        difference = np.linalg.norm(
            np.abs(
                getattr(analytic[inds], component) - getattr(numeric[inds], component)
            )
        )
        print(f"{component} numeric    : {numeric_norm:1.4e}")
        print(f"{component} analytic   : {analytic_norm:1.4e}")
        print(f"{component} difference : {difference:1.4e}\n")
        results.append(difference / np.mean([numeric_norm, analytic_norm]) < threshold)
    print(results)
    return results


@pytest.mark.parametrize("epsilon", epsilon_list)
@pytest.mark.parametrize("frequency", frequency_list)
@pytest.mark.parametrize(
    "simulation",
    [
        lambda survey, epsilon: fdem.Simulation3DElectricField(
            mesh,
            survey=survey,
            forward_only=True,
            sigma=conductivity,
            permittivity=epsilon,
            solver=Pardiso,
        ),
        lambda survey, epsilon: fdem.Simulation3DMagneticFluxDensity(
            mesh,
            survey=survey,
            forward_only=True,
            sigma=conductivity,
            permittivity=epsilon,
            solver=Pardiso,
        ),
    ],
)
def test_mag_dipole(epsilon, frequency, simulation):
    sources = [fdem.sources.MagDipole([], frequency, location=np.r_[0, 0, 0])]
    survey = fdem.Survey(sources)
    sim = simulation(survey, epsilon)
    fields = sim.fields()

    analytic_bdipole = geoana.em.fdem.MagneticDipoleWholeSpace(
        sigma=sigma, epsilon=epsilon, frequency=frequency, orientation="Z"
    )
    analytics = {
        "b": np.hstack(
            [
                analytic_bdipole.magnetic_flux_density(mesh.faces_x)[:, 0],
                analytic_bdipole.magnetic_flux_density(mesh.faces_z)[:, 2],
            ]
        ),
        "e": analytic_bdipole.electric_field(mesh.edges_y)[:, 1],
        "h": np.hstack(
            [
                analytic_bdipole.magnetic_field(mesh.faces_x)[:, 0],
                analytic_bdipole.magnetic_field(mesh.faces_z)[:, 2],
            ]
        ),
        "j": analytic_bdipole.current_density(mesh.edges_y)[:, 1],
    }

    for f, analytic in analytics.items():
        print(f"Testing Mag dipole: {f}")
        test = print_comparison(fields[:, f].squeeze(), analytic)
        assert np.all(test)


@pytest.mark.parametrize("epsilon", epsilon_list)
@pytest.mark.parametrize("frequency", frequency_list)
@pytest.mark.parametrize(
    "simulation",
    [
        lambda survey, epsilon: fdem.Simulation3DCurrentDensity(
            mesh,
            survey=survey,
            forward_only=True,
            sigma=conductivity,
            permittivity=epsilon,
            solver=Pardiso,
        ),
        lambda survey, epsilon: fdem.Simulation3DMagneticField(
            mesh,
            survey=survey,
            forward_only=True,
            sigma=conductivity,
            permittivity=epsilon,
            solver=Pardiso,
        ),
    ],
)
def test_e_dipole(epsilon, frequency, simulation):
    sources = [
        fdem.sources.LineCurrent(
            [], frequency, location=np.array([[0, 0, 1], [0, 0, -1]]), current=1 / 2
        )
    ]
    survey = fdem.Survey(sources)
    sim = simulation(survey, epsilon)
    fields = sim.fields()

    analytic_edipole = geoana.em.fdem.ElectricDipoleWholeSpace(
        sigma=sigma, epsilon=epsilon, frequency=frequency, orientation="Z"
    )
    analytics = {
        "j": np.hstack(
            [
                analytic_edipole.current_density(mesh.faces_x)[:, 0],
                analytic_edipole.current_density(mesh.faces_z)[:, 2],
            ]
        ),
        "h": analytic_edipole.magnetic_field(mesh.edges_y)[:, 1],
        "e": np.hstack(
            [
                analytic_edipole.electric_field(mesh.faces_x)[:, 0],
                analytic_edipole.electric_field(mesh.faces_z)[:, 2],
            ]
        ),
        "b": analytic_edipole.magnetic_flux_density(mesh.edges_y)[:, 1],
    }

    for f, analytic in analytics.items():
        print(f"Testing E dipole: {f}")
        test = print_comparison(fields[:, f].squeeze(), analytic)
        assert np.all(test)
