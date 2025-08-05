import numpy as np
import pytest

import discretize

from simpeg import maps

from simpeg.electromagnetics import frequency_domain as fdem
from simpeg.utils.solver_utils import get_default_solver

SOLVER = get_default_solver()

# relative tolerances
RELTOL = 1e-2
MINTOL = 1e-20  # minimum tolerance we test, anything below this is "ZERO"

FREQUENCY = 5e-1
SIMULATION_TYPES = ["e", "b", "h", "j"]
FIELDS_TEST = ["e", "b", "h", "j", "charge", "charge_density"]

VERBOSE = True


def get_fdem_simulation(mesh, fdem_type, frequency):
    mapping = maps.ExpMap(mesh)

    source_list = [
        fdem.sources.MagDipole([], frequency=frequency, location=np.r_[0.0, 0.0, 10.0])
    ]
    survey = fdem.Survey(source_list)

    if fdem_type == "e":
        sim = fdem.Simulation3DElectricField(
            mesh, survey=survey, sigmaMap=mapping, solver=SOLVER
        )

    elif fdem_type == "b":
        sim = fdem.Simulation3DMagneticFluxDensity(
            mesh, survey=survey, sigmaMap=mapping, solver=SOLVER
        )

    elif fdem_type == "j":
        sim = fdem.Simulation3DCurrentDensity(
            mesh, survey=survey, sigmaMap=mapping, solver=SOLVER
        )

    elif fdem_type == "h":
        sim = fdem.Simulation3DMagneticField(
            mesh, survey=survey, sigmaMap=mapping, solver=SOLVER
        )

    return sim


class TestFieldsCrosscheck:

    @property
    def mesh(self):
        if getattr(self, "_mesh", None) is None:
            cs = 10.0
            ncx, ncy, ncz = 4, 4, 4
            npad = 4
            pf = 1.3
            hx = [(cs, npad, -pf), (cs, ncx), (cs, npad, pf)]
            hy = [(cs, npad, -pf), (cs, ncy), (cs, npad, pf)]
            hz = [(cs, npad, -pf), (cs, ncz), (cs, npad, pf)]
            self._mesh = discretize.TensorMesh([hx, hy, hz], ["C", "C", "C"])
        return self._mesh

    @property
    def model(self):
        if getattr(self, "_model", None) is None:
            sigma_background = 10
            sigma_target = 1e-2
            sigma_air = 1e-8

            target_width = 40
            target_depth = -20

            inds_target = (
                (self.mesh.cell_centers[:, 0] > -target_width / 2)
                & (self.mesh.cell_centers[:, 0] < target_width / 2)
                & (self.mesh.cell_centers[:, 1] > -target_width / 2)
                & (self.mesh.cell_centers[:, 1] < target_width / 2)
                & (self.mesh.cell_centers[:, 2] > -target_width / 2 + target_depth)
                & (self.mesh.cell_centers[:, 2] < target_width / 2 + target_depth)
            )

            sigma_model = sigma_background * np.ones(self.mesh.n_cells)
            sigma_model[self.mesh.cell_centers[:, 2] > 0] = sigma_air

            sigma_model[inds_target] = sigma_target

            self._model = np.log(sigma_model)
        return self._model

    @property
    def simulation_dict(self):
        if getattr(self, "_simulation_dict", None) is None:
            self._simulation_dict = {
                key: get_fdem_simulation(self.mesh, key, FREQUENCY)
                for key in SIMULATION_TYPES
            }
        return self._simulation_dict

    @property
    def fields_dict(self):
        if getattr(self, "_fields_dict", None) is None:
            self._fields_dict = {
                key: sim.fields(self.model) for key, sim in self.simulation_dict.items()
            }
        return self._fields_dict

    def compare_fields(self, field1, field2, relative_tolerance, verbose=False):
        norm_diff = np.linalg.norm(field1 - field2)
        abs_tol = np.max(
            [
                relative_tolerance
                * (np.linalg.norm(field1) + np.linalg.norm(field2))
                / 2,
                MINTOL,
            ]
        )
        test = norm_diff < abs_tol

        if verbose is True:
            print(f"||diff||: {norm_diff:1.2e} < TOL: {abs_tol:1.2e} ?  {test}")

        return test

    @pytest.mark.parametrize("sim_pairs", [("e", "b"), ("h", "j")])
    @pytest.mark.parametrize("field_test", FIELDS_TEST)
    def test_fields_cross_check_EBHJ(
        self, sim_pairs, field_test, relative_tolerance=RELTOL, verbose=VERBOSE
    ):
        field1 = self.fields_dict[sim_pairs[0]][:, field_test]
        field2 = self.fields_dict[sim_pairs[1]][:, field_test]

        if verbose is True:
            print(f"Testing simulations {sim_pairs} for field {field_test}")

        assert self.compare_fields(field1, field2, relative_tolerance, verbose)
