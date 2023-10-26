import unittest

import discretize
import matplotlib.pyplot as plt
import numpy as np
from pymatsolver import Pardiso as Solver
from scipy.constants import mu_0
from SimPEG import maps
from SimPEG.electromagnetics import analytics
from SimPEG.electromagnetics import frequency_domain as fdem


def analytic_layer_small_loop_face_conductivity_comparison(
    mesh_type="CYL",
    formulation="ElectricField",
    rx_type="MagneticFluxDensity",
    orientation="Z",
    bounds=None,
    plotIt=False,
):
    # Some static parameters
    PHI = np.linspace(0, 2 * np.pi, 21)
    loop_radius = np.pi**-0.5
    receiver_location = np.c_[50.0, 0.0, 1.0]
    source_location = np.r_[0.0, 0.0, 1.0]

    if orientation == "X":
        source_nodes = np.c_[
            np.zeros_like(PHI),
            loop_radius * np.cos(PHI),
            1.0 + loop_radius * np.sin(PHI),
        ]
    elif orientation == "Z":
        source_nodes = np.c_[
            loop_radius * np.cos(PHI), loop_radius * np.sin(PHI), np.ones_like(PHI)
        ]

    layer_depth = 24.0
    layer_thickness = 0.1
    layer_conductivity = 5e-3
    background_conductivity = 5e-3

    tau = layer_thickness * layer_conductivity

    # if bounds is None:
    #     bounds = [1e-5, 1e-3]

    # 1D LAYER MODEL
    thicknesses = np.array([layer_depth - layer_thickness / 2, layer_thickness])
    n_layer = len(thicknesses) + 1

    sigma_1d = background_conductivity * np.ones(n_layer)
    sigma_1d[1] = layer_conductivity

    sigma_map_1d = maps.IdentityMap(nP=n_layer)

    # 3D LAYER MODEL
    if mesh_type == "CYL":
        cs, ncx, ncz, npad = 4.0, 40, 20, 20
        hx = [(cs, ncx), (cs, npad, 1.3)]
        hz = [(cs, npad, -1.3), (cs, ncz), (cs, npad, 1.3)]
        mesh = discretize.CylindricalMesh([hx, 1, hz], "00C")

    elif mesh_type == "TENSOR":
        cs, nc, npad = 8.0, 15, 10
        hx = [(cs, npad, -1.3), (cs, nc), (cs, npad, 1.3)]
        hy = [(cs, npad, -1.3), (cs, nc), (cs, npad, 1.3)]
        hz = [(cs, npad, -1.3), (cs, nc), (cs, npad, 1.3)]
        mesh = discretize.TensorMesh([hx, hy, hz], "CCC")

    sigma_3d = 1e-8 * np.ones(mesh.nC)
    sigma_3d[mesh.cell_centers[:, -1] < 0.0] = background_conductivity

    tau_3d = np.zeros(mesh.nF)
    # tau_3d[np.isclose(mesh.faces[:, -1], -layer_depth)] = tau
    tau_map = maps.IdentityMap(nP=mesh.n_faces)

    # DEFINE SURVEY
    frequencies = np.logspace(3, 4, 2)
    rx_list = [
        getattr(fdem.receivers, "Point{}Secondary".format(rx_type))(
            receiver_location, component="real", orientation=orientation
        ),
        getattr(fdem.receivers, "Point{}Secondary".format(rx_type))(
            receiver_location, component="imag", orientation=orientation
        )
    ]

    # 1D SURVEY AND SIMULATION
    src_1d = [
        fdem.sources.MagDipole(
            rx_list,
            f,
            location=np.r_[0.0, 0.0, 1.0],
            orientation=orientation
        ) for f in frequencies
    ]
    survey_1d = fdem.Survey(src_1d)

    sim_1d = fdem.Simulation1DLayered(
        survey=survey_1d,
        thicknesses=thicknesses,
        sigmaMap=sigma_map_1d,
    )

    # 3D SURVEY AND SIMULATION
    if mesh_type == "CYL":
        src_3d = [
            fdem.sources.CircularLoop(
                rx_list,
                f,
                radius=loop_radius,
                location=source_location,
            ) for f in frequencies
        ]
    else:
        if formulation == "MagneticFluxDensity":
            src_3d = [
                fdem.sources.MagDipole(
                    rx_list,
                    f,
                    location=source_location,
                    orientation=orientation,
                ) for f in frequencies
            ]
        else:
            src_3d = [
                fdem.sources.LineCurrent(
                    rx_list, f, location=source_nodes,
                ) for f in frequencies
            ]

    survey_3d = fdem.Survey(src_3d)

    # DEFINE THE SIMULATIONS
    if formulation == "MagneticFluxDensity":
        sim_3d = fdem.simulation.Simulation3DMagneticFluxDensityFaceEdgeConductivity(
            mesh=mesh, survey=survey_3d, sigma=sigma_3d, tauMap=tau_map
        )
    else:
        sim_3d = fdem.simulation.Simulation3DElectricFieldFaceEdgeConductivity(
            mesh=mesh, survey=survey_3d, sigma=sigma_3d, tauMap=tau_map
        )

    # COMPUTE SOLUTIONS
    analytic_solution = mu_0 * sim_1d.dpred(sigma_1d)  # ALWAYS RETURNS H-FIELD
    numeric_solution = sim_3d.dpred(tau_3d)
    
    print(analytic_solution)
    print(numeric_solution)

    diff = (
        np.linalg.norm(np.abs(numeric_solution - analytic_solution)) /
        np.linalg.norm(np.abs(analytic_solution))
    )

    print(
        " |bz_ana| = {ana} |bz_num| = {num} |bz_ana-bz_num| = {diff}".format(
            ana=np.linalg.norm(analytic_solution),
            num=np.linalg.norm(numeric_solution),
            diff=np.linalg.norm(analytic_solution - numeric_solution),
        )
    )
    print("Difference: {}".format(diff))
    
    return diff


class LayerConductanceTests(unittest.TestCase):
    # Compares analytic 1D layered Earth solution to a plate of equivalent
    # conductance.

    # def test_tensor_magdipole_b_x(self):
    #     assert (
    #         analytic_layer_small_loop_face_conductivity_comparison(
    #             mesh_type="TENSOR",
    #             formulation="MagneticFluxDensity",
    #             rx_type="MagneticFluxDensity",
    #             orientation="X",
    #             bounds=None,
    #             plotIt=False,
    #         )
    #         < 0.01
    #     )

    # def test_tensor_magdipole_b_z(self):
    #     assert (
    #         analytic_layer_small_loop_face_conductivity_comparison(
    #             mesh_type="TENSOR",
    #             formulation="MagneticFluxDensity",
    #             rx_type="MagneticFluxDensity",
    #             orientation="Z",
    #             bounds=None,
    #             plotIt=False,
    #         )
    #         < 0.02
    #     )

    def test_cyl_magdipole_b_z(self):
        assert (
            analytic_layer_small_loop_face_conductivity_comparison(
                mesh_type="CYL",
                formulation="MagneticFluxDensity",
                rx_type="MagneticFluxDensity",
                orientation="Z",
                bounds=None,
                plotIt=False,
            )
            < 0.01
        )

    # def test_tensor_linecurrent_b_x(self):
    #     assert (
    #         analytic_layer_small_loop_face_conductivity_comparison(
    #             mesh_type="TENSOR",
    #             formulation="ElectricField",
    #             rx_type="MagneticFluxDensity",
    #             orientation="X",
    #             bounds=None,
    #             plotIt=False,
    #         )
    #         < 0.01
    #     )

    # def test_tensor_linecurrent_b_z(self):
    #     assert (
    #         analytic_layer_small_loop_face_conductivity_comparison(
    #             mesh_type="TENSOR",
    #             formulation="ElectricField",
    #             rx_type="MagneticFluxDensity",
    #             orientation="Z",
    #             bounds=None,
    #             plotIt=False,
    #         )
    #         < 0.01
    #     )

    def test_cyl_linecurrent_b_z(self):
        assert (
            analytic_layer_small_loop_face_conductivity_comparison(
                mesh_type="CYL",
                formulation="ElectricField",
                rx_type="MagneticFluxDensity",
                orientation="Z",
                bounds=None,
                plotIt=False,
            )
            < 0.01
        )


if __name__ == "__main__":
    unittest.main()