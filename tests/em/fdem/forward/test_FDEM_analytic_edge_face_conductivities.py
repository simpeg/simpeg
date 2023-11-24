import unittest

import discretize
import numpy as np
from scipy.constants import mu_0
from SimPEG import maps
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
    loop_radius = np.pi**-0.5
    receiver_location = np.c_[12.0, 0.0, 1.0]
    source_location = np.r_[0.0, 0.0, 1.0]
    frequencies = np.logspace(2, 3, 2)

    layer_depth = 40.0
    layer_thickness = 0.1
    layer_conductivity = 100
    background_conductivity = 2.5e-3

    tau = layer_thickness * layer_conductivity

    # 1D LAYER MODEL
    thicknesses = np.array([layer_depth - layer_thickness / 2, layer_thickness])
    n_layer = len(thicknesses) + 1

    sigma_1d = background_conductivity * np.ones(n_layer)
    sigma_1d[1] = layer_conductivity

    sigma_map_1d = maps.IdentityMap(nP=n_layer)

    # 3D LAYER MODEL
    if mesh_type == "CYL":
        hr = [(2.0, 120), (2.0, 25, 1.3)]
        hz = [(2.0, 25, -1.3), (2.0, 200), (2.0, 25, 1.3)]

        mesh = discretize.CylindricalMesh([hr, 1, hz], x0="00C")

        ind = np.where(mesh.h[2] == np.min(mesh.h[2]))[0]
        ind = ind[int(len(ind) / 2)]

        mesh.origin = mesh.origin - np.r_[0.0, 0.0, mesh.nodes_z[ind] - 24]

    elif mesh_type == "TREE":
        dh = 2.5  # base cell width
        dom_width = 8000.0  # domain width
        nbc = 2 ** int(
            np.round(np.log(dom_width / dh) / np.log(2.0))
        )  # num. base cells

        h = [(dh, nbc)]
        mesh = discretize.TreeMesh([h, h, h], x0="CCC")
        mesh.refine_points(
            np.reshape(source_location, (1, 3)),
            level=-1,
            padding_cells_by_level=[8, 4, 4, 4],
            finalize=False,
        )
        x0s = np.vstack([ii * np.c_[-60, -60, -60] for ii in range(1, 5)])
        x1s = np.vstack([ii * np.c_[60, 60, 10] for ii in range(1, 5)])

        mesh.refine_box(x0s, x1s, levels=[-2, -3, -4, -5], finalize=False)
        mesh.finalize()

    sigma_3d = 1e-8 * np.ones(mesh.nC)
    sigma_3d[mesh.cell_centers[:, -1] < 0.0] = background_conductivity

    tau_3d = np.zeros(mesh.nF)
    tau_3d[np.isclose(mesh.faces[:, -1], -layer_depth)] = tau
    tau_map = maps.IdentityMap(nP=mesh.n_faces)

    # DEFINE SURVEY
    rx_list = [
        getattr(fdem.receivers, "Point{}Secondary".format(rx_type))(
            receiver_location, component="real", orientation=orientation
        ),
        getattr(fdem.receivers, "Point{}Secondary".format(rx_type))(
            receiver_location, component="imag", orientation=orientation
        ),
    ]

    # 1D SURVEY AND SIMULATION
    src_1d = [
        fdem.sources.MagDipole(
            rx_list, f, location=np.r_[0.0, 0.0, 1.0], orientation=orientation
        )
        for f in frequencies
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
            )
            for f in frequencies
        ]
    else:
        src_3d = [
            fdem.sources.MagDipole(
                rx_list,
                f,
                location=source_location,
                orientation=orientation,
            )
            for f in frequencies
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

    # print(analytic_solution)
    # print(numeric_solution)

    diff = np.linalg.norm(
        np.abs(numeric_solution - analytic_solution)
    ) / np.linalg.norm(np.abs(analytic_solution))

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

    def test_tree_Bform_magdipole_b_x(self):
        assert (
            analytic_layer_small_loop_face_conductivity_comparison(
                mesh_type="TREE",
                formulation="MagneticFluxDensity",
                rx_type="MagneticFluxDensity",
                orientation="X",
                bounds=None,
                plotIt=False,
            )
            < 0.04
        )

    def test_tree_Bform_magdipole_b_z(self):
        assert (
            analytic_layer_small_loop_face_conductivity_comparison(
                mesh_type="TREE",
                formulation="MagneticFluxDensity",
                rx_type="MagneticFluxDensity",
                orientation="Z",
                bounds=None,
                plotIt=False,
            )
            < 0.04
        )

    def test_cyl_Bform_loop_b_z(self):
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

    def test_tree_Eform_magdipole_b_x(self):
        assert (
            analytic_layer_small_loop_face_conductivity_comparison(
                mesh_type="TREE",
                formulation="ElectricField",
                rx_type="MagneticFluxDensity",
                orientation="X",
                bounds=None,
                plotIt=False,
            )
            < 0.04
        )

    def test_tree_Eform_magdipole_b_z(self):
        assert (
            analytic_layer_small_loop_face_conductivity_comparison(
                mesh_type="TREE",
                formulation="ElectricField",
                rx_type="MagneticFluxDensity",
                orientation="Z",
                bounds=None,
                plotIt=False,
            )
            < 0.04
        )

    def test_cyl_Eform_loop_b_z(self):
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
