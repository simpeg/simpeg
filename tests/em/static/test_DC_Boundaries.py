import numpy as np
import pytest
import discretize
from discretize.utils import example_simplex_mesh

import simpeg.electromagnetics.static.resistivity as dc


tens_2d = discretize.TensorMesh([8, 9])
tens_3d = discretize.TensorMesh([8, 9, 10])
tree_2d = discretize.TreeMesh([16, 16])
tree_2d.refine(4)
tree_3d = discretize.TreeMesh([16, 16, 16])
tree_3d.refine(4)

simp_2d = discretize.SimplexMesh(*example_simplex_mesh((5, 6)))
simp_3d = discretize.SimplexMesh(*example_simplex_mesh((4, 5, 6)))


@pytest.mark.parametrize(
    "mesh,sim_class,tx_loc",
    [
        (tens_3d, dc.Simulation3DNodal, [0.5, 0.5, 1]),
        (tens_3d, dc.Simulation3DCellCentered, [0.5, 0.5, 1]),
        (tens_2d, dc.Simulation3DNodal, [0.5, 1]),
        (tens_2d, dc.Simulation3DCellCentered, [0.5, 1]),
        (tens_2d, dc.Simulation2DNodal, [0.5, 1]),
        (tens_2d, dc.Simulation2DCellCentered, [0.5, 1]),
        (tree_2d, dc.Simulation2DNodal, [0.5, 1]),
        (tree_2d, dc.Simulation2DCellCentered, [0.5, 1]),
        (tree_3d, dc.Simulation3DNodal, [0.5, 0.5, 1]),
        (tree_3d, dc.Simulation3DCellCentered, [0.5, 0.5, 1]),
    ],
)
def test_custom_surface(mesh, sim_class, tx_loc):
    bf = mesh.boundary_faces
    surface_boundary_faces = bf[:, -1] == bf[:, -1].max()

    src = dc.sources.Pole([], location=tx_loc)
    survey = dc.Survey([src])

    sigma = np.full(mesh.n_cells, 0.1)

    sim1 = sim_class(mesh, survey, sigma=sigma)
    sim2 = sim_class(mesh, survey, surface_faces=surface_boundary_faces, sigma=sigma)

    f1 = sim1.fields()
    f2 = sim2.fields()

    np.testing.assert_equal(f1[src, "phiSolution"], f2[src, "phiSolution"])


@pytest.mark.parametrize(
    "mesh,sim_class,tx_loc",
    [
        (simp_3d, dc.Simulation3DNodal, [0.5, 0.5, 1]),
        (simp_3d, dc.Simulation3DCellCentered, [0.5, 0.5, 1]),
        (simp_2d, dc.Simulation2DNodal, [0.5, 1]),
        (simp_2d, dc.Simulation2DCellCentered, [0.5, 1]),
    ],
)
def test_not_implemented(mesh, sim_class, tx_loc):
    src = dc.sources.Pole([], location=tx_loc)
    survey = dc.Survey([src])
    sigma = np.full(mesh.n_cells, 0.1)

    with pytest.raises(NotImplementedError):
        sim = sim_class(mesh, survey, sigma=sigma)
        sim.fields()
