import numpy as np
import pytest
import discretize

import SimPEG.electromagnetics.static.resistivity as dc


@pytest.mark.parametrize(
    "mesh,sim_class,tx_loc",
    [
        (discretize.TensorMesh([8, 9, 10]), dc.Simulation3DNodal, [0.5, 0.5, 1]),
        (discretize.TensorMesh([8, 9, 10]), dc.Simulation3DCellCentered, [0.5, 0.5, 1]),
        (discretize.TensorMesh([8, 9]), dc.Simulation3DNodal, [0.5, 1]),
        (discretize.TensorMesh([8, 9]), dc.Simulation3DCellCentered, [0.5, 1]),
        (discretize.TensorMesh([8, 9]), dc.Simulation2DNodal, [0.5, 1]),
        (discretize.TensorMesh([8, 9]), dc.Simulation2DCellCentered, [0.5, 1]),
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

    print(bf[sim1.surface_faces])
    print(bf[sim2.surface_faces])

    np.testing.assert_equal(f1[src, "phiSolution"], f2[src, "phiSolution"])
