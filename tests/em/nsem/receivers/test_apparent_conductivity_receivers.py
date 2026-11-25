import numpy as np
from discretize import TensorMesh

from simpeg.electromagnetics.natural_source.receivers import ApparentConductivity


def test_locations_order():
    """Magnetic fields project from location 0, electric fields from location 1."""
    mesh = TensorMesh([np.full(10, 1.0)] * 3, origin=(-10, -10, -10))

    rng = np.random.default_rng(0)
    locations_e = mesh.cell_centers[rng.choice(mesh.n_cells, 10, replace=False)]
    locations_h = mesh.cell_centers[rng.choice(mesh.n_cells, 10, replace=False)]

    rx = ApparentConductivity(locations_e=locations_e, locations_h=locations_h)

    # interpolating the cell center coordinates recovers the projected locations
    P_h = rx.getP(mesh, "cell_centers", 0)
    np.testing.assert_allclose(P_h @ mesh.cell_centers, locations_h)

    P_e = rx.getP(mesh, "cell_centers", 1)
    np.testing.assert_allclose(P_e @ mesh.cell_centers, locations_e)

    np.testing.assert_array_equal(rx.locations_e, locations_e)
    np.testing.assert_array_equal(rx.locations_h, locations_h)
