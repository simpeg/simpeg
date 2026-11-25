from simpeg.electromagnetics.natural_source.receivers import ApparentConductivity
from discretize import TensorMesh
import numpy as np
import random


def test_locations_order():
    # Generate a simple 3D tensor mesh
    mesh = TensorMesh([np.full(10, 1.0)] * 3, origin=(-10, -10, -10))

    # Random locations between -10 and 10
    locations_e = random.sample(range(mesh.nC), 10)

    # Random locations between -10 and 10
    locations_h = random.sample(range(mesh.nC), 10)
    receivers = ApparentConductivity(
        locations_e=mesh.cell_centers[locations_e],
        locations_h=mesh.cell_centers[locations_h],
    )

    # Check the 0 projections refer to the expected locations
    proj_h = receivers.getP(mesh, "cell_centers", 0)
    active_inds = proj_h.data == 1

    np.testing.assert_allclose(proj_h.indices[active_inds], locations_h)

    # Check the 1 projections refer to the expected locations
    proj_e = receivers.getP(mesh, "cell_centers", 1)
    active_inds = proj_e.data == 1

    np.testing.assert_allclose(proj_e.indices[active_inds], locations_e)
