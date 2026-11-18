"""
The magnetic field simulations accept a scalar resistivity (or conductivity).
"""

import numpy as np
import pytest
import discretize

from simpeg.electromagnetics import natural_source as nsem
from simpeg.utils import get_default_solver


@pytest.mark.parametrize("dim", [1, 2])
@pytest.mark.parametrize("prop", ["rho", "sigma"])
def test_scalar_matches_array(dim, prop):
    if dim == 1:
        mesh = discretize.TensorMesh([np.full(30, 100.0)], "N")
        sim_class = nsem.Simulation1DMagneticField
        rx = nsem.receivers.Impedance([[0.0]], orientation="xy", component="app_res")
    elif dim == 2:
        mesh = discretize.TensorMesh([np.full(10, 100.0), np.full(30, 100.0)], "CN")
        sim_class = nsem.Simulation2DMagneticField
        rx = nsem.receivers.Impedance(
            np.array([[0.0, 0.0]]), orientation="yx", component="app_res"
        )
    else:  # pragma: nocover
        raise ValueError()
    value = 100.0 if prop == "rho" else 0.01
    sources = [nsem.sources.Planewave([rx], f) for f in (1.0, 10.0)]
    d = [
        sim_class(
            mesh,
            survey=nsem.Survey(sources),
            solver=get_default_solver(),
            **{prop: v},
        ).dpred()
        for v in (value, np.full(mesh.n_cells, value))
    ]
    np.testing.assert_allclose(d[0], d[1])
