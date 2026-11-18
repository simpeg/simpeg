"""
An array of permeabilities without a map gives the same fields as with a map.
"""

import numpy as np
import discretize
from scipy.constants import mu_0

from simpeg import maps
from simpeg.electromagnetics import natural_source as nsem
from simpeg.utils import get_default_solver


def test_1d_electric_field_mu_array():
    mesh = discretize.TensorMesh([[(50.0, 10, -1.3), (50.0, 40), (50.0, 10, 1.3)]], "C")
    sigma = np.where(mesh.cell_centers < 0, 1e-2, 1e-8)
    mu = np.where(mesh.cell_centers < -500, 2 * mu_0, mu_0)
    rx = nsem.receivers.Impedance([[0.0]], orientation="yx", component="app_res")
    sources = [nsem.sources.Planewave([rx], f) for f in (1.0, 10.0)]

    fields = []
    for kwargs, model in (
        ({"mu": mu}, None),
        ({"muMap": maps.IdentityMap(mesh)}, mu),
    ):
        sim = nsem.Simulation1DElectricField(
            mesh,
            survey=nsem.Survey(sources),
            sigma=sigma,
            solver=get_default_solver(),
            **kwargs,
        )
        fields.append(sim.fields(model))
    for src in sources:
        h_array, h_map = fields[0][src, "h"], fields[1][src, "h"]
        assert h_array.shape == (mesh.n_cells, 1)
        np.testing.assert_allclose(h_array, h_map)
