"""
Fields requested for several sources at once match the fields of each source.
"""

import numpy as np
import discretize

from simpeg.electromagnetics import frequency_domain as fdem
from simpeg.utils import get_default_solver


def test_e_formulation_b_for_several_sources():
    mesh = discretize.TensorMesh([np.full(8, 50.0)] * 3, "CCC")
    # different frequencies, so each source's column has its own scaling
    sources = [
        fdem.sources.MagDipole([], frequency=f, location=np.r_[0.0, 0.0, 10.0])
        for f in (1.0, 10.0, 100.0)
    ]
    sim = fdem.Simulation3DElectricField(
        mesh,
        survey=fdem.Survey(sources),
        sigma=1e-2,
        solver=get_default_solver(),
    )
    f = sim.fields()
    b_all = f[sources, "b"]
    for i, src in enumerate(sources):
        np.testing.assert_allclose(b_all[:, i], f[src, "b"][:, 0])
