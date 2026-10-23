import warnings

import discretize
import numpy as np
import pytest

from simpeg import maps
from simpeg.electromagnetics.static import induced_polarization as ip
from simpeg.electromagnetics.static import resistivity as dc
from simpeg.electromagnetics.static import spectral_induced_polarization as sip


@pytest.mark.parametrize("kind", ["sigma", "rho"])
@pytest.mark.parametrize("sim_name", ["Simulation3DNodal", "Simulation3DCellCentered"])
def test_ip_construction_has_no_future_warning(sim_name, kind):
    mesh = discretize.TensorMesh([3, 3, 3])
    kwargs = {kind: np.ones(mesh.n_cells)}
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        sim = getattr(ip, sim_name)(
            mesh, survey=dc.Survey([]), eta=maps.IdentityMap(), **kwargs
        )
    np.testing.assert_allclose(getattr(sim, kind), 1.0)


@pytest.mark.parametrize("sim_name", ["Simulation3DNodal", "Simulation3DCellCentered"])
def test_sip_construction_has_no_future_warning(sim_name):
    mesh = discretize.TensorMesh([3, 3, 3])
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        sim = getattr(sip, sim_name)(
            mesh,
            survey=sip.Survey([]),
            sigma=np.ones(mesh.n_cells),
            eta=maps.IdentityMap(),
        )
    np.testing.assert_allclose(sim.sigma, 1.0)
