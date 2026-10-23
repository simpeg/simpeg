import numpy as np
import pytest

from simpeg import maps
from simpeg.simulation import LinearSimulation


class LinearSimulationWithJtJ(LinearSimulation):
    def getJtJdiag(self, m, W=None, f=None):
        J = self.getJ(m)
        if W is not None:
            J = W @ J
        return np.sum(np.asarray(J) ** 2, axis=0)


@pytest.fixture
def linear_sims_factory():
    """Build identical lists of cheap simulations and mappings for a meta simulation.

    The Jacobians depend on the model (through an ExpMap), so cached values that are
    not cleared when the model changes show up as wrong answers.
    """

    def make(n_param=6, n_sim=2):
        rng = np.random.default_rng(0)
        sims = [
            LinearSimulationWithJtJ(
                G=rng.normal(size=(4, n_param)), linear_model=maps.ExpMap()
            )
            for _ in range(n_sim)
        ]
        mappings = [maps.IdentityMap(nP=n_param) for _ in range(n_sim)]
        return sims, mappings

    return make
