"""
Additional tests to the magnetic simulation
"""
import pytest

from discretize import TensorMesh
from SimPEG.potential_fields import magnetics


@pytest.fixture
def tensor_mesh():
    """
    Return sample TensorMesh
    """
    h = (3, 3, 3)
    return TensorMesh(h)


class TestMagSimulation:
    def test_choclo_and_n_proceesses(self, tensor_mesh):
        """Check if warning is raised after passing n_processes with choclo engine."""
        msg = "The 'n_processes' will be ignored when selecting 'choclo'"
        with pytest.warns(UserWarning, match=msg):
            simulation = magnetics.Simulation3DIntegral(
                tensor_mesh,
                engine="choclo",
                n_processes=2,
            )
        # Check if n_processes was overwritten and set to None
        assert simulation.n_processes is None
