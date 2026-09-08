import discretize
import pytest
import simpeg.electromagnetics.viscous_remanent_magnetization as vrm


def test_mesh_required():
    with pytest.raises(
        TypeError, match=".*missing 1 required positional argument: 'mesh'"
    ):
        vrm.Simulation3DLinear()


def test_bad_mesh_type():
    mesh = discretize.CylindricalMesh([3, 3, 3])
    with pytest.raises(
        TypeError,
        match="mesh must be an instance of TensorMesh or TreeMesh, not CylindricalMesh",
    ):
        vrm.Simulation3DLinear(mesh)


def test_bad_mesh_dim():
    mesh = discretize.TensorMesh([3, 3])
    with pytest.raises(
        ValueError, match="Simulation3DLinear mesh must be 3D, received a 2D mesh."
    ):
        vrm.Simulation3DLinear(mesh)
