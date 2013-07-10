import numpy as np
from BaseMesh import BaseMesh
from TensorGrid import TensorGrid
from TensorView import TensorView


class TensorMesh(BaseMesh, TensorGrid, TensorView):
    """
    TensorMesh is a mesh class that deals with tensor product meshes.

    Any Mesh that has a constant width along the entire axis
    such that it can defined by a single width vector, called 'h'.

    e.g.

        hx = np.array([1,1,1])
        hy = np.array([1,2])
        hz = np.array([1,1,1,1])

        mesh = TensorMesh([hx, hy, hz])

    """
    def __init__(self, h, x0=None):
        super(TensorMesh, self).__init__(np.array([len(x) for x in h]), x0)

        assert len(h) == len(x0), "Dimension mismatch. x0 != len(h)"

        for i, h_i in enumerate(h):
            assert type(h_i) == np.ndarray, ("h[%i] is not a numpy array." % i)

        # Ensure h contains 1D vectors
        self._h = [x.ravel() for x in h]

    def h():
        doc = "h is a list containing the cell widths of the tensor mesh in each dimension."
        fget = lambda self: self._h
        return locals()
    h = property(**h())

    def hx():
        doc = "Width of cells in the x direction"
        fget = lambda self: self._h[0]
        return locals()
    hx = property(**hx())

    def hy():
        doc = "Width of cells in the y direction"
        fget = lambda self: None if self.dim < 2 else self._h[1]
        return locals()
    hy = property(**hy())

    def hz():
        doc = "Width of cells in the z direction"
        fget = lambda self: None if self.dim < 3 else self._h[2]
        return locals()
    hz = property(**hz())


if __name__ == '__main__':
    print('Welcome to tensor mesh!')

    testDim = 1
    h1 = 0.3*np.ones((1, 7))
    h1[:, 0] = 0.5
    h1[:, -1] = 0.6
    h2 = .5 * np.ones((1, 4))
    h3 = .4 * np.ones((1, 6))
    x0 = np.zeros((3, 1))

    if testDim == 1:
        h = [h1]
        x0 = x0[0]
    elif testDim == 2:
        h = [h1, h2]
        x0 = x0[0:2]
    else:
        h = [h1, h2, h3]

    I = np.linspace(0, 1, 8)
    M = TensorMesh(h, x0)

    xn = M.plotGrid()
