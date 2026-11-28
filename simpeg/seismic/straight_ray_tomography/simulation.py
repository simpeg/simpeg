import discretize
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

from ... import props
from ...simulation import LinearSimulation
from ...utils import sub2ind, validate_type


def _lengthInCell(O, D, xyz, plotIt=False):
    """Return the length of a transmitter-receiver segment inside one cell."""
    O = np.asarray(O, dtype=float)
    D = np.asarray(D, dtype=float)
    bounds = np.asarray(xyz, dtype=float)

    if O.ndim != 1 or D.shape != O.shape:
        raise ValueError("O and D must be one-dimensional arrays of equal length.")
    if O.size not in (2, 3) or bounds.shape != (O.size, 2):
        raise ValueError("Cell bounds must have shape (2, 2) or (3, 2).")

    maxD = np.linalg.norm(D)
    if maxD == 0.0:
        return None
    direction = D / maxD

    enter = 0.0
    exit = maxD
    for axis, (lower, upper) in enumerate(bounds):
        lower, upper = sorted((lower, upper))
        if direction[axis] == 0.0:
            if O[axis] < lower or O[axis] > upper:
                return None
            continue

        t0 = (lower - O[axis]) / direction[axis]
        t1 = (upper - O[axis]) / direction[axis]
        if t0 > t1:
            t0, t1 = t1, t0
        enter = max(enter, t0)
        exit = min(exit, t1)
        if exit <= enter:
            return None

    if plotIt:
        start = O + enter * direction
        end = O + exit * direction
        if O.size == 2:
            x, y = bounds
            plt.plot(
                [x[0], x[0], x[1], x[1], x[0]],
                [y[0], y[1], y[1], y[0], y[0]],
                "b",
            )
            plt.plot(O[0], O[1], "rs")
            plt.plot([start[0], end[0]], [start[1], end[1]], "r", lw=2)
        else:
            ax = plt.gca()
            if not hasattr(ax, "get_zlim"):
                ax = plt.figure().add_subplot(111, projection="3d")
            ax.scatter(*O, color="r")
            ax.plot(
                [start[0], end[0]],
                [start[1], end[1]],
                [start[2], end[2]],
                "r--",
                lw=2,
            )

    return exit - enter


def _lineintegral(M, Tx, Rx):
    """Return cell indices and segment lengths for a single ray."""
    if M.dim not in (2, 3):
        raise ValueError(
            f"Straight-ray tomography requires a 2D or 3D mesh, got {M.dim}D."
        )

    origin = np.asarray(Tx, dtype=float)
    direction = np.asarray(Rx, dtype=float) - origin
    axes = (M.nodes_x, M.nodes_y, M.nodes_z)[: M.dim]

    cell_indices = []
    lengths = []
    for cell_index in np.ndindex(M.shape_cells):
        cell_bounds = [
            nodes[index : index + 2] for nodes, index in zip(axes, cell_index)
        ]
        length = _lengthInCell(origin, direction, cell_bounds)
        if length is not None:
            cell_indices.append(cell_index)
            lengths.append(length)

    if not cell_indices:
        return np.empty(0, dtype=int), np.empty(0, dtype=float)

    inds = sub2ind(M.vnC, np.asarray(cell_indices, dtype=int))
    return inds, np.asarray(lengths)


class SimulationIntegral(LinearSimulation):
    """Straight-ray travel-time simulation on a 2D or 3D TensorMesh."""

    slowness, slownessMap, slownessDeriv = props.Invertible("Slowness model (1/v)")
    _allowed_mesh_dims = (2, 3)

    def __init__(self, mesh, survey=None, slowness=None, slownessMap=None, **kwargs):
        self.mesh = mesh
        super().__init__(survey=survey, **kwargs)
        self.slowness = slowness
        self.slownessMap = slownessMap

    @property
    def mesh(self):
        return self._mesh

    @mesh.setter
    def mesh(self, value):
        value = validate_type("mesh", value, discretize.TensorMesh, cast=False)
        if value.dim not in self._allowed_mesh_dims:
            dims = " or ".join(f"{dim}D" for dim in self._allowed_mesh_dims)
            raise ValueError(
                f"{type(self).__name__} mesh must be {dims}, received a "
                f"{value.dim}D mesh."
            )
        self._mesh = value

    @property
    def A(self):
        if getattr(self, "_A", None) is not None:
            return self._A
        self._A = sp.lil_matrix((self.survey.nD, self.mesh.nC))
        row = 0
        for src in self.survey.source_list:
            for rx in src.receiver_list:
                for loc_i in range(rx.locations.shape[0]):
                    inds, lengths = _lineintegral(
                        self.mesh, src.location, rx.locations[loc_i, :]
                    )
                    self._A[inds * 0 + row, inds] = lengths
                    row += 1
        return self._A

    def fields(self, m):
        self.model = m
        return self.A * self.slowness

    def Jvec(self, m, v, f=None):
        self.model = m
        return self.A * self.slownessDeriv * v

    def Jtvec(self, m, v, f=None):
        self.model = m
        return self.slownessDeriv.T * self.A.T * v


class Simulation2DIntegral(SimulationIntegral):
    """Backward-compatible 2D-only tomography simulation."""

    _allowed_mesh_dims = (2,)
