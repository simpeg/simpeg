import os

import discretize
import numpy as np
import warnings
from ..simulation import LinearSimulation
from scipy.sparse import csr_matrix as csr
from SimPEG.utils import mkvc
from ..utils import validate_string, validate_active_indices
import multiprocessing

###############################################################################
#                                                                             #
#                             Base Potential Fields Simulation                #
#                                                                             #
###############################################################################


class BasePFSimulation(LinearSimulation):
    r"""Base class for potential field simulations that use integral formulations.

    For integral formulations, the forward simulation for a set of voxel cells
    can be defined as a linear operation of the form:

    .. math::
        \mathbf{d} = \mathbf{Am}

    where :math:`\mathbf{d}` are the data, :math:`\mathbf{m}` are the model paramters
    and :math:`\mathbf{A}` is a linear operator defining the sensitivities. The primary
    difference between child simulation classes is the kernel function used to create
    the rows of :math:`\mathbf{A}`.

    Parameters
    ----------
    mesh : discretize.TensorMesh or discretize.TreeMesh
        A 3D tensor or tree mesh.
    ind_active : np.ndarray of int or bool
        Indices array denoting the active topography cells.
    store_sensitivities : {'ram', 'disk', 'forward_only'}
        Options for storing sensitivities. There are 3 options

        - 'ram': sensitivities are stored in the computer's RAM
        - 'disk': sensitivities are written to a directory
        - 'forward_only': you intend only do perform a forward simulation and sensitivities do no need to be stored

    """

    def __init__(self, mesh, ind_active=None, store_sensitivities="ram", **kwargs):

        # If deprecated property set with kwargs
        if "actInd" in kwargs:
            raise AttributeError(
                "actInd was removed in SimPEG 0.17.0, please use ind_active"
            )

        if "forwardOnly" in kwargs:
            raise AttributeError(
                "forwardOnly was removed in SimPEG 0.17.0, please set store_sensitivities='forward_only'"
            )

        self.store_sensitivities = store_sensitivities
        super().__init__(mesh, **kwargs)
        self.solver = None

        # Find non-zero cells indices
        if ind_active is not None:
            ind_active = validate_active_indices("ind_active", ind_active, mesh.n_cells)
        else:
            ind_active = np.ones(mesh.n_cells, dtype=bool)
        self._ind_active = ind_active

        self.nC = sum(ind_active)

        if isinstance(mesh, discretize.TensorMesh):
            nodes = mesh.nodes
            inds = np.arange(mesh.n_nodes).reshape(mesh.shape_nodes, order="F")
            if mesh.dim == 2:
                cell_nodes = [
                    inds[:-1, :-1].reshape(-1, order="F"),
                    inds[1:, :-1].reshape(-1, order="F"),
                    inds[:-1, 1:].reshape(-1, order="F"),
                    inds[1:, 1:].reshape(-1, order="F"),
                ]
            if mesh.dim == 3:
                cell_nodes = [
                    inds[:-1, :-1, :-1].reshape(-1, order="F"),
                    inds[1:, :-1, :-1].reshape(-1, order="F"),
                    inds[:-1, 1:, :-1].reshape(-1, order="F"),
                    inds[1:, 1:, :-1].reshape(-1, order="F"),
                    inds[:-1, :-1, 1:].reshape(-1, order="F"),
                    inds[1:, :-1, 1:].reshape(-1, order="F"),
                    inds[:-1, 1:, 1:].reshape(-1, order="F"),
                    inds[1:, 1:, 1:].reshape(-1, order="F"),
                ]
            cell_nodes = np.stack(cell_nodes, axis=-1)[ind_active]
        elif isinstance(mesh, discretize.TreeMesh):
            nodes = np.r_[mesh.nodes, mesh.hanging_nodes]
            cell_nodes = mesh.cell_nodes[ind_active]
        else:
            raise ValueError("Mesh must be 3D tensor or Octree.")
        unique, unique_inv = np.unique(cell_nodes.T, return_inverse=True)
        self._nodes = nodes[unique]  # unique active nodes
        self._unique_inv = unique_inv.reshape(cell_nodes.T.shape)

    @property
    def store_sensitivities(self):
        """Options for storing sensitivities.

        There are 3 options:

        - 'ram': sensitivity matrix stored in RAM
        - 'disk': sensitivities written and stored to disk
        - 'forward_only': sensitivities are not store (only use for forward simulation)

        Returns
        -------
        {'disk', 'ram', 'forward_only'}
            A string defining the model type for the simulation.
        """
        if self._store_sensitivities is None:
            self._store_sensitivities = "ram"
        return self._store_sensitivities

    @store_sensitivities.setter
    def store_sensitivities(self, value):
        self._store_sensitivities = validate_string(
            "store_sensitivities", value, ["disk", "ram", "forward_only"]
        )

    @property
    def ind_active(self):
        """Active topography cells

        Returns
        -------
        (n_cell) numpy.ndarray of bool
            Returns the active topography cells
        """
        return self._ind_active

    @property
    def actInd(self):
        """'actInd' is deprecated. Use 'ind_active' instead."""
        raise AttributeError(
            "The 'actInd' property has been deprecated. "
            "Please use 'ind_active'. This will be removed in version 0.17.0 of SimPEG.",
        )

    def linear_operator(self):
        """Return linear operator

        Returns
        -------
        numpy.ndarray
            Linear operator
        """
        n_cells = self.nC
        if getattr(self, "model_type", None) == "vector":
            n_cells *= 3
        if self.store_sensitivities == "disk":
            sens_name = os.path.join(self.sensitivity_path, "sensitivity.npy")
            if os.path.exists(sens_name):
                # do not pull array completely into ram, just need to check the size
                kernel = np.load(sens_name, mmap_mode="r")
                if kernel.shape == (self.survey.nD, n_cells):
                    print(f"Found sensitivity file at {sens_name} with expected shape")
                    kernel = np.asarray(kernel)
                    return kernel
        # multiprocessed
        with multiprocessing.pool.Pool() as pool:
            kernel = pool.starmap(
                self.evaluate_integral, self.survey._location_component_iterator()
            )
        if self.store_sensitivities != "forward_only":
            kernel = np.vstack(kernel)
        else:
            kernel = np.concatenate(kernel)
        if self.store_sensitivities == "disk":
            print(f"writing sensitivity to {sens_name}")
            os.makedirs(self.sensitivity_path, exist_ok=True)
            np.save(sens_name, kernel)
        return kernel


class BaseEquivalentSourceLayerSimulation(BasePFSimulation):
    """Base equivalent source layer simulation class

    Parameters
    ----------
    mesh : discretize.BaseMesh
        A 2D tensor or tree mesh defining discretization along the x and y directions
    cell_z_top : numpy.ndarray or float
        Define the elevations for the top face of all cells in the layer
    cell_z_bottom : numpy.ndarray or float
        Define the elevations for the bottom face of all cells in the layer

    """

    def __init__(self, mesh, cell_z_top, cell_z_bottom, **kwargs):

        if mesh.dim != 2:
            raise AttributeError("Mesh to equivalent source layer must be 2D.")

        super().__init__(mesh, **kwargs)

        if isinstance(cell_z_top, (int, float)):
            cell_z_top = float(cell_z_top) * np.ones(mesh.nC)

        if isinstance(cell_z_bottom, (int, float)):
            cell_z_bottom = float(cell_z_bottom) * np.ones(mesh.nC)

        if (mesh.nC != len(cell_z_top)) | (mesh.nC != len(cell_z_bottom)):
            raise AttributeError(
                "'cell_z_top' and 'cell_z_bottom' must have length equal to number of cells."
            )

        all_nodes = self._nodes[self._unique_inv]
        all_nodes = [
            np.c_[all_nodes[0], cell_z_bottom],
            np.c_[all_nodes[1], cell_z_bottom],
            np.c_[all_nodes[2], cell_z_bottom],
            np.c_[all_nodes[3], cell_z_bottom],
            np.c_[all_nodes[0], cell_z_top],
            np.c_[all_nodes[1], cell_z_top],
            np.c_[all_nodes[2], cell_z_top],
            np.c_[all_nodes[3], cell_z_top],
        ]
        self._nodes = np.stack(all_nodes, axis=0)
        self._unique_inv = None


def progress(iter, prog, final):
    """Progress (% complete) for constructing sensitivity matrix

    Parameters
    ----------
    iter : int
        Current rows
    prog : float
        Progress
    final : int
        Number of rows (= number of receivers)

    Returns
    -------
    float
        % completed
    """
    arg = np.floor(float(iter) / float(final) * 10.0)

    if arg > prog:

        print("Done " + str(arg * 10) + " %")
        prog = arg

    return prog


def get_dist_wgt(mesh, receiver_locations, actv, R, R0):
    """Compute distance weights for potential field simulations

    Parameters
    ----------
    mesh : discretize.BaseMesh
        A discretize mesh
    receiver_locations : (n, 3) numpy.ndarray
        Observation locations [x, y, z]
    actv : (n_cell) numpy.ndarray of bool
        Active cells vector [0:air , 1: ground]
    R : float
        Decay factor (mag=3, grav =2)
    R0 : float
        Stabilization factor. Usually a fraction of the minimum cell size

    Returns
    -------
    wr : (n_cell) numpy.ndarray
        Distance weighting model; 0 for all inactive cells
    """

    # Find non-zero cells
    if actv.dtype == "bool":
        inds = (
            np.asarray([inds for inds, elem in enumerate(actv, 1) if elem], dtype=int)
            - 1
        )
    else:
        inds = actv

    nC = len(inds)

    # Create active cell projector
    P = csr((np.ones(nC), (inds, range(nC))), shape=(mesh.nC, nC))

    # Geometrical constant
    p = 1 / np.sqrt(3)

    # Create cell center location
    Ym, Xm, Zm = np.meshgrid(
        mesh.cell_centers_y, mesh.cell_centers_x, mesh.cell_centers_z
    )
    hY, hX, hZ = np.meshgrid(mesh.h[1], mesh.h[0], mesh.h[2])

    # Remove air cells
    Xm = P.T * mkvc(Xm)
    Ym = P.T * mkvc(Ym)
    Zm = P.T * mkvc(Zm)

    hX = P.T * mkvc(hX)
    hY = P.T * mkvc(hY)
    hZ = P.T * mkvc(hZ)

    V = P.T * mkvc(mesh.cell_volumes)
    wr = np.zeros(nC)

    ndata = receiver_locations.shape[0]
    count = -1
    print("Begin calculation of distance weighting for R= " + str(R))

    for dd in range(ndata):

        nx1 = (Xm - hX * p - receiver_locations[dd, 0]) ** 2
        nx2 = (Xm + hX * p - receiver_locations[dd, 0]) ** 2

        ny1 = (Ym - hY * p - receiver_locations[dd, 1]) ** 2
        ny2 = (Ym + hY * p - receiver_locations[dd, 1]) ** 2

        nz1 = (Zm - hZ * p - receiver_locations[dd, 2]) ** 2
        nz2 = (Zm + hZ * p - receiver_locations[dd, 2]) ** 2

        R1 = np.sqrt(nx1 + ny1 + nz1)
        R2 = np.sqrt(nx1 + ny1 + nz2)
        R3 = np.sqrt(nx2 + ny1 + nz1)
        R4 = np.sqrt(nx2 + ny1 + nz2)
        R5 = np.sqrt(nx1 + ny2 + nz1)
        R6 = np.sqrt(nx1 + ny2 + nz2)
        R7 = np.sqrt(nx2 + ny2 + nz1)
        R8 = np.sqrt(nx2 + ny2 + nz2)

        temp = (
            (R1 + R0) ** -R
            + (R2 + R0) ** -R
            + (R3 + R0) ** -R
            + (R4 + R0) ** -R
            + (R5 + R0) ** -R
            + (R6 + R0) ** -R
            + (R7 + R0) ** -R
            + (R8 + R0) ** -R
        )

        wr = wr + (V * temp / 8.0) ** 2.0

        count = progress(dd, count, ndata)

    wr = np.sqrt(wr) / V
    wr = mkvc(wr)
    wr = np.sqrt(wr / (np.max(wr)))

    print("Done 100% ...distance weighting completed!!\n")

    return wr
