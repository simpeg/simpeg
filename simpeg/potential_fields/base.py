import os
from multiprocessing.pool import Pool

import discretize
import numpy as np
from discretize import TensorMesh, TreeMesh

from ..simulation import LinearSimulation
from ..utils import validate_active_indices, validate_integer, validate_string
from ..utils.code_utils import deprecate_property, validate_type

try:
    import choclo
except ImportError:
    choclo = None

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
    active_cells : np.ndarray of int or bool
        Indices array denoting the active topography cells.
    store_sensitivities : {'ram', 'disk', 'forward_only'}
        Options for storing sensitivities. There are 3 options

        - 'ram': sensitivities are stored in the computer's RAM
        - 'disk': sensitivities are written to a directory
        - 'forward_only': you intend only do perform a forward simulation and sensitivities do not need to be stored

    n_processes : None or int, optional
        The number of processes to use in the internal multiprocessing pool for forward
        modeling. The default value of 1 will not use multiprocessing. Any other setting
        will. `None` implies setting by the number of cpus. If engine is
        ``"choclo"``, then this argument will be ignored.
    engine : {"geoana", "choclo"}, optional
       Choose which engine should be used to run the forward model.
    numba_parallel : bool, optional
        If True, the simulation will run in parallel. If False, it will
        run in serial. If ``engine`` is not ``"choclo"`` this argument will be
        ignored.

    Notes
    -----
    If using multiprocessing by setting `n_processes` to a value other than 1, you must
    be aware of the method your operating system uses to spawn the subprocesses. On
    Windows the default method starts new processes that all import the main script.
    Therefor you must protect the calls to this class by testing if you are in
    the main process with:

    >>> from simpeg.potential_fields import gravity
    >>> if __name__ == '__main__':
    ...     # Do your processing here
    ...     sim = gravity.Simulation3DIntegral(n_processes=4, ...)
    ...     sim.dpred(m)

    This usually does not affect jupyter notebook environments.
    """

    def __init__(
        self,
        mesh,
        active_cells=None,
        store_sensitivities="ram",
        n_processes=1,
        sensitivity_dtype=np.float32,
        engine="geoana",
        numba_parallel=True,
        **kwargs,
    ):
        # removed ind_active argument
        if kwargs.pop("ind_active", None) is not None:
            raise TypeError(
                "'ind_active' has been removed in "
                "SimPEG v0.24.0, please use 'active_cells' instead.",
            )

        self.mesh = mesh
        self.store_sensitivities = store_sensitivities
        self.sensitivity_dtype = sensitivity_dtype
        self.engine = engine
        self.numba_parallel = numba_parallel
        super().__init__(**kwargs)
        self.n_processes = n_processes

        # Check sensitivity_path when engine is "choclo"
        self._check_engine_and_sensitivity_path()

        # Find non-zero cells indices
        if active_cells is None:
            active_cells = np.ones(mesh.n_cells, dtype=bool)
        else:
            active_cells = validate_active_indices(
                "active_cells", active_cells, mesh.n_cells
            )
        self._active_cells = active_cells

        self.nC = int(sum(active_cells))

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
            cell_nodes = np.stack(cell_nodes, axis=-1)[active_cells]
        elif isinstance(mesh, discretize.TreeMesh):
            nodes = np.r_[mesh.nodes, mesh.hanging_nodes]
            cell_nodes = mesh.cell_nodes[active_cells]
        else:
            raise ValueError("Mesh must be 3D tensor or Octree.")
        unique, unique_inv = np.unique(cell_nodes.T, return_inverse=True)
        self._nodes = nodes[unique]  # unique active nodes
        self._unique_inv = unique_inv.reshape(cell_nodes.T.shape)

    @property
    def mesh(self):
        """Mesh for the integral potential field simulations.

        Returns
        -------
        discretize.TensorMesh or discretize.TreeMesh
            3D Mesh on which the forward problem is discretized.
        """
        return self._mesh

    @mesh.setter
    def mesh(self, value):
        value = validate_type("mesh", value, (TensorMesh, TreeMesh), cast=False)
        if value.dim != 3:
            raise ValueError(
                f"{type(self).__name__} mesh must be 3D, received a {value.dim}D mesh."
            )
        self._mesh = value

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
    def sensitivity_dtype(self):
        """dtype of the sensitivity matrix.

        Returns
        -------
        numpy.float32 or numpy.float64
            The dtype used to store the sensitivity matrix
        """
        if self.store_sensitivities == "forward_only":
            return np.float64
        return self._sensitivity_dtype

    @sensitivity_dtype.setter
    def sensitivity_dtype(self, value):
        if value is not np.float32 and value is not np.float64:
            raise TypeError(
                "sensitivity_dtype must be either np.float32 or np.float64."
            )
        self._sensitivity_dtype = value

    @property
    def n_processes(self):
        """
        Number of processes to use for forward modeling.

        If ``engine`` is ``"choclo"``, then this property will be ignored.
        """
        return self._n_processes

    @n_processes.setter
    def n_processes(self, value):
        if value is not None:
            value = validate_integer("n_processes", value, min_val=1)
        self._n_processes = value

    @property
    def engine(self) -> str:
        """
        Engine that will be used to run the simulation.

        It can be either ``"geoana"`` or "``choclo``".
        """
        return self._engine

    @engine.setter
    def engine(self, value: str):
        validate_string(
            "engine", value, string_list=("geoana", "choclo"), case_sensitive=True
        )
        if value == "choclo" and choclo is None:
            raise ImportError(
                "The choclo package couldn't be found."
                "Running a gravity simulation with 'engine=\"choclo\"' needs "
                "choclo to be installed."
                "\nTry installing choclo with:"
                "\n    pip install choclo"
                "\nor:"
                "\n    conda install choclo"
            )
        self._engine = value

    @property
    def numba_parallel(self) -> bool:
        """
        Run simulation in parallel or single-threaded when using Numba.

        If True, the simulation will run in parallel. If False, it will
        run in serial.

        .. important::

            If ``engine`` is not ``"choclo"`` this property will be ignored.
        """
        return self._numba_parallel

    @numba_parallel.setter
    def numba_parallel(self, value: bool):
        if not isinstance(value, bool):
            raise TypeError(
                f"Invalid 'numba_parallel' value of type {type(value)}. Must be a bool."
            )
        self._numba_parallel = value

    @property
    def active_cells(self):
        """Active cells in the mesh.

        Returns
        -------
        (n_cell) numpy.ndarray of bool
            Returns the active cells in the mesh.
        """
        return self._active_cells

    ind_active = deprecate_property(
        active_cells,
        "ind_active",
        "active_cells",
        removal_version="0.24.0",
        error=True,
    )

    def linear_operator(self):
        """Return linear operator.

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
        if self.store_sensitivities == "forward_only":
            kernel_shape = (self.survey.nD,)
        else:
            kernel_shape = (self.survey.nD, n_cells)
        dtype = self.sensitivity_dtype
        kernel = np.empty(kernel_shape, dtype=dtype)
        if self.n_processes == 1:
            id0 = 0
            for args in self.survey._location_component_iterator():
                rows = self.evaluate_integral(*args)
                n_c = rows.shape[0]
                id1 = id0 + n_c
                kernel[id0:id1] = rows.astype(dtype, copy=False)
                id0 = id1
        else:
            # multiprocessed
            with Pool(processes=self.n_processes) as pool:
                id0 = 0
                for rows in pool.starmap(
                    self.evaluate_integral, self.survey._location_component_iterator()
                ):
                    n_c = rows.shape[0]
                    id1 = id0 + n_c
                    kernel[id0:id1] = rows.astype(dtype, copy=False)
                    id0 = id1

        # if self.store_sensitivities != "forward_only":
        #     kernel = np.vstack(kernel)
        # else:
        #     kernel = np.concatenate(kernel)
        if self.store_sensitivities == "disk":
            print(f"writing sensitivity to {sens_name}")
            os.makedirs(self.sensitivity_path, exist_ok=True)
            np.save(sens_name, kernel)
        return kernel

    def _check_engine_and_sensitivity_path(self):
        """
        Check if sensitivity_path is a file if engine is set to "choclo"
        """
        if (
            self.engine == "choclo"
            and self.store_sensitivities == "disk"
            and os.path.isdir(self.sensitivity_path)
        ):
            raise ValueError(
                f"The passed sensitivity_path '{self.sensitivity_path}' is "
                "a directory. "
                "When using 'choclo' as the engine, 'senstivity_path' "
                "should be the path to a new or existing file."
            )

    def _get_active_nodes(self):
        """
        Return locations of nodes only for active cells

        Also return an array containing the indices of the "active nodes" for
        each active cell in the mesh
        """
        # Get all nodes in the mesh
        if isinstance(self.mesh, discretize.TreeMesh):
            nodes = self.mesh.total_nodes
        elif isinstance(self.mesh, discretize.TensorMesh):
            nodes = self.mesh.nodes
        else:
            raise TypeError(f"Invalid mesh of type {self.mesh.__class__.__name__}.")
        # Get original cell_nodes
        cell_nodes = self.mesh.cell_nodes
        # If all cells in the mesh are active, return nodes and cell_nodes
        if self.nC == self.mesh.n_cells:
            return nodes, cell_nodes
        # Keep only the cell_nodes for active cells
        cell_nodes = cell_nodes[self.active_cells]
        # Get the unique indices of the nodes that belong to every active cell
        # (these indices correspond to the original `nodes` array)
        unique_nodes, active_cell_nodes = np.unique(cell_nodes, return_inverse=True)
        # Select only the nodes that belong to the active cells (active nodes)
        active_nodes = nodes[unique_nodes]
        # Reshape indices of active cell nodes for each active cell in the mesh
        active_cell_nodes = active_cell_nodes.reshape(cell_nodes.shape)
        return active_nodes, active_cell_nodes

    def _get_components_and_receivers(self):
        """Generator for receiver locations and their field components."""
        if not hasattr(self.survey, "source_field"):
            raise AttributeError(
                f"The survey '{self.survey}' has no 'source_field' attribute."
            )
        for receiver_object in self.survey.source_field.receiver_list:
            yield receiver_object.components, receiver_object.locations


class BaseEquivalentSourceLayerSimulation(BasePFSimulation):
    """Base equivalent source layer simulation class.

    Parameters
    ----------
    mesh : discretize.BaseMesh
        A 2D tensor or tree mesh defining discretization along the x and y directions
    cell_z_top : numpy.ndarray or float
        Define the elevations for the top face of all cells in the layer. If an array,
        it should be the same size as the active cell set.
    cell_z_bottom : numpy.ndarray or float
        Define the elevations for the bottom face of all cells in the layer. If an array,
        it should be the same size as the active cell set.
    """

    def __init__(self, mesh, cell_z_top, cell_z_bottom, **kwargs):
        super().__init__(mesh=mesh, **kwargs)

        if isinstance(cell_z_top, (int, float)):
            cell_z_top = float(cell_z_top) * np.ones(self.nC)

        if isinstance(cell_z_bottom, (int, float)):
            cell_z_bottom = float(cell_z_bottom) * np.ones(self.nC)

        if (self.nC != len(cell_z_top)) | (self.nC != len(cell_z_bottom)):
            raise AttributeError(
                "'cell_z_top' and 'cell_z_bottom' must have length equal to number of",
                "cells, and match the number of active cells.",
            )

        self._cell_z_top, self._cell_z_bottom = cell_z_top, cell_z_bottom

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

    @property
    def cell_z_top(self) -> np.ndarray:
        """
        Elevations for the top face of all cells in the layer.
        """
        return self._cell_z_top

    @property
    def cell_z_bottom(self) -> np.ndarray:
        """
        Elevations for the bottom face of all cells in the layer.
        """
        return self._cell_z_bottom

    @BasePFSimulation.mesh.setter
    def mesh(self, value):
        value = validate_type("mesh", value, (TensorMesh, TreeMesh), cast=False)
        if value.dim != 2:
            raise ValueError(
                f"{type(self).__name__} mesh must be 2D, received a {value.dim}D mesh."
            )
        self._mesh = value


def progress(iteration, prog, final):
    """Progress (% complete) for constructing sensitivity matrix.

    Parameters
    ----------
    iteration : int
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
    arg = np.floor(float(iteration) / float(final) * 10.0)

    if arg > prog:
        print("Done " + str(arg * 10) + " %")
        prog = arg

    return prog


def get_dist_wgt(*args, **kwargs):
    raise NotImplementedError(
        "The get_dist_wgt function has been removed in SimPEG 0.24.0, please import simpeg.utils.distance_weighting."
    )
