import warnings
from typing import List, Optional

import discretize
from SimPEG import dask
import numpy as np
from discretize.utils import mesh_builder_xyz, refine_tree_xyz, active_from_xyz
from discretize import TreeMesh
from SimPEG.maps import TileMap
from SimPEG import data, data_misfit, maps, simulation, survey
from scipy.spatial import Delaunay, cKDTree, qhull
from pymatsolver import Solver


def create_local_misfit(
    simulation_class: simulation.BaseSimulation,
    local_survey: survey.BaseSurvey,
    global_mesh: TreeMesh,
    global_active: np.ndarray,
    tile_id: int,
    tile_buffer: float = 100,
    min_level: int = 4,
    workers: Optional[List] = None,
    solver: Optional[Solver] = None,
) -> data_misfit.BaseDataMisfit:
    """
    Create a nested mesh and misfit function for tiled simulation.

    :param simulation_class: A :obj:`SimPEG.simulation` class type.
    :param local_survey: Survey object used over the local tile.
    :param global_mesh: Tree mesh defining the global inversion.
    :param global_active: shape(global_mesh.nC,)
        Array of bool defining the ground cells.
    :param tile_id: Unique identifier for the tile.
    :param tile_buffer: Horizontal distance around the local_survey to
        preserve fine cells.
    :param min_level: Highest octree level used outside the tile_buffer distance
        (min_level = 0: finest).
    :param workers: List of workers IP addresses to handle the local simulation.
    :param solver: Solver to be used by the simulation

    :return local_misfit: A local misfit
    """
    local_mesh = create_nested_mesh(
        local_survey.unique_locations,
        global_mesh,
        method="radial",
        max_distance=tile_buffer,
        pad_distance=tile_buffer * 2,
        min_level=min_level,
    )

    local_map = maps.TileMap(global_mesh, global_active, local_mesh)
    local_active = local_map.local_active

    actmap = maps.InjectActiveCells(
        local_mesh, indActive=local_active, valInactive=np.log(1e-8)
    )
    expmap = maps.ExpMap(local_mesh)
    mapping = expmap * actmap

    # Create the local misfit
    simulation = simulation_class(
        local_mesh,
        survey=local_survey,
        sigmaMap=mapping,
        Solver=solver,
        workers=workers
    )
    simulation.sensitivity_path = "./sensitivity/Tile" + str(tile_id) + "/"
    data_object = data.Data(
        local_survey,
        dobs=local_survey.dobs,
        standard_deviation=local_survey.std,
    )
    local_misfit = data_misfit.L2DataMisfit(
        data=data_object, simulation=simulation, model_map=local_map
    )
    local_misfit.W = 1.0 / local_survey.std

    return local_misfit

def create_tile_meshes(
    locations,
    topography,
    indices,
    base_mesh=None,
    max_distance=np.inf,
    core_cells=[10, 10, 10],
    locations_refinement=[5, 5, 5],
    topography_refinement=[0, 0, 2],
    padding_distance=[[0, 0], [0, 0], [0, 0]],
):

    assert isinstance(indices, list), "'indices' must be a list of integers"

    if len(padding_distance) == 1:
        padding_distance = [padding_distance*2]*3

    global_mesh = mesh_builder_xyz(
        locations, core_cells,
        padding_distance=padding_distance,
        mesh_type='TREE', base_mesh=base_mesh,
        depth_core=1000
    )
    local_meshes = []
    for ind in indices:
        local_mesh = mesh_builder_xyz(
            locations, core_cells,
            padding_distance=padding_distance,
            mesh_type='TREE', base_mesh=base_mesh,
            depth_core=1000
        )
        local_mesh = refine_tree_xyz(
            local_mesh, topography,
            method='surface', octree_levels=topography_refinement,
            finalize=False
        )
        local_mesh = refine_tree_xyz(
            local_mesh, locations[ind],
            method='surface', octree_levels=locations_refinement,
            max_distance=max_distance,
            finalize=True
        )
        global_mesh.insert_cells(
            local_mesh.gridCC,
            local_mesh.cell_levels_by_index(np.arange(local_mesh.nC)),
            finalize=False,
        )

        local_meshes.append(local_mesh)

    global_mesh.finalize()
    global_active = active_from_xyz(global_mesh, topography, method='linear')

    # Cycle back to all local meshes and create tile maps
    local_maps = []
    for mesh in local_meshes:
        local_maps.append(
            TileMap(global_mesh, global_active, mesh)
        )

    return (global_mesh, global_active), (local_meshes, local_maps)


def create_nested_mesh(
        locations: np.ndarray,
        base_mesh: TreeMesh,
        method: str = "padding_cells",
        padding_cells: int = 8,
        minimum_level: int = 3,
        finalize: bool = True
):
    """
    Create a nested mesh with the same extent as the input global mesh.
    Refinement levels are preserved only around the input locations (local survey).

    Parameters
    ----------

    locations: Array of coordinates for the local survey shape(*, 3).
    base_mesh: Input global TreeMesh object.
    method: Refinement of cells from the base mesh determined by from either
        'convex_hull': Cells that fall inside a 2D Delaunay triangulation of the input locations.
        'padding_cells':  Cells inside concentric shells made of 'padding_cells'
    padding_cells: Used for 'method'= 'padding_cells'. Number of cells in each concentric shell.
    minimum_level: Minimum octree level to preserve everywhere outside the local survey area.
    finalize: Return a finalized local treemesh.
    """
    nested_mesh = TreeMesh(
        [base_mesh.h[0], base_mesh.h[1], base_mesh.h[2]], x0=base_mesh.x0
    )
    base_level = (base_mesh.max_level - minimum_level)
    base_refinement = base_mesh.cell_levels_by_index(np.arange(base_mesh.nC))
    base_refinement[base_refinement > base_level] = base_level
    nested_mesh.insert_cells(
        base_mesh.gridCC,
        base_refinement,
        finalize=False,
    )

    if method == "convex_hull":
        # Find cells inside the data extant
        try:
            tri2D = Delaunay(locations[:, :2])
            indices = tri2D.find_simplex(base_mesh.gridCC[:, :2]) != -1
            return nested_mesh.insert_cells(
                base_mesh.gridCC[indices, :],
                base_mesh.cell_levels_by_index(np.where(indices)[0]),
                finalize=finalize,
            )
        except qhull.QhullError:
            warnings.warn("qhull failed to triangulate. Defaulting to 'padding_cells' method.")
            pass
    elif method != "padding_cells":
        raise ValueError("Input 'method' must be one of 'convex_hull' or 'padding_cells'.")

    base_cell = np.min([base_mesh.h[0][0], base_mesh.h[1][0]])
    tree = cKDTree(locations[:, :2])
    center = np.mean(base_mesh.gridCC[:, :2], axis=0)
    stretched_cc = (
        (base_mesh.gridCC[:, :2] - center)
        * (base_cell / np.r_[base_mesh.h[0][0], base_mesh.h[1][0]])
    ) + center
    rad, _ = tree.query(stretched_cc)
    pad_distance = 0.
    for ii in range(minimum_level):
        pad_distance += base_cell * 2 ** ii * padding_cells
        indices = np.where(rad < pad_distance)[0]
        # indices = np.where(tri2D.find_simplex(base_mesh.gridCC[:, :2]) != -1)[0]
        levels = base_mesh.cell_levels_by_index(indices)
        levels[levels > (base_mesh.max_level - ii)] = base_mesh.max_level - ii
        nested_mesh.insert_cells(
            base_mesh.gridCC[indices, :],
            levels,
            finalize=False,
        )

    if finalize:
        nested_mesh.finalize()

    return nested_mesh