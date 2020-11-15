import numpy as np
from discretize.utils import mesh_builder_xyz, refine_tree_xyz
from SimPEG.utils import surface2ind_topo
from SimPEG.maps import TileMap


def create_tile_meshes(
    locations,
    topography,
    indices,
    base_mesh=None,
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
            finalize=True
        )
        global_mesh.insert_cells(
            local_mesh.gridCC,
            local_mesh.cell_levels_by_index(np.arange(local_mesh.nC)),
            finalize=False,
        )

        local_meshes.append(local_mesh)

    global_mesh.finalize()
    global_active = surface2ind_topo(global_mesh, topography, method='linear')

    # Cycle back to all local meshes and create tile maps
    local_maps = []
    for mesh in local_meshes:
        local_maps.append(
            TileMap(global_mesh, global_active, mesh)
        )

    return (global_mesh, global_active), (local_meshes, local_maps)
