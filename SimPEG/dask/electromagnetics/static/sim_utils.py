from .... import maps, utils
import discretize
from discretize.utils import mesh_builder_xyz, refine_tree_xyz, mkvc
from ....electromagnetics import resistivity as dc
from ....electromagnetics.static.utils import StaticUtils as dcutils
from pymatsolver import Pardiso as Solver
import numpy as np


# def create_tile_meshes(source_list,
#                            global_electrodes,
#                            dem,
#                            base_mesh,
#                            oct_levels=[5, 3, 2, 2],
#                            remote_quadrant=0,
#                            output_meshes=False,
#                            depth_core=1000,
#                            padLen=800,
#                            remote_present=True):
#
#     # combine topo
#     topography = np.vstack((dem, global_electrodes))
#     padding_distance = np.r_[np.c_[padLen, padLen],
#                                  np.c_[padLen, padLen], np.c_[padLen, padLen]]
#     # creating mesh using Discretize!
#     h = [base_mesh.h[0].min(), base_mesh.h[1].min(), base_mesh.h[2].min()]
#     # create global mesh
#     global_mesh = mesh_builder_xyz(
#         global_electrodes, h,
#         padding_distance=padding_distance,
#         mesh_type='TREE', base_mesh=base_mesh,
#         depth_core=depth_core
#     )
#     local_meshes = []
#
#     for source in source_list:
#         sub_survey = dc.Survey([source])
#         electrodes = np.vstack((sub_survey.locations_a,
#                                 sub_survey.locations_b,
#                                 sub_survey.locations_m,
#                                 sub_survey.locations_n))
#
#         flag_remote = electrodes[:, 1] > electrodes[:, 1].min()
#         remote_only = electrodes[:, 1] <= electrodes[:, 1].min()
#
#         # figure remote location so discretization is good there
#         if remote_quadrant == 1:
#             flag_remote = electrodes[:, 1] > electrodes[:, 1].max()
#             remote_only = electrodes[:, 1] <= electrodes[:, 1].max()
#         elif remote_quadrant == 2:
#             flag_remote = electrodes[:, 0] > electrodes[:, 0].max()
#             remote_only = electrodes[:, 0] <= electrodes[:, 0].max()
#         elif remote_quadrant == 3:
#             flag_remote = electrodes[:, 1] > electrodes[:, 1].min()
#             remote_only = electrodes[:, 1] <= electrodes[:, 1].min()
#         elif remote_quadrant == 4:
#             flag_remote = electrodes[:, 1] > electrodes[:, 1].min()
#             remote_only = electrodes[:, 1] <= electrodes[:, 1].min()
#
#         # create databases
#         no_remote_db = electrodes[flag_remote]
#         remote_db = electrodes[remote_only]
#
#
#         padding_distance = np.r_[np.c_[padLen, padLen],
#                                  np.c_[padLen, padLen], np.c_[padLen, padLen]]
#
#         local_mesh = discretize.utils.mesh_builder_xyz(
#             global_electrodes, h,
#             padding_distance=padding_distance,
#             mesh_type='TREE', base_mesh=base_mesh,
#             depth_core=depth_core
#         )
#         local_mesh = discretize.utils.refine_tree_xyz(
#             local_mesh, topography,
#             method='surface', octree_levels=[1, 1, 0],
#             finalize=False
#         )
#
#         if remote_present:
#             local_mesh = discretize.utils.refine_tree_xyz(
#                 local_mesh, remote_db,
#                 method='radial', octree_levels=[2, 1, 1],
#                 finalize=False
#             )
#             local_mesh = discretize.utils.refine_tree_xyz(
#                 local_mesh, no_remote_db,
#                 method='surface', octree_levels=oct_levels,
#                 # octree_levels_padding=[4, 4, 2, 2],
#                 finalize=True
#             )
#         else:
#             local_mesh = discretize.utils.refine_tree_xyz(
#                 local_mesh, electrodes,
#                 method='surface', octree_levels=oct_levels,
#                 # octree_levels_padding=[4, 4, 2, 2],
#                 finalize=True
#             )
#         global_mesh.insert_cells(
#             local_mesh.gridCC,
#             local_mesh.cell_levels_by_index(np.arange(local_mesh.nC)),
#             finalize=False,
#         )
#
#         local_meshes.append(local_mesh)
#
#         if output_meshes:
#             discretize.TreeMesh.writeUBC(
#                 local_mesh, 'OctreeMesh-pre.msh',
#                 models={'SigmaOctree-pre.dat': np.ones(local_mesh.nC)}
#             )
#
#     global_mesh.finalize()
#     global_active = utils.surface2ind_topo(
#         global_mesh, topography, method='linear'
#     )
#
#     # Cycle back to all local meshes and create tile maps
#     local_maps = []
#     for mesh in local_meshes:
#         local_maps.append(
#             maps.TileMap(global_mesh, global_active, mesh)
#         )
#
#     return (global_mesh, global_active), (local_meshes, local_maps)


def create_sub_simulations_old(source_list,
                               dem,
                               base_mesh,
                               oct_levels=[5, 3, 2, 2],
                               remote_quadrant=0,
                               output_meshes=False,
                               padLen=800,
                               remote_present=True):

    sub_survey = dc.Survey(source_list)
    electrodes = np.vstack((sub_survey.locations_a,
                            sub_survey.locations_b,
                            sub_survey.locations_m,
                            sub_survey.locations_n))

    flag_remote = electrodes[:, 1] > electrodes[:, 1].min()
    remote_only = electrodes[:, 1] <= electrodes[:, 1].min()

    # figure remote location so discretization is good there
    if remote_quadrant == 1:
        flag_remote = electrodes[:, 1] > electrodes[:, 1].max()
        remote_only = electrodes[:, 1] <= electrodes[:, 1].max()
    elif remote_quadrant == 2:
        flag_remote = electrodes[:, 0] > electrodes[:, 0].max()
        remote_only = electrodes[:, 0] <= electrodes[:, 0].max()
    elif remote_quadrant == 3:
        flag_remote = electrodes[:, 1] > electrodes[:, 1].min()
        remote_only = electrodes[:, 1] <= electrodes[:, 1].min()
    elif remote_quadrant == 4:
        flag_remote = electrodes[:, 1] > electrodes[:, 1].min()
        remote_only = electrodes[:, 1] <= electrodes[:, 1].min()

    # create databases
    no_remote_db = electrodes[flag_remote]
    remote_db = electrodes[remote_only]

    # combine topo
    topo = np.vstack((dem, electrodes))
    # creating mesh using Discretize!
    h = [base_mesh.h[0].min(), base_mesh.h[1].min(), base_mesh.h[1].min()]

    padding_distance = np.r_[np.c_[padLen, padLen],
                             np.c_[padLen, padLen], np.c_[padLen, padLen]]

    mesh = discretize.utils.mesh_builder_xyz(electrodes, h,
                                  padding_distance=padding_distance,
                                  mesh_type='TREE', base_mesh=base_mesh,
                                  depth_core=1000)
    mesh = discretize.utils.refine_tree_xyz(mesh, topo,
                                 method='surface', octree_levels=[1, 1, 0],
                                 finalize=False)
    if remote_present:
        mesh = discretize.utils.refine_tree_xyz(mesh, remote_db,
                                     method='radial', octree_levels=[2, 1, 1],
                                     finalize=False)
        mesh = discretize.utils.refine_tree_xyz(mesh, no_remote_db,
                                     method='box', octree_levels=oct_levels,
                                     # octree_levels_padding=[4, 4, 2, 2],
                                     finalize=True)
    else:
        mesh = discretize.utils.refine_tree_xyz(mesh, electrodes,
                                     method='box', octree_levels=oct_levels,
                                     # octree_levels_padding=[4, 4, 2, 2],
                                     finalize=True)

    if output_meshes:
        discretize.TreeMesh.writeUBC(mesh, 'OctreeMesh-pre.msh', models={'SigmaOctree-pre.dat': np.ones(mesh.nC)})

    # actinds = mesh.gridCC[:, 2] < 0
    actinds = utils.surface2ind_topo(mesh, topo, method='linear')
    # print(actinds.shape)
    # drape topo
    sub_survey.drape_electrodes_on_topography(mesh, actinds)

    actmap = maps.InjectActiveCells(
        mesh, indActive=actinds, valInactive=np.log(1e-8)
    )
    mapping = maps.ExpMap(mesh) * actmap

    # Generate 3D DC problem
    # "CC" means potential is defined at center
    sim = dc.Simulation3DCellCentered(
            mesh, survey=sub_survey, sigmaMap=mapping, storeJ=False,
            Solver=Solver
    )
    sim.actinds = actinds
    return sim
