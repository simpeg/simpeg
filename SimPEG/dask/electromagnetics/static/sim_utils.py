from .... import maps, utils
import discretize
from discretize.utils import mkvc, refine_tree_xyz
from ....electromagnetics import resistivity as dc
from ....electromagnetics.static.utils import StaticUtils as dcutils
from pymatsolver import Pardiso as Solver
import numpy as np


def create_sub_simulations(source_list,
                           dem,
                           base_mesh,
                           global_active,
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
    h = [base_mesh.hx.min(), base_mesh.hy.min(), base_mesh.hy.min()]

    padding_distance = np.r_[np.c_[padLen, padLen],
                             np.c_[padLen, padLen], np.c_[padLen, padLen]]

    mesh = discretize.utils.mesh_builder_xyz(
        electrodes, h,
        padding_distance=padding_distance,
        mesh_type='TREE', base_mesh=base_mesh,
        depth_core=1000
    )
    mesh = discretize.utils.refine_tree_xyz(
        mesh, topo,
        method='surface', octree_levels=[1, 1, 0],
        finalize=False
    )

    if remote_present:
        mesh = discretize.utils.refine_tree_xyz(
            mesh, remote_db,
            method='radial', octree_levels=[2, 1, 1],
            finalize=False
        )
        mesh = discretize.utils.refine_tree_xyz(
            mesh, no_remote_db,
            method='box', octree_levels=oct_levels,
            # octree_levels_padding=[4, 4, 2, 2],
            finalize=True
        )
    else:
        mesh = discretize.utils.refine_tree_xyz(
            mesh, electrodes,
            method='box', octree_levels=oct_levels,
            # octree_levels_padding=[4, 4, 2, 2],
            finalize=True
        )

    if output_meshes:
        discretize.TreeMesh.writeUBC(
            mesh, 'OctreeMesh-pre.msh',
            models={'SigmaOctree-pre.dat': np.ones(mesh.nC)}
        )

    # actinds = mesh.gridCC[:, 2] < 0
    local_active = utils.surface2ind_topo(mesh, topo, method='linear')
    # print(actinds.shape)
    # drape topo
    sub_survey.drape_electrodes_on_topography(mesh, local_active)

    # actmap = maps.InjectActiveCells(
    #     mesh, indActive=actinds, valInactive=np.log(1e-8)
    # )
    # mapping = maps.ExpMap(mesh) * actmap

    tile_map = maps.TileMap(base_mesh, global_active, mesh)

    # Generate 3D DC problem
    # "CC" means potential is defined at center
    sim = dc.Simulation3DCellCentered(
        mesh, survey=sub_survey, sigmaMap=tile_map, storeJ=False,
        Solver=Solver
    )

    del mesh
    sim.actinds = local_active
    return sim


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
    h = [base_mesh.hx.min(), base_mesh.hy.min(), base_mesh.hy.min()]

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
