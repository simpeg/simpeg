from SimPEG import dask
from discretize import TreeMesh
from SimPEG import maps, utils, data, optimization, maps, objective_function, regularization, inverse_problem, directives, inversion, data_misfit
from discretize.utils import mkvc, refine_tree_xyz
from SimPEG.electromagnetics import natural_source as ns
import numpy as np
from dask.distributed import Client, LocalCluster
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from pymatsolver import Pardiso as Solver
from SimPEG.utils import plot2Ddata, surface2ind_topo
from SimPEG.utils.drivers import create_nested_mesh
from time import time


def create_tile_em_misfit(sources, obs, uncert, global_mesh, global_active, tile_id, mstart, use_global=False):
    local_survey = ns.Survey(sources)

    electrodes = []
    for source in sources:
        electrodes += [source.location]
        electrodes += [receiver.locations for receiver in source.receiver_list]
    electrodes = np.vstack(electrodes)
    local_survey.dobs = obs
    local_survey.std = uncert

    # Create tile map between global and local
    if use_global:
        local_mesh = global_mesh
        local_active = global_active
        local_map = maps.IdentityMap(nP=int(local_active.sum()))
    else:
        local_mesh = create_nested_mesh(
            electrodes, global_mesh,
        )
        local_map = maps.TileMap(global_mesh, global_active, local_mesh)
        local_active = local_map.local_active

    actmap = maps.InjectActiveCells(
        local_mesh, indActive=local_active, valInactive=np.log(1e-8)
    )
    expmap = maps.ExpMap(local_mesh)
    mapping = expmap * actmap
    # Create the local misfit
    max_chunk_size = 256
    simulation = ns.simulation.Simulation3DMagneticFluxDensity(
        local_mesh, survey=local_survey, sigmaMap=mapping,
        Solver=Solver,
#         chunk_format="row",
#         max_chunk_size=max_chunk_size,
#         workers=workers
    )
    simulation.sensitivity_path = './sensitivity/Tile' + str(tile_id) + '/'
#     print(simulation.getSourceTerm().shape)
    data_object = data.Data(
        local_survey,
        dobs=obs,
        standard_deviation=uncert,
    )
    data_object.dobs = obs
    data_object.standard_deviation = uncert
    local_misfit = data_misfit.L2DataMisfit(
        data=data_object, simulation=simulation, model_map=local_map
    )
    local_misfit.W = 1 / uncert

    return local_misfit

def run():
    cluster = LocalCluster(processes=False)
    client = Client(cluster)
    xx, yy = np.meshgrid(np.linspace(-3000, 3000, 101), np.linspace(-3000, 3000, 101))
    zz = np.zeros(np.shape(xx))
    topo_xyz = np.c_[mkvc(xx), mkvc(yy), mkvc(zz)]

    # create receiver locations
    N = 9
    xrx, yrx, zrx = np.meshgrid(np.linspace(-200, 200, N), np.linspace(-200, 200, N), [0])
    receiver_locations = np.c_[mkvc(xrx), mkvc(yrx), mkvc(zrx)]

    # Create mesh and model
    dh = 25.0  # base cell width
    dom_width = 3000.0  # domain width
    nbc = 2 ** int(np.round(np.log(dom_width / dh) / np.log(2.0)))  # num. base cells
    frequencies = [10, 50, 200]
    # Define the base mesh
    h = [(dh, nbc)]
    mesh = TreeMesh([h, h, h], x0="CCC")

    # Mesh refinement based on topography
    mesh = refine_tree_xyz(
        mesh, topo_xyz, octree_levels=[0, 0, 0, 1], method="surface", finalize=False
    )

    # Mesh refinement near transmitters and receivers
    mesh = refine_tree_xyz(
        mesh, receiver_locations, octree_levels=[2, 4], method="radial", finalize=False
    )

    # Refine core mesh region
    xp, yp, zp = np.meshgrid([-250.0, 250.0], [-250.0, 250.0], [-300.0, 0.0])
    xyz = np.c_[mkvc(xp), mkvc(yp), mkvc(zp)]
    mesh = refine_tree_xyz(mesh, xyz, octree_levels=[0, 2, 4], method="box", finalize=False)

    mesh.finalize()

    # Conductivity in S/m (or resistivity in Ohm m)
    air_conductivity = np.log(1e-8)
    background_conductivity = np.log(1e-2)
    block_conductivity = np.log(1e1)

    # Find cells that are active in the forward modeling (cells below surface)
    active_cells = surface2ind_topo(mesh, topo_xyz)

    # Define mapping from model to active cells
    expmap = maps.ExpMap(mesh)
    model_map = expmap * maps.InjectActiveCells(mesh, active_cells, air_conductivity)

    # Define model. Models in SimPEG are vector arrays
    model = background_conductivity * np.ones(active_cells.sum())
    m_background = background_conductivity * np.ones(active_cells.sum())
    ind_block = (
            (mesh.gridCC[active_cells, 0] < 100.0)
            & (mesh.gridCC[active_cells, 0] > -100.0)
            & (mesh.gridCC[active_cells, 1] < 100.0)
            & (mesh.gridCC[active_cells, 1] > -100.0)
            & (mesh.gridCC[active_cells, 2] > -275.0)
            & (mesh.gridCC[active_cells, 2] < -75.0)
    )
    model[ind_block] = block_conductivity

    # Create receivers
    rxList = []
    for rx_orientation in ['xx', 'xy', 'yx', 'yy']:
        #     rxList.append(ns.Rx.Point3DComplexResistivity(locations=None, locations_e=rx_loc, locations_h=rx_loc, orientation=rx_orientation, component='apparent_resistivity'))
        #     rxList.append(ns.Rx.Point3DComplexResistivity(locations=None,locations_e=rx_loc, locations_h=rx_loc, orientation=rx_orientation, component='phase'))
        rxList.append(ns.Rx.Point3DImpedance(receiver_locations, rx_orientation, 'real'))
        rxList.append(ns.Rx.Point3DImpedance(receiver_locations, rx_orientation, 'imag'))
    # for rx_orientation in ['zx', 'zy']:
    #     rxList.append(ns.Rx.Point3DTipper(rx_loc, rx_orientation, 'real'))
    #     rxList.append(ns.Rx.Point3DTipper(rx_loc, rx_orientation, 'imag'))

    # Source list
    srcList = [
        ns.Src.Planewave_xy_1Dprimary(rxList, freq, sigma_primary=model_map * m_background)
        for freq in frequencies
    ]

    # Survey MT
    survey = ns.Survey(srcList)

    actMap = maps.InjectActiveCells(
        mesh=mesh, indActive=active_cells, valInactive=np.log(1e-8)
    )
    mapping = maps.ExpMap(mesh) * actMap

    # Setup the problem object
    simulation = ns.simulation.Simulation3DPrimarySecondary(mesh, survey=survey,
                                               sigmaMap=model_map,
                                               sigmaPrimary=model_map * m_background,
                                               solver=Solver)

    simulation.model = model
    # fields = simulation.fields()

    # v = np.random.rand(survey.nD,)
    # # print problem.PropMap.PropModel.nP
    # w = np.random.rand(active.sum(),)
    # # np.random.seed(1983)
    # # fwd = simulation.getJ(sigBG[active])
    # # fwd = simulation.Jtvec(sigBG[active], v)
    # fwd = simulation.Jvec(sigBG[active], w)

    simulation.survey.dtrue = simulation.dpred(model)
    # create observations
    simulation.survey.dobs = survey.dtrue

    # Assign uncertainties
    # make data object
    global_data = simulation.make_synthetic_data(model, relative_error=0.0, noise_floor=5e-3, add_noise=True)

    survey.dobs = global_data.dobs
    survey.std = np.abs(survey.dobs * global_data.relative_error) + global_data.noise_floor


    num_station = receiver_locations.shape[0]
    num_sets = int(survey.dobs.shape[0] / len(frequencies))
    dnew = np.reshape(survey.dobs, (3, num_sets))
    stdnew = np.reshape(survey.std, (3, num_sets))

    cnt = 0

    for freq in frequencies:
        cnt_comp = 0
    #     comps_ = ['rho_xy', 'phi_xy', 'rho_yx', 'phi_yx']
        comps_ = ['r_xx', 'i_xx', 'r_xy', 'i_xy', 'r_yx', 'i_yx', 'r_yy', 'i_yy']
        fig1 = plt.figure(figsize=(18, 14))
        for rx_orientation in range(len(comps_)):
            pert = 1e-3  # np.percentile(np.abs(dnew[cnt, num_station*rx_orientation:num_station*(rx_orientation + 1)]), 20)
            if comps_[rx_orientation][-2:] == 'xx':
                stdnew[cnt, num_station*rx_orientation:num_station*(rx_orientation + 1)] = stdnew[cnt, num_station*rx_orientation:num_station*(rx_orientation + 1)] + pert
            elif comps_[rx_orientation][-2:] == 'yy':
                stdnew[cnt, num_station*rx_orientation:num_station*(rx_orientation + 1)] = stdnew[cnt, num_station*rx_orientation:num_station*(rx_orientation + 1)] + pert
            ax = plt.subplot(4,2, cnt_comp + 1)
            im = utils.plot_utils.plot2Ddata(receiver_locations, dnew[cnt, num_station*rx_orientation:num_station*(rx_orientation + 1)], ax=ax)
            ax.set_title(comps_[rx_orientation])
            plt.colorbar(im[0])
            cnt_comp += 1
        cnt += 1
        fig1.suptitle(f'{freq} Hz Forward Data', fontsize='20')
    plt.show()
    print(mkvc(stdnew).shape)

    local_misfits = [data_misfit.L2DataMisfit(
        data=global_data, simulation=simulation
    )]
    local_misfits[0].W = 1 / survey.std


    global_misfit = objective_function.ComboObjectiveFunction(
                    local_misfits
    )

    use_preconditioner = False
    coolingFactor = 2
    coolingRate = 1
    beta0_ratio = 1e-1

    # Map for a regularization
    regmap = maps.IdentityMap(nP=int(active_cells.sum()))
    reg = regularization.Sparse(mesh, indActive=active_cells, mapping=regmap)

    print('[INFO] Getting things started on inversion...')
    # set alpha length scales
    reg.alpha_s = 1  # alpha_s
    reg.alpha_x = 1
    reg.alpha_y = 1
    reg.alpha_z = 1
    reg.mref = background_conductivity * np.ones(active_cells.sum())

    opt = optimization.ProjectedGNCG(
        maxIter=1, upper=np.inf, lower=-np.inf, tolCG=1e-5,
        maxIterCG=20,
    )
    invProb = inverse_problem.BaseInvProblem(global_misfit, reg, opt)
    beta = directives.BetaSchedule(
        coolingFactor=coolingFactor, coolingRate=coolingRate
    )
    # plot_data(receiver_locations, np.hstack(invProb.get_dpred(model)))
    betaest = directives.BetaEstimate_ByEig(beta0_ratio=beta0_ratio, method="old")
    # target = directives.TargetMisfit()
    # target.target = survey.nD
    update_IRLS = directives.Update_IRLS(
        f_min_change=1e-4, max_irls_iterations=0, coolEpsFact=1.5, beta_tol=4., coolingRate=coolingRate,
        coolingFactor=coolingFactor
    )
    # saveIter = directives.SaveModelEveryIteration()
    save_model = directives.SaveUBCModelEveryIteration(mesh=mesh, mapping=model_map, file_name="FEM_", replace=False)

    saveIterVar = directives.SaveOutputEveryIteration()
    # Need to have basice saving function
    if use_preconditioner:
        update_Jacobi = directives.UpdatePreconditioner()
        updateSensW = directives.UpdateSensitivityWeights()
        directiveList = [
            save_model, updateSensW, update_IRLS, update_Jacobi, betaest
        ]
    else:
        directiveList = [
            save_model, betaest, update_IRLS
        ]
    inv = inversion.BaseInversion(
        invProb, directiveList=directiveList)
    opt.LSshorten = 0.5
    opt.remember('xc')

    # Run Inversion ================================================================
    tc = time()
    minv = inv.run(m_background)
    rho_est = model_map * minv
    print("Total runtime: ", time() - tc)

if __name__ == '__main__':
    run()