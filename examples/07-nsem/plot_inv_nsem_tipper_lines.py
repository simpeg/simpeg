from discretize import TreeMesh
from SimPEG import optimization, maps, objective_function, regularization, inverse_problem, directives, inversion
from discretize.utils import mkvc, refine_tree_xyz
from SimPEG.electromagnetics import natural_source as ns
import numpy as np
from dask.distributed import Client, LocalCluster
from pymatsolver import Pardiso as Solver
from SimPEG.utils import surface2ind_topo
from SimPEG.utils.drivers import create_local_misfit
from SimPEG.utils.model_builder import addBlock
from time import time


def run():
    """
    Forward and tiled inversion of Tipper line data
    """
    dh = 25.0  # base cell width
    dom_width = 3000.0  # domain width
    frequencies = [10, 50, 200]

    tile_buffer = 100  # Distance of refinement around local points

    cluster = LocalCluster(processes=False)
    client = Client(cluster)
    xx, yy = np.meshgrid(np.linspace(-3000, 3000, 101), np.linspace(-3000, 3000, 101))
    zz = np.zeros(np.shape(xx))
    topo_xyz = np.c_[mkvc(xx), mkvc(yy), mkvc(zz)]

    # Lines of receiver locations
    n_lines = 5
    n_stn = 15
    x_locs = np.linspace(-200, 200, n_stn)
    y_locs = np.linspace(-200, 200, n_lines).tolist()
    z_locs = np.ones_like(x_locs) * dh
    receiver_locations = []
    line_ids = []
    for ii, yy in enumerate(y_locs):
        receiver_locations += [np.c_[x_locs, np.ones_like(x_locs)*yy, z_locs]]
        line_ids += [np.ones_like(x_locs) * ii]

    receiver_locations = np.vstack(receiver_locations)
    line_ids = np.vstack(line_ids)

    np.savetxt("Receivers.dat", receiver_locations)
    # Create mesh and model
    nbc = 2 ** int(np.round(np.log(dom_width / dh) / np.log(2.0)))  # num. base cells

    # Define the base mesh
    h = [(dh, nbc)]
    mesh = TreeMesh([h, h, h], x0="CCC")

    # Mesh refinement based on topography
    mesh = refine_tree_xyz(
        mesh, topo_xyz, octree_levels=[0, 0, 2], method="surface", finalize=False
    )

    # Mesh refinement near transmitters and receivers
    mesh = refine_tree_xyz(
        mesh, receiver_locations, octree_levels=[6, 6], method="radial", finalize=True
    )

    # Find cells that are active in the forward modeling (cells below surface)
    active_cells = surface2ind_topo(mesh, topo_xyz)

    # Define mapping from model to active cells
    expmap = maps.ExpMap(mesh)

    # Conductivity in S/m (or resistivity in Ohm m)
    air_conductivity = np.log(1e-8)
    background_conductivity = np.log(1e-2)
    block_conductivity = np.log(1e1)

    model_map = expmap * maps.InjectActiveCells(mesh, active_cells, air_conductivity)

    # Define model. Models in SimPEG are vector arrays
    model = background_conductivity * np.ones(active_cells.sum())
    m_background = background_conductivity * np.ones(active_cells.sum())
    model = addBlock(
        mesh.gridCC, model, [-100, -100, -275], [100, 100, -75], block_conductivity
    )

    # Create receivers
    rxList = []
    channels = []
    components = ['zx', 'zy']
    parts = ['real', 'imag']
    for component in components:
        for part in parts:
            rec = ns.Rx.Point3DTipper(receiver_locations, component, part)
            # then hacked and input to assign a reference station
            rec.reference_locations = np.zeros_like(receiver_locations) + np.asarray([-400, -400, 0])
            rxList.append(rec)
            channels += [f"{component}_{part}"]

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
    simulation.survey.dtrue = simulation.dpred(model)

    # Assign uncertainties
    # make data object
    global_data = simulation.make_synthetic_data(model, relative_error=0.0, noise_floor=5e-3, add_noise=True)

    survey.dobs = global_data.dobs
    survey.std = np.abs(survey.dobs * global_data.relative_error) + global_data.noise_floor

    data_ids = np.arange(survey.dobs.shape[0]).reshape(
        (len(frequencies), len(channels), -1)
    )

    print("Creating tiles ... ")
    local_misfits = []
    tile_count = 0
    data_ordering = []
    freq_blocks = [[freq] for freq in frequencies] # Split on frequencies
    line_segs = []
    for ii in np.unique(line_ids).tolist():
        line_segs += [np.where(line_ids == ii)[0]]

    for freq_block in freq_blocks:
        freq_rows = [ind for ind, freq in enumerate(frequencies) if freq in freq_block]
        for line_seg in line_segs:
            block_ind = data_ids[freq_rows, :, :]
            block_ind = block_ind[:, :, line_seg].flatten()

            data_ordering += [block_ind]
            rxList = []
            for component in components:
                for part in parts:
                    rxList.append(
                        ns.Rx.Point3DTipper(
                            receiver_locations[line_seg, :], component, part
                        )
                    )

            source_list = []
            for freq in freq_block:
                source_list += [
                    ns.Src.Planewave_xy_1Dprimary(
                        rxList, freq, sigma_primary=[np.exp(background_conductivity)]
                    )
                ]

            local_survey = ns.Survey(source_list)
            local_survey.data_index = block_ind

            local_survey.dobs = survey.dobs[block_ind]
            local_survey.std = survey.std[block_ind]

            local_misfits += [
                create_local_misfit(
                    ns.simulation.Simulation3DPrimarySecondary,
                    local_survey,
                    mesh,
                    active_cells,
                    tile_count,
                    tile_buffer=tile_buffer,
                    min_level=4,
                    solver=Solver
                )
            ]

            tile_count += 1
            print(f"Tile {tile_count} of {len(freq_blocks) * len(line_segs)}")

    # for freq in frequencies:
    #     cnt_comp = 0
    #     comps_ = ['rho_xy', 'phi_xy', 'rho_yx', 'phi_yx']
    #     # comps_ = ['r_xx', 'i_xx', 'r_xy', 'i_xy', 'r_yx', 'i_yx', 'r_yy', 'i_yy']
    #     fig1 = plt.figure(figsize=(18, 14))
    #     for rx_orientation in range(len(comps_)):
    #         pert = 1e-3  # np.percentile(np.abs(dnew[cnt, num_station*rx_orientation:num_station*(rx_orientation + 1)]), 20)
    #         if comps_[rx_orientation][-2:] == 'xx':
    #             stdnew[cnt, num_station*rx_orientation:num_station*(rx_orientation + 1)] = stdnew[cnt, num_station*rx_orientation:num_station*(rx_orientation + 1)] + pert
    #         elif comps_[rx_orientation][-2:] == 'yy':
    #             stdnew[cnt, num_station*rx_orientation:num_station*(rx_orientation + 1)] = stdnew[cnt, num_station*rx_orientation:num_station*(rx_orientation + 1)] + pert
    #         ax = plt.subplot(4,2, cnt_comp + 1)
    #         im = utils.plot_utils.plot2Ddata(receiver_locations, dnew[cnt, num_station*rx_orientation:num_station*(rx_orientation + 1)], ax=ax)
    #         ax.set_title(comps_[rx_orientation])
    #         plt.colorbar(im[0])
    #         cnt_comp += 1
    #     cnt += 1
    #     fig1.suptitle(f'{freq} Hz Forward Data', fontsize='20')
    # plt.show()
    # print(mkvc(stdnew).shape)

    # local_misfits = [data_misfit.L2DataMisfit(
    #     data=global_data, simulation=simulation
    # )]
    # local_misfits[0].W = 1 / survey.std
    #

    global_misfit = objective_function.ComboObjectiveFunction(
                    local_misfits
    )

    use_preconditioner = True
    coolingFactor = 2
    coolingRate = 1
    beta0_ratio = 1e+1

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
        maxIter=5, upper=np.inf, lower=-np.inf, tolCG=1e-5,
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