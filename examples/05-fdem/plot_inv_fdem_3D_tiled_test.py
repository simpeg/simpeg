from discretize import TreeMesh
from discretize.utils import mkvc, refine_tree_xyz
from time import time
import dask.array as da
from dask.distributed import Client, LocalCluster
from SimPEG.utils import plot2Ddata, surface2ind_topo
from SimPEG import maps
from SimPEG import dask
import SimPEG.electromagnetics.frequency_domain as fdem
from SimPEG.utils.drivers import create_nested_mesh
from SimPEG import (
    maps,
    utils,
    data_misfit,
    regularization,
    optimization,
    inverse_problem,
    directives,
    inversion,
    objective_function,
    data
)
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

try:
    from pymatsolver import Pardiso as solver
except ImportError:
    from SimPEG import SolverLU as solver

save_file = False

def create_tile_em_misfit(sources, obs, uncert, global_mesh, global_active, tile_id, mstart, use_global=False):
    local_survey = fdem.Survey(sources)

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
    simulation = fdem.simulation.Simulation3DMagneticFluxDensity(
        local_mesh, survey=local_survey, sigmaMap=mapping,
        solver=solver,
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

    # Frequencies being predicted
    frequencies = [100, 500, 2500]

    # Defining transmitter locations
    N = 9
    xtx, ytx, ztx = np.meshgrid(np.linspace(-200, 200, N), np.linspace(-200, 200, N), [40])
    source_locations = np.c_[mkvc(xtx), mkvc(ytx), mkvc(ztx)]
    np.savetxt("Receivers.dat", source_locations)
    ntx = np.size(xtx)
    print("Number of sources: ", ntx)
    # Define receiver locations
    xrx, yrx, zrx = np.meshgrid(np.linspace(-200, 200, N), np.linspace(-200, 200, N), [20])
    receiver_locations = np.c_[mkvc(xrx), mkvc(yrx), mkvc(zrx)]

    source_list = []  # Create empty list to store sources

    # Each unique location and frequency defines a new transmitter
    for ii in range(len(frequencies)):
        for jj in range(ntx):

            # Define receivers of different type at each location
            bzr_receiver = fdem.receivers.PointMagneticFluxDensitySecondary(
                receiver_locations[jj, :], "z", "real"
            )
            bzi_receiver = fdem.receivers.PointMagneticFluxDensitySecondary(
                receiver_locations[jj, :], "z", "imag"
            )
            receivers_list = [bzr_receiver, bzi_receiver]

            # Must define the transmitter properties and associated receivers
            source_list.append(
                fdem.sources.MagDipole(
                    receivers_list,
                    frequencies[ii],
                    source_locations[jj],
                    orientation="z",
                    moment=100,
                )
            )

    survey = fdem.Survey(source_list)

    dh = 25.0  # base cell width
    dom_width = 3000.0  # domain width
    nbc = 2 ** int(np.round(np.log(dom_width / dh) / np.log(2.0)))  # num. base cells

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
    mstart = background_conductivity * np.ones(active_cells.sum())
    ind_block = (
        (mesh.gridCC[active_cells, 0] < 100.0)
        & (mesh.gridCC[active_cells, 0] > -100.0)
        & (mesh.gridCC[active_cells, 1] < 100.0)
        & (mesh.gridCC[active_cells, 1] > -100.0)
        & (mesh.gridCC[active_cells, 2] > -275.0)
        & (mesh.gridCC[active_cells, 2] < -75.0)
    )
    model[ind_block] = block_conductivity

    # Plot Resistivity Model
    # mpl.rcParams.update({"font.size": 12})
    # fig = plt.figure(figsize=(7, 6))
    #
    # ploting_map = expmap * maps.InjectActiveCells(mesh, active_cells, np.nan)
    # ax1 = fig.add_axes([0.13, 0.1, 0.6, 0.85])
    # mesh.plotSlice(
    #     np.log10(ploting_map * model),
    #     normal="Y",
    #     ax=ax1,
    #     ind=int(mesh.hx.size / 2),
    #     grid=True,
    # )
    # ax1.set_title("Conductivity Model at Y = 0 m")
    #
    # ax2 = fig.add_axes([0.75, 0.1, 0.05, 0.85])
    # cbar = mpl.colorbar.ColorbarBase(
    #     ax2, orientation="vertical", format="$10^{%.1f}$"
    # )
    # cbar.set_label("Conductivity [S/m]", rotation=270, labelpad=15, size=12)

    simulation_g = fdem.simulation.Simulation3DMagneticFluxDensity(
        mesh, survey=survey, sigmaMap=model_map, solver=solver
    )

    TreeMesh.writeUBC(mesh, "Mesh.msh", models={"True.con": model_map * model})
    # Compute predicted data for a your model.
    # dpred = simulation.dpred(model)
    d_true = simulation_g.dpred(model)

    global_data = simulation_g.make_synthetic_data(model, relative_error=0.0, noise_floor=5e-14, add_noise=True)

    survey.dobs = global_data.dobs
    survey.std = np.abs(survey.dobs * global_data.relative_error) + global_data.noise_floor
    # Data are organized by frequency, transmitter location, then by receiver. We nFreq transmitters
    # and each transmitter had 2 receivers (real and imaginary component). So
    # first we will pick out the real and imaginary data
    # bz_real = global_data.dobs[0::2]
    # bz_imag = global_data.dobs[1::2]

    # Then we will will reshape the data for plotting.
    # bz_real_plotting = np.reshape(bz_real, (len(frequencies), ntx))
    # bz_imag_plotting = np.reshape(bz_imag, (len(frequencies), ntx))



    # Real Component
    frequencies_index = 0
    # v_max = np.max(np.abs(bz_real_plotting[frequencies_index, :]))
    def plot_data(locations, data):
        fig = plt.figure(figsize=(10, 12))
        for ii in range(len(frequencies)):

            ax1 = plt.subplot(len(frequencies), 2, ii*2+1)
            im, _ = plot2Ddata(
                locations[:, 0:2],
                data[2*ntx*ii:2*ntx*(ii+1)][::2],
                ax=ax1,
                ncontour=30,
                # clim=(-v_max, v_max),
                contourOpts={"cmap": "bwr"},
            )
            ax1.set_title(f"Re[$B_z$] at {frequencies[ii]} Hz")
            plt.colorbar(im)
            # ax2 = fig.add_axes([0.41, 0.05, 0.02, 0.9])
            # norm = mpl.colors.Normalize(vmin=-v_max, vmax=v_max)
            # cbar = mpl.colorbar.ColorbarBase(
            #     ax2, norm=norm, orientation="vertical", cmap=mpl.cm.bwr
            # )
            # cbar.set_label("$T$", rotation=270, labelpad=15, size=12)

            # Imaginary Component
            # v_max = np.max(np.abs(bz_imag_plotting[frequencies_index, :]))
            ax2 = plt.subplot(len(frequencies), 2, ii*2+2)
            im, _ = plot2Ddata(
                locations[:, 0:2],
                data[2*ntx*ii:2*ntx*(ii+1)][1::2],
                ax=ax2,
                ncontour=30,
                # clim=(-v_max, v_max),
                contourOpts={"cmap": "bwr"},
            )
            ax2.set_title(f"Im[$B_z$] at {frequencies[ii]} Hz")
            plt.colorbar(im)
            # ax2 = fig.add_axes([0.91, 0.05, 0.02, 0.9])
            # norm = mpl.colors.Normalize(vmin=-v_max, vmax=v_max)
            # cbar = mpl.colorbar.ColorbarBase(
            #     ax2, norm=norm, orientation="vertical", cmap=mpl.cm.bwr
            # )
            # cbar.set_label("$T$", rotation=270, labelpad=15, size=12)

        plt.show()

    plot_data(receiver_locations, global_data.dobs)
    # Split the problem in tiles (over frequency)

    #np.abs(survey.dobs * global_data.relative_error) + global_data.noise_floor

    ######### To run with tiling
    idx_start = 0
    idx_end = 0
    # do every 5 sources
    cnt = 0
    local_misfits = []
    for ii, frequency in enumerate(survey.frequencies):
        idx_start, idx_end = ii*2*ntx, (ii+1)*2*ntx
        sources = survey.get_sources_by_frequency(frequency)
        dobs = survey.dobs[idx_start:idx_end]
    #         print(dobs.shape, len(src_collect))
        delayed_misfit = create_tile_em_misfit(
            sources,
            survey.dobs[idx_start:idx_end],
            survey.std[idx_start:idx_end],
            mesh, active_cells, ii, model,
            use_global=True
        )
        local_misfits += [delayed_misfit]

    ########## To run without tiling
    # local_misfits = [data_misfit.L2DataMisfit(
    #     data=global_data, simulation=simulation_g
    # )]
    # local_misfits[0].W = 1 / survey.std
    # local_misfits[0].simulation.model = model
    global_misfit = objective_function.ComboObjectiveFunction(
                    local_misfits
    )

    JtJdiag = np.zeros_like(model)
    scale = []
    for local_misfit in global_misfit.objfcts:

        local_misfit.simulation.model = model
        JtJdiag += local_misfit.simulation.getJtJdiag(mstart, W=local_misfit.W)
        scale += [da.sum(local_misfit.simulation.Jmatrix[0::2, :]**2., axis=0).compute().max()]
        scale += [da.sum(local_misfit.simulation.Jmatrix[1::2, :]**2., axis=0).compute().max()]
    JtJdiag **= 0.5
    # TreeMesh.writeUBC(mesh, "Mesh.msh", models={"Sensitivities.con": np.log10(model_map * np.log(np.abs(JtJdiag)))})

    # scale = [15.4, 2.8, 21.6, 1, 6.55, 2.7]
    for ii, local_misfit in enumerate(global_misfit.objfcts):
        idx_start, idx_end = ii * 2 * ntx, (ii + 1) * 2 * ntx
        std = survey.std[idx_start:idx_end]
        std[::2] = std[::2] * (np.max(scale)/scale[ii*2])**0.25
        std[1::2] = std[1::2] * (np.max(scale)/scale[ii*2+1])**0.25

        # std[::2] *= (scale[ii*2])**0.5/5.
        # std[1::2] *= (scale[ii*2+1])**0.5/5.

        local_misfit.W = 1./std
        local_misfit.simulation._Jmatrix = None
    #
    # Jvec = simulation_g.Jvec(model, np.ones_like(model))
    # Jtvec = simulation_g.Jtvec(model, np.ones_like(survey.std))
    # np.savetxt("Jvec_dask.dat", Jvec.compute())
    # np.savetxt("Jtvec_dask.dat", Jtvec.compute())
    #
    #
    # Jvec = simulation_g.Jvec(model, np.ones_like(model))
    # Jtvec = simulation_g.Jtvec(model, np.ones_like(survey.std))
    # np.savetxt("Jvec.dat", Jvec)
    # np.savetxt("Jtvec.dat", Jtvec)

    use_preconditioner = False
    coolingFactor = 2
    coolingRate = 1
    beta0_ratio = 1e-1

    # Map for a regularization
    regmap = maps.IdentityMap(nP=int(active_cells.sum()))
    reg = regularization.Sparse(mesh, indActive=active_cells, mapping=regmap)

    print('[INFO] Getting things started on inversion...')
    # set alpha length scales
    reg.alpha_s = 1 # alpha_s
    reg.alpha_x = 1
    reg.alpha_y = 1
    reg.alpha_z = 1
    reg.mref = background_conductivity * np.ones(active_cells.sum())

    opt = optimization.ProjectedGNCG(
        maxIter=2, upper=np.inf, lower=-np.inf, tolCG=1e-5,
        maxIterCG=20,
    )
    invProb = inverse_problem.BaseInvProblem(global_misfit, reg, opt, beta=1e+2)
    beta = directives.BetaSchedule(
        coolingFactor=coolingFactor, coolingRate=coolingRate
    )
    # plot_data(receiver_locations, np.hstack(invProb.get_dpred(model)))
    betaest = directives.BetaEstimate_ByEig(beta0_ratio=beta0_ratio, method="old")
    # target = directives.TargetMisfit()
    # target.target = survey.nD
    update_IRLS = directives.Update_IRLS(
        f_min_change=1e-4, max_irls_iterations=0, coolEpsFact=1.5, beta_tol=4., coolingRate=coolingRate, coolingFactor=coolingFactor
    )
    # saveIter = directives.SaveModelEveryIteration()
    save_model = directives.SaveUBCModelEveryIteration(mesh=mesh, mapping=model_map, file_name="FEM_", replace=False)

    saveIterVar = directives.SaveOutputEveryIteration()
    # Need to have basice saving function
    if use_preconditioner:
        update_Jacobi = directives.UpdatePreconditioner()
        updateSensW = directives.UpdateSensitivityWeights()
        directiveList = [
            save_model, updateSensW, update_IRLS, update_Jacobi
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
    minv = inv.run(mstart)
    rho_est = model_map * minv
    print("Total runtime: ", time()-tc)
    # np.save('model_out.npy', rho_est)
    plot_data(receiver_locations, np.hstack(invProb.dpred))
    plot_data(receiver_locations, global_data.dobs - np.hstack(invProb.dpred))
    plot_data(receiver_locations, (global_data.dobs - np.hstack(invProb.dpred))/survey.std)

    # for ii, local_misfit in enumerate(global_misfit.objfcts):
    #     idx_start, idx_end = ii * 2 * ntx, (ii + 1) * 2 * ntx
    #     obs, pre, std = survey.dobs[idx_start:idx_end], np.hstack(invProb.dpred)[idx_start:idx_end], survey.std[idx_start:idx_end]
    #     print(np.sum((obs[::2] - pre[::2])**2. / std[::2]**2.))
    #     print(np.sum((obs[1::2] - pre[1::2]) ** 2. / std[1::2] ** 2.))

    # mesh.writeUBC('OctreeMesh-test.msh', models={'ubc.con': rho_est})

if __name__ == '__main__':
    run()