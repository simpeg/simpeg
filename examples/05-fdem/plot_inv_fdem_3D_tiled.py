from discretize import TreeMesh
from discretize.utils import mkvc, refine_tree_xyz
from time import time
from dask.distributed import Client, LocalCluster
from SimPEG.utils import plot2Ddata, surface2ind_topo
from SimPEG import maps
# from SimPEG import dask
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
    from pymatsolver import Pardiso as Solver
except ImportError:
    from SimPEG import SolverLU as Solver

save_file = False

def create_tile_em_misfit(sources, obs, uncert, global_mesh, global_active, tile_id, mstart):
    local_survey = fdem.Survey(sources)

    electrodes = []
    for source in sources:
        electrodes += [source.location]
        electrodes += [receiver.locations for receiver in source.receiver_list]
    electrodes = np.vstack(electrodes)
    local_survey.dobs = obs
    local_survey.std = uncert

    # Create tile map between global and local
    local_mesh = create_nested_mesh(
        electrodes, global_mesh,
    )
    local_map = maps.TileMap(global_mesh, global_active, local_mesh)
    actmap = maps.InjectActiveCells(
        local_mesh, indActive=local_map.local_active, valInactive=np.log(1e-8)
    )
    expmap = maps.ExpMap(local_mesh)
    mapping = expmap * actmap
    # Create the local misfit
    max_chunk_size = 256
    simulation = fdem.simulation.Simulation3DMagneticFluxDensity(
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

    # Frequencies being predicted
    frequencies = [100, 500, 2500]

    # Defining transmitter locations
    N = 9
    xtx, ytx, ztx = np.meshgrid(np.linspace(-200, 200, N), np.linspace(-200, 200, N), [40])
    source_locations = np.c_[mkvc(xtx), mkvc(ytx), mkvc(ztx)]
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
    mpl.rcParams.update({"font.size": 12})
    fig = plt.figure(figsize=(7, 6))

    ploting_map = expmap * maps.InjectActiveCells(mesh, active_cells, np.nan)
    ax1 = fig.add_axes([0.13, 0.1, 0.6, 0.85])
    mesh.plotSlice(
        np.log10(ploting_map * model),
        normal="Y",
        ax=ax1,
        ind=int(mesh.hx.size / 2),
        grid=True,
    )
    ax1.set_title("Conductivity Model at Y = 0 m")

    ax2 = fig.add_axes([0.75, 0.1, 0.05, 0.85])
    cbar = mpl.colorbar.ColorbarBase(
        ax2, orientation="vertical", format="$10^{%.1f}$"
    )
    cbar.set_label("Conductivity [S/m]", rotation=270, labelpad=15, size=12)

    simulation_g = fdem.simulation.Simulation3DMagneticFluxDensity(
        mesh, survey=survey, sigmaMap=model_map, Solver=Solver
    )


    # Compute predicted data for a your model.
    # dpred = simulation.dpred(model)
    global_data = simulation_g.make_synthetic_data(model, relative_error=0.05, noise_floor=5e-14, add_noise=True)

    # Data are organized by frequency, transmitter location, then by receiver. We nFreq transmitters
    # and each transmitter had 2 receivers (real and imaginary component). So
    # first we will pick out the real and imaginary data
    bz_real = global_data.dobs[0::2]
    bz_imag = global_data.dobs[1::2]

    # Then we will will reshape the data for plotting.
    bz_real_plotting = np.reshape(bz_real, (len(frequencies), ntx))
    bz_imag_plotting = np.reshape(bz_imag, (len(frequencies), ntx))

    fig = plt.figure(figsize=(10, 4))

    # Real Component
    frequencies_index = 0
    v_max = np.max(np.abs(bz_real_plotting[frequencies_index, :]))
    ax1 = fig.add_axes([0.05, 0.05, 0.35, 0.9])
    plot2Ddata(
        receiver_locations[:, 0:2],
        bz_real_plotting[frequencies_index, :],
        ax=ax1,
        ncontour=30,
        clim=(-v_max, v_max),
        contourOpts={"cmap": "bwr"},
    )
    ax1.set_title("Re[$B_z$] at 100 Hz")

    ax2 = fig.add_axes([0.41, 0.05, 0.02, 0.9])
    norm = mpl.colors.Normalize(vmin=-v_max, vmax=v_max)
    cbar = mpl.colorbar.ColorbarBase(
        ax2, norm=norm, orientation="vertical", cmap=mpl.cm.bwr
    )
    cbar.set_label("$T$", rotation=270, labelpad=15, size=12)

    # Imaginary Component
    v_max = np.max(np.abs(bz_imag_plotting[frequencies_index, :]))
    ax1 = fig.add_axes([0.55, 0.05, 0.35, 0.9])
    plot2Ddata(
        receiver_locations[:, 0:2],
        bz_imag_plotting[frequencies_index, :],
        ax=ax1,
        ncontour=30,
        clim=(-v_max, v_max),
        contourOpts={"cmap": "bwr"},
    )
    ax1.set_title("Im[$B_z$] at 100 Hz")

    ax2 = fig.add_axes([0.91, 0.05, 0.02, 0.9])
    norm = mpl.colors.Normalize(vmin=-v_max, vmax=v_max)
    cbar = mpl.colorbar.ColorbarBase(
        ax2, norm=norm, orientation="vertical", cmap=mpl.cm.bwr
    )
    cbar.set_label("$T$", rotation=270, labelpad=15, size=12)

    plt.show()

    # Split the problem in tiles (over frequency)
    survey.dobs = global_data.dobs
    survey.std = np.abs(survey.dobs * global_data.relative_error) + global_data.noise_floor


    idx_start = 0
    idx_end = 0
    # do every 5 sources
    cnt = 0
    local_misfits = []
    # for ii, frequency in enumerate(survey.frequencies):
    #     idx_start, idx_end = ii*ntx, ii*ntx+2*ntx
    #     sources = survey.get_sources_by_frequency(frequency)
    #     dobs = survey.dobs[idx_start:idx_end]
    # #         print(dobs.shape, len(src_collect))
    #     delayed_misfit = create_tile_em_misfit(
    #         sources,
    #         survey.dobs[idx_start:idx_end],
    #         survey.std[idx_start:idx_end],
    #         mesh, active_cells, ii, model
    #     )
    #     local_misfits += [delayed_misfit]

    local_misfits = [data_misfit.L2DataMisfit(
        data=global_data, simulation=simulation_g
    )]
    local_misfits[0].W = 1 / survey.std
    local_misfits[0].simulation.model = model
    # local_misfits[0].simulation.Jmatrix

    global_misfit = objective_function.ComboObjectiveFunction(
                    local_misfits
    )




    # for local_misfit in global_misfit.objfcts:
    #     local_misfit.simulation.model = local_misfit.model_map @ model
    #     local_misfit.simulation.Jmatrix
    # simulation_g.model = mstart
    # simulation_g.compute_J()
    #
    # Jvec = simulation_g.Jvec(mstart, v=mstart)

    use_preconditioner = False
    coolingFactor = 2
    coolingRate = 1
    beta0_ratio = 1e1

    # Map for a regularization
    regmap = maps.IdentityMap(nP=int(active_cells.sum()))
    # reg = regularization.Tikhonov(mesh, indActive=global_actinds, mapping=regmap)
    reg = regularization.Sparse(mesh, indActive=active_cells, mapping=regmap)

    print('[INFO] Getting things started on inversion...')
    # set alpha length scales
    reg.alpha_s = 1 # alpha_s
    reg.alpha_x = 1
    reg.alpha_y = 1
    reg.alpha_z = 1
    reg.mref = background_conductivity * np.ones(active_cells.sum())

    opt = optimization.ProjectedGNCG(
        maxIter=5, upper=np.inf, lower=-np.inf, tolCG=1e-4,
        maxIterCG=10,
    )
    invProb = inverse_problem.BaseInvProblem(global_misfit, reg, opt)
    beta = directives.BetaSchedule(
        coolingFactor=coolingFactor, coolingRate=coolingRate
    )
    betaest = directives.BetaEstimate_ByEig(beta0_ratio=beta0_ratio)
    target = directives.TargetMisfit()
    target.target = survey.nD
    saveIter = directives.SaveModelEveryIteration()
    saveIterVar = directives.SaveOutputEveryIteration()
    # Need to have basice saving function
    if use_preconditioner:
        update_Jacobi = directives.UpdatePreconditioner()
        updateSensW = directives.UpdateSensitivityWeights()
        # updateWj = Directives.Update_Wj()
        # directiveList = [
        #     beta, betaest, target, updateSensW, saveIter, update_Jacobi
        # ]
        directiveList = [
            updateSensW, beta, betaest, target, update_Jacobi
        ]
    else:
        directiveList = [
            beta, betaest, target
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

    mesh.writeUBC('OctreeMesh-test.msh', models={'ubc.con': rho_est})

if __name__ == '__main__':
    run()