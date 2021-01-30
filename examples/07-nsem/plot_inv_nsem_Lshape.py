# import SimPEG.dask as simpeg
from discretize import TreeMesh
from SimPEG import maps, utils, data, optimization, maps, objective_function, regularization, inverse_problem, directives, inversion, data_misfit
import discretize
from discretize.utils import mkvc, refine_tree_xyz
from SimPEG.electromagnetics import natural_source as ns
import numpy as np
from dask.distributed import Client, LocalCluster
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from pymatsolver import Pardiso as Solver
from SimPEG.utils import plot2Ddata, surface2ind_topo

def run():
    cluster = LocalCluster(processes=False)
    client = Client(cluster)
    xx, yy = np.meshgrid(np.linspace(-10000, 10000, 101), np.linspace(-10000, 10000, 101))
    zz = np.zeros(np.shape(xx))
    topo_xyz = np.c_[mkvc(xx), mkvc(yy), mkvc(zz)]

    # create receivers
    rx_x, rx_y = np.meshgrid(np.arange(0, 6500, 2000), np.arange(0, 6500, 2000))
    rx_loc = np.hstack((mkvc(rx_x, 2), mkvc(rx_y, 2), np.zeros((np.prod(rx_x.shape), 1))))
    rx_loc[:, 2] = 0

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
        mesh, rx_loc, octree_levels=[4, 4], method="radial", finalize=False
    )

    # Refine core mesh region
    xp, yp, zp = np.meshgrid([-250.0, 250.0], [-250.0, 250.0], [-300.0, 0.0])
    xyz = np.c_[mkvc(xp), mkvc(yp), mkvc(zp)]
    mesh = refine_tree_xyz(mesh, xyz, octree_levels=[0, 2, 4], method="box", finalize=True)

    air_conductivity = np.log(1e-8)

    # Find cells that are active in the forward modeling (cells below surface)
    active_cells = surface2ind_topo(mesh, topo_xyz)

    # Define mapping from model to active cells
    expmap = maps.ExpMap(mesh)
    model_map = expmap * maps.InjectActiveCells(mesh, active_cells, air_conductivity)

    # Define model. Models in SimPEG are vector arrays
    block_1 = utils.ModelBuilder.getIndicesBlock([1600, 4600, -1200], [4600, 3400, -400], mesh.gridCC)[
        0].tolist()  # Porphyry Intrusion
    block_2 = utils.ModelBuilder.getIndicesBlock([3400, 4600, -1200], [4600, 1200, -400], mesh.gridCC)[
        0].tolist()  # Porphyry Intrusion
    # print(block_1)
    background = 100  # ohm-m
    target = 1  # ohm-m

    ## set elvation to a hieght
    active = mesh.gridCC[:, 2] < 1
    sig = np.ones(mesh.nC) * 1 / background

    ## assign conductivities of target
    sig[block_1] = 1 / target
    sig[block_2] = 1 / target
    sig[~active] = 1e-8
    model_true = np.log(sig)

    # create background conductivity model
    sigBG = np.zeros(mesh.nC) + 1 / background
    sigBG[~active] = 1e-8

    # Make a receiver list
    rxList = []
    for rx_orientation in ['xx', 'xy', 'yx', 'yy']:
    #     rxList.append(ns.Rx.Point3DComplexResistivity(locations=None, locations_e=rx_loc, locations_h=rx_loc, orientation=rx_orientation, component='apparent_resistivity'))
    #     rxList.append(ns.Rx.Point3DComplexResistivity(locations=None,locations_e=rx_loc, locations_h=rx_loc, orientation=rx_orientation, component='phase'))
        rxList.append(ns.Rx.Point3DImpedance(rx_loc, rx_orientation, 'real'))
        rxList.append(ns.Rx.Point3DImpedance(rx_loc, rx_orientation, 'imag'))
    # for rx_orientation in ['zx', 'zy']:
    #     rxList.append(ns.Rx.Point3DTipper(rx_loc, rx_orientation, 'real'))
    #     rxList.append(ns.Rx.Point3DTipper(rx_loc, rx_orientation, 'imag'))

    # Source list
    srcList = [
        ns.Src.Planewave_xy_1Dprimary(rxList, freq, sigma_primary=sigBG)
        for freq in [10, 50, 200]
    ]

    # Survey MT
    survey = ns.Survey(srcList)
    survey.m_true = model_true

    actMap = maps.InjectActiveCells(
        mesh=mesh, indActive=active, valInactive=np.log(1e-8)
    )
    mapping = maps.ExpMap(mesh) * actMap

    # Setup the problem object
    sim = ns.simulation.Simulation3DPrimarySecondary(mesh, survey=survey,
                                               sigmaMap=mapping,
                                               sigmaPrimary=sigBG,
                                               solver=Solver)

    sim.model = sigBG[active]
    # fields = sim.fields()

    # v = np.random.rand(survey.nD,)
    # # print problem.PropMap.PropModel.nP
    # w = np.random.rand(active.sum(),)
    # # np.random.seed(1983)
    # # fwd = sim.getJ(sigBG[active])
    # # fwd = sim.Jtvec(sigBG[active], v)
    # fwd = sim.Jvec(sigBG[active], w)

    sim.survey.dtrue = sim.dpred(model_true[active])
    # create observations
    sim.survey.dobs = survey.dtrue

    # Assign uncertainties
    std = 0.5  # 5% std
    sim.survey.std = np.abs(survey.dobs * std)
    # make data object
    fwd_data = data.Data(sim.survey)

    num_station = rx_loc.shape[0]
    freqs = ['10', '50', '200']
    num_sets = int(survey.dobs.shape[0] / len(freqs))
    dnew = np.reshape(survey.dobs, (3, num_sets))
    stdnew = np.reshape(survey.std, (3, num_sets))

    cnt = 0

    for freq in freqs:
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
            im = utils.plot_utils.plot2Ddata(rx_loc, dnew[cnt, num_station*rx_orientation:num_station*(rx_orientation + 1)], ax=ax)
            ax.set_title(comps_[rx_orientation])
            plt.colorbar(im[0])
            cnt_comp += 1
        cnt += 1
        fig1.suptitle(freq + ' Hz Forward Data', fontsize='20')
    plt.show()
    print(mkvc(stdnew).shape)

    ## Fill the data object
    fwd_data.dobs = sim.survey.dobs
    fwd_data.standard_deviation = mkvc(stdnew)  # sim.survey.std
    sim.survey.std = mkvc(stdnew)
    survey.std = mkvc(stdnew)

    # Set the conductivity values
    sig_half = 0.01
    sig_air = 1e-8
    # Make the background model
    sigma_0 = np.ones(mesh.nC) * sig_air
    sigma_0[active] = sig_half
    m_0 = np.log(sigma_0[active])

    Wd = 1 / survey.std
    plt.hist(Wd, 100)
    plt.show()
    # Setup the inversion proceedure
    # Define a counter
    C = utils.Counter()
    # Optimization
    opt = optimization.ProjectedGNCG(maxIter=1, upper=np.inf, lower=-np.inf)
    opt.counter = C
    opt.maxIterCG = 20
    # opt.LSshorten = 0.5
    opt.remember('xc')
    # Data misfit
    dmis = data_misfit.L2DataMisfit(data=fwd_data, simulation=sim)
    dmis.W = Wd

    local_misfits = [data_misfit.L2DataMisfit(
        data=fwd_data, simulation=sim
    )]
    local_misfits[0].W = Wd
    local_misfits[0].simulation.model = m_0
    global_misfit = objective_function.ComboObjectiveFunction(
        local_misfits
    )

    # Regularization
    regmap = maps.IdentityMap(nP=int(active.sum()))
    reg = regularization.Simple(mesh, indActive=active, mapping=regmap)

    # Inversion problem
    invProb = inverse_problem.BaseInvProblem(global_misfit, reg, opt)
    invProb.counter = C
    # Beta schedule
    beta = directives.BetaSchedule()
    beta.coolingRate = 2.
    beta.coolingFactor = 4.
    # Initial estimate of beta
    beta_est = directives.BetaEstimate_ByEig(beta0_ratio=1e0, method="old")
    # Target misfit stop
    targmis = directives.TargetMisfit()
    saveIter = directives.SaveModelEveryIteration()
    # Create an inversion object
    directive_list = [beta, beta_est, targmis, saveIter]
    inv = inversion.BaseInversion(invProb, directiveList=directive_list)



    import time
    start = time.time()
    # Run the inversion
    mopt = inv.run(m_0)
    print('Inversion took {0} seconds'.format(time.time() - start))

if __name__ == '__main__':
    run()