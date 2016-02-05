from SimPEG import *
import SimPEG.EM as EM
from SimPEG.EM import mu_0


def run(plotIt=True):
    """
        EM: TDEM: 1D: Inversion
        =======================

        Here we will create and run a TDEM 1D inversion.

    """

    cs, ncx, ncz, npad = 5., 25, 15, 15
    hx = [(cs,ncx), (cs,npad,1.3)]
    hz = [(cs,npad,-1.3), (cs,ncz), (cs,npad,1.3)]
    mesh = Mesh.CylMesh([hx,1,hz], '00C')

    active = mesh.vectorCCz<0.
    layer = (mesh.vectorCCz<0.) & (mesh.vectorCCz>=-100.)
    actMap = Maps.ActiveCells(mesh, active, np.log(1e-8), nC=mesh.nCz)
    mapping = Maps.ExpMap(mesh) * Maps.Vertical1DMap(mesh) * actMap
    sig_half = 2e-3
    sig_air = 1e-8
    sig_layer = 1e-3
    sigma = np.ones(mesh.nCz)*sig_air
    sigma[active] = sig_half
    sigma[layer] = sig_layer
    mtrue = np.log(sigma[active])


    if plotIt:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1,1, figsize = (3, 6))
        plt.semilogx(sigma[active], mesh.vectorCCz[active])
        ax.set_ylim(-600, 0)
        ax.set_xlim(1e-4, 1e-2)
        ax.set_xlabel('Conductivity (S/m)', fontsize = 14)
        ax.set_ylabel('Depth (m)', fontsize = 14)
        ax.grid(color='k', alpha=0.5, linestyle='dashed', linewidth=0.5)


    rxOffset=1e-3
    rx = EM.TDEM.RxTDEM(np.array([[rxOffset, 0., 30]]), np.logspace(-5,-3, 31), 'bz')
    src = EM.TDEM.SrcTDEM_VMD_MVP([rx], np.array([0., 0., 80]))
    survey = EM.TDEM.SurveyTDEM([src])
    prb = EM.TDEM.ProblemTDEM_b(mesh, mapping=mapping)

    prb.Solver = SolverLU
    prb.timeSteps = [(1e-06, 20),(1e-05, 20), (0.0001, 20)]
    prb.pair(survey)

    # create observed data
    std = 0.05
    
    survey.dobs = survey.makeSyntheticData(mtrue,std)
    survey.std = std 
    survey.eps = 1e-5*np.linalg.norm(survey.dobs)

    if plotIt:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1,1, figsize = (10, 6))
        ax.loglog(rx.times, survey.dtrue, 'b.-')
        ax.loglog(rx.times, survey.dobs, 'r.-')
        ax.legend(('Noisefree', '$d^{obs}$'), fontsize = 16)
        ax.set_xlabel('Time (s)', fontsize = 14)
        ax.set_ylabel('$B_z$ (T)', fontsize = 16)
        ax.set_xlabel('Time (s)', fontsize = 14)
        ax.grid(color='k', alpha=0.5, linestyle='dashed', linewidth=0.5)

    dmisfit = DataMisfit.l2_DataMisfit(survey)
    regMesh = Mesh.TensorMesh([mesh.hz[mapping.maps[-1].indActive]])
    reg = Regularization.Tikhonov(regMesh)
    opt = Optimization.InexactGaussNewton(maxIter = 5)
    invProb = InvProblem.BaseInvProblem(dmisfit, reg, opt)

    # Create an inversion object
    beta = Directives.BetaSchedule(coolingFactor=5, coolingRate=2)
    betaest = Directives.BetaEstimate_ByEig(beta0_ratio=1e0)
    inv = Inversion.BaseInversion(invProb, directiveList=[beta,betaest])
    m0 = np.log(np.ones(mtrue.size)*sig_half)
    reg.alpha_s = 1e-2
    reg.alpha_x = 1.
    prb.counter = opt.counter = Utils.Counter()
    opt.LSshorten = 0.5
    opt.remember('xc')

    mopt = inv.run(m0)

    if plotIt:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1,1, figsize = (3, 6))
        plt.semilogx(sigma[active], mesh.vectorCCz[active])
        plt.semilogx(np.exp(mopt), mesh.vectorCCz[active])
        ax.set_ylim(-600, 0)
        ax.set_xlim(1e-4, 1e-2)
        ax.set_xlabel('Conductivity (S/m)', fontsize = 14)
        ax.set_ylabel('Depth (m)', fontsize = 14)
        ax.grid(color='k', alpha=0.5, linestyle='dashed', linewidth=0.5)
        plt.legend(['$\sigma_{true}$', '$\sigma_{pred}$'])
        plt.show()


if __name__ == '__main__':
    run()
