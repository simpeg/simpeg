from SimPEG import np, Mesh, Maps, Utils, DataMisfit, Regularization, Optimization, Inversion, InvProblem, Directives
from SimPEG import SolverLU
from SimPEG.EM import FDEM, TDEM, mu_0
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.size'] = 14

def run(plotIt=True):
    # Set up cylindrically symmeric mesh
    cs, ncx, ncz, npad = 10., 15, 25, 13  # padded cyl mesh
    hx = [(cs,ncx), (cs,npad,1.3)]
    hz = [(cs,npad,-1.3), (cs,ncz), (cs,npad,1.3)]
    mesh = Mesh.CylMesh([hx,1,hz], '00C')

    # Conductivity model
    layerz = np.r_[-200., -100.]
    layer = (mesh.vectorCCz>=layerz[0]) & (mesh.vectorCCz<=layerz[1])
    active = mesh.vectorCCz<0.
    sig_half  = 1e-2  # Half-space conductivity
    sig_air   = 1e-8  # Air conductivity
    sig_layer = 5e-2  # Layer conductivity
    sigma = np.ones(mesh.nCz)*sig_air
    sigma[active] = sig_half
    sigma[layer] = sig_layer

    # Mapping
    actMap = Maps.InjectActiveCells(mesh, active, np.log(1e-8), nC=mesh.nCz)
    mapping = Maps.ExpMap(mesh) * Maps.SurjectVertical1D(mesh) * actMap
    mtrue = np.log(sigma[active])

    # FDEM problem & survey
    rxlocs = Utils.ndgrid([np.r_[50.], np.r_[0], np.r_[0.]])
    bzi = FDEM.Rx.Point_b(rxlocs, 'z', 'real')
    bzr = FDEM.Rx.Point_b(rxlocs, 'z', 'imag')

    freqs = np.logspace(2, 3, 5)
    srcLoc = np.array([0., 0., 0.])

    print 'min skin depth = ', 500./np.sqrt(freqs.max() * sig_half), 'max skin depth = ', 500./np.sqrt(freqs.min() * sig_half)
    print 'max x ', mesh.vectorCCx.max(), 'min z ', mesh.vectorCCz.min(), 'max z ', mesh.vectorCCz.max()

    srcList = []
    [srcList.append(FDEM.Src.MagDipole([bzr, bzi],freq, srcLoc,orientation='Z')) for freq in freqs]

    surveyFD = FDEM.Survey(srcList)
    prbFD = FDEM.Problem3D_b(mesh, mapping=mapping)
    prbFD.pair(surveyFD)
    std = 0.02
    surveyFD.makeSyntheticData(mtrue, std)
    surveyFD.eps = np.linalg.norm(surveyFD.dtrue)*1e-5

    # FDEM inversion
    np.random.seed(1)
    dmisfit = DataMisfit.l2_DataMisfit(surveyFD)
    regMesh = Mesh.TensorMesh([mesh.hz[mapping.maps[-1].indActive]])
    reg = Regularization.Simple(regMesh)
    opt = Optimization.InexactGaussNewton(maxIterCG=3, maxIter=7)
    invProb = InvProblem.BaseInvProblem(dmisfit, reg, opt)
    # Inversion Directives
    beta = Directives.BetaSchedule(coolingFactor=5, coolingRate=3)
    # betaest = Directives.BetaEstimate_ByEig(beta0_ratio=10.)
    invProb.beta = 1.
    target = Directives.TargetMisfit()

    inv = Inversion.BaseInversion(invProb, directiveList=[beta,target])
    m0 = np.log(np.ones(mtrue.size)*sig_half)
    reg.alpha_s = 1e-1
    reg.alpha_x = 1.
    prbFD.counter = opt.counter = Utils.Counter()
    opt.LSshorten = 0.5
    opt.tolG = 1e-10
    opt.eps = 1e-10
    opt.remember('xc')
    moptFD = inv.run(m0)

    # TDEM problem
    times = np.logspace(-4, np.log10(2e-3), 10)
    print 'min diffusion distance ', 1.28*np.sqrt(times.min()/(sig_half*mu_0)), 'max diffusion distance ', 1.28*np.sqrt(times.max()/(sig_half*mu_0))
    rx = TDEM.Rx(rxlocs, times, 'bz')
    src = TDEM.Src.MagDipole([rx], waveform=TDEM.Src.StepOffWaveform(), loc=srcLoc) # same src location as FDEM problem

    surveyTD = TDEM.Survey([src])
    prbTD = TDEM.Problem_b(mesh, mapping=mapping)
    prbTD.timeSteps = [(5e-5, 10),(1e-4, 10),(5e-4, 10)]
    prbTD.pair(surveyTD)
    prbTD.Solver = SolverLU

    std = 0.05
    surveyTD.makeSyntheticData(mtrue, std)
    surveyTD.std = std
    surveyTD.eps = np.linalg.norm(surveyTD.dtrue)*1e-5

    # TDEM inversion
    dmisfit = DataMisfit.l2_DataMisfit(surveyTD)
    regMesh = Mesh.TensorMesh([mesh.hz[mapping.maps[-1].indActive]])
    reg = Regularization.Simple(regMesh)
    opt = Optimization.InexactGaussNewton(maxIterCG=3, maxIter=7)
    invProb = InvProblem.BaseInvProblem(dmisfit, reg, opt)

    # Inversion Directives
    beta = Directives.BetaSchedule(coolingFactor=5, coolingRate=3)
    invProb.beta = 1.
    # betaest = Directives.BetaEstimate_ByEig(beta0_ratio=1.)
    target = Directives.TargetMisfit()

    inv = Inversion.BaseInversion(invProb, directiveList=[beta, target])
    m0 = np.log(np.ones(mtrue.size)*sig_half)
    reg.alpha_s = 1e-1
    reg.alpha_x = 1.
    prbTD.counter = opt.counter = Utils.Counter()
    opt.LSshorten = 0.5
    opt.remember('xc')
    moptTD = inv.run(m0)

    if plotIt:
        fig, ax = plt.subplots(1,1, figsize = (4, 6))
        plt.semilogx(sigma[active], mesh.vectorCCz[active], 'k-', lw=2)
        plt.semilogx(np.exp(moptFD), mesh.vectorCCz[active], 'ko', ms=3)
        plt.semilogx(np.exp(moptTD), mesh.vectorCCz[active], 'k*')
        ax.set_ylim(-1000, 0)
        ax.set_xlim(5e-3, 1e-1)

        ax.set_xlabel('Conductivity (S/m)', fontsize = 14)
        ax.set_ylabel('Depth (m)', fontsize = 14)
        ax.grid(color='k', alpha=0.5, linestyle='dashed', linewidth=0.5)
        plt.legend(['True', 'Pred (FD)', 'Pred (TD)'], fontsize=13, loc=4)
        plt.show()

        fig = plt.figure(figsize = (10*1.3, 5*1.3))
        ax2 = plt.subplot(122)
        ax2.plot(times, surveyTD.dobs, 'k-', lw=2)
        ax2.plot(times, surveyTD.dpred(moptTD), 'ko', ms=4)
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.set_xlim(times.min(), times.max())
        ax1 = plt.subplot(121)
        ax1.plot(freqs, -surveyFD.dobs[::2], 'k-', lw=2)
        ax1.plot(freqs, -surveyFD.dobs[1::2], 'k--', lw=2)
        dpredFD = surveyFD.dpred(moptTD)
        ax1.plot(freqs, -dpredFD[::2], 'ko', ms=4)
        ax1.plot(freqs, -dpredFD[1::2], 'k+', markeredgewidth=2., ms=10)
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax2.set_xlabel('Time (s)', fontsize = 14)
        ax1.set_xlabel('Frequency (Hz)', fontsize = 14)
        ax1.set_ylabel('Vertical magnetic field (T)', fontsize = 14)
        ax2.grid(True,which='minor')
        ax1.grid(True,which='minor')
        ax2.set_title("(b) TD observed vs. predicted", fontsize = 14)
        ax1.set_title("(a) FD observed vs. predicted", fontsize = 14)
        ax2.legend(("Obs", "Pred"), fontsize = 12)
        ax1.legend(("Obs", "Pred (real)", "Pred (imag)"), fontsize = 12, loc=3)
        ax1.set_xlim(freqs.max(), freqs.min())
        plt.show()

if __name__ == '__main__':
    run()
