from SimPEG import (np, Mesh, Maps, Utils, DataMisfit, Regularization,
                    Optimization, Inversion, InvProblem, Directives)
from SimPEG import SolverLU
from SimPEG.EM import FDEM, TDEM, mu_0
import warnings
import matplotlib.pyplot as plt


def run(plotIt=True):
    """
    1D FDEM and TDEM inversions
    ===========================

    This example is used in the paper Heagy et al 2016 (in prep)

    """

    # Set up cylindrically symmeric mesh
    cs, ncx, ncz, npad = 10., 15, 25, 13  # padded cyl mesh
    hx = [(cs, ncx), (cs, npad, 1.3)]
    hz = [(cs, npad, -1.3), (cs, ncz), (cs, npad, 1.3)]
    mesh = Mesh.CylMesh([hx, 1, hz], '00C')

    # Conductivity model
    layerz = np.r_[-200., -100.]
    layer = (mesh.vectorCCz >= layerz[0]) & (mesh.vectorCCz <= layerz[1])
    active = mesh.vectorCCz < 0.
    sig_half = 1e-2  # Half-space conductivity
    sig_air = 1e-8  # Air conductivity
    sig_layer = 5e-2  # Layer conductivity
    sigma = np.ones(mesh.nCz)*sig_air
    sigma[active] = sig_half
    sigma[layer] = sig_layer

    # Mapping
    actMap = Maps.InjectActiveCells(mesh, active, np.log(1e-8), nC=mesh.nCz)
    mapping = Maps.ExpMap(mesh) * Maps.SurjectVertical1D(mesh) * actMap
    mtrue = np.log(sigma[active])

    # ----- FDEM problem & survey -----
    rxlocs = Utils.ndgrid([np.r_[50.], np.r_[0], np.r_[0.]])
    bzi = FDEM.Rx.Point_bSecondary(rxlocs, 'z', 'real')
    bzr = FDEM.Rx.Point_bSecondary(rxlocs, 'z', 'imag')

    freqs = np.logspace(2, 3, 5)
    srcLoc = np.array([0., 0., 0.])

    print('min skin depth = ', 500./np.sqrt(freqs.max() * sig_half),
          'max skin depth = ', 500./np.sqrt(freqs.min() * sig_half))
    print('max x ', mesh.vectorCCx.max(), 'min z ', mesh.vectorCCz.min(),
          'max z ', mesh.vectorCCz.max())

    srcList = []
    [srcList.append(FDEM.Src.MagDipole([bzr, bzi], freq, srcLoc,
                                       orientation='Z')) for freq in freqs]

    surveyFD = FDEM.Survey(srcList)
    prbFD = FDEM.Problem3D_b(mesh, mapping=mapping)
    prbFD.pair(surveyFD)
    std = 0.03
    surveyFD.makeSyntheticData(mtrue, std)
    surveyFD.eps = np.linalg.norm(surveyFD.dtrue)*1e-5

    # FDEM inversion
    np.random.seed(1)
    dmisfit = DataMisfit.l2_DataMisfit(surveyFD)
    regMesh = Mesh.TensorMesh([mesh.hz[mapping.maps[-1].indActive]])
    reg = Regularization.Simple(regMesh)
    opt = Optimization.InexactGaussNewton(maxIterCG=10)
    invProb = InvProblem.BaseInvProblem(dmisfit, reg, opt)

    # Inversion Directives
    beta = Directives.BetaSchedule(coolingFactor=4, coolingRate=3)
    betaest = Directives.BetaEstimate_ByEig(beta0_ratio=2.)
    target = Directives.TargetMisfit()

    inv = Inversion.BaseInversion(invProb, directiveList=[beta, betaest, target])
    m0 = np.log(np.ones(mtrue.size)*sig_half)
    reg.alpha_s = 5e-1
    reg.alpha_x = 1.
    prbFD.counter = opt.counter = Utils.Counter()
    opt.remember('xc')
    moptFD = inv.run(m0)

    # TDEM problem
    times = np.logspace(-4, np.log10(2e-3), 10)
    print('min diffusion distance ', 1.28*np.sqrt(times.min()/(sig_half*mu_0)),
          'max diffusion distance ', 1.28*np.sqrt(times.max()/(sig_half*mu_0)))
    rx = TDEM.Rx(rxlocs, times, 'bz')
    src = TDEM.Src.MagDipole([rx], waveform=TDEM.Src.StepOffWaveform(),
                             loc=srcLoc)  # same src location as FDEM problem

    surveyTD = TDEM.Survey([src])
    prbTD = TDEM.Problem3D_b(mesh, mapping=mapping)
    prbTD.timeSteps = [(5e-5, 10), (1e-4, 10), (5e-4, 10)]
    prbTD.pair(surveyTD)
    prbTD.Solver = SolverLU

    std = 0.03
    surveyTD.makeSyntheticData(mtrue, std)
    surveyTD.std = std
    surveyTD.eps = np.linalg.norm(surveyTD.dtrue)*1e-5

    # TDEM inversion
    dmisfit = DataMisfit.l2_DataMisfit(surveyTD)
    regMesh = Mesh.TensorMesh([mesh.hz[mapping.maps[-1].indActive]])
    reg = Regularization.Simple(regMesh)
    opt = Optimization.InexactGaussNewton(maxIterCG=10)
    invProb = InvProblem.BaseInvProblem(dmisfit, reg, opt)

    # Inversion Directives
    beta = Directives.BetaSchedule(coolingFactor=4, coolingRate=3)
    betaest = Directives.BetaEstimate_ByEig(beta0_ratio=2.)
    target = Directives.TargetMisfit()

    inv = Inversion.BaseInversion(invProb, directiveList=[beta, betaest, target])
    m0 = np.log(np.ones(mtrue.size)*sig_half)
    reg.alpha_s = 5e-1
    reg.alpha_x = 1.
    prbTD.counter = opt.counter = Utils.Counter()
    opt.remember('xc')
    moptTD = inv.run(m0)

    if plotIt:
        import matplotlib
        fig = plt.figure(figsize = (10, 8))
        ax0 = plt.subplot2grid((2, 2), (0, 0), rowspan=2)
        ax1 = plt.subplot2grid((2, 2), (0, 1))
        ax2 = plt.subplot2grid((2, 2), (1, 1))

        fs = 13  # fontsize
        matplotlib.rcParams['font.size'] = fs

        # Plot the model
        ax0.semilogx(sigma[active], mesh.vectorCCz[active], 'k-', lw=2)
        ax0.semilogx(np.exp(moptFD), mesh.vectorCCz[active], 'bo', ms=6)
        ax0.semilogx(np.exp(moptTD), mesh.vectorCCz[active], 'r*', ms=10)
        ax0.set_ylim(-700, 0)
        ax0.set_xlim(5e-3, 1e-1)

        ax0.set_xlabel('Conductivity (S/m)', fontsize=fs)
        ax0.set_ylabel('Depth (m)', fontsize=fs)
        ax0.grid(which='both', color='k', alpha=0.5, linestyle='-',
                 linewidth=0.2)
        ax0.legend(['True', 'FDEM', 'TDEM'], fontsize=fs, loc=4)

        # plot the data misfits - negative b/c we choose positive to be in the
        # direction of primary

        ax1.plot(freqs, -surveyFD.dobs[::2], 'k-', lw=2)
        ax1.plot(freqs, -surveyFD.dobs[1::2], 'k--', lw=2)

        dpredFD = surveyFD.dpred(moptTD)
        ax1.loglog(freqs, -dpredFD[::2], 'bo', ms=6)
        ax1.loglog(freqs, -dpredFD[1::2], 'b+', markeredgewidth=2., ms=10)

        ax2.loglog(times, surveyTD.dobs, 'k-', lw=2)
        ax2.loglog(times, surveyTD.dpred(moptTD), 'r*', ms=10)
        ax2.set_xlim(times.min(), times.max())

        # Labels, gridlines, etc
        ax2.grid(which='both', alpha=0.5, linestyle='-', linewidth=0.2)
        ax1.grid(which='both', alpha=0.5, linestyle='-', linewidth=0.2)

        ax1.set_xlabel('Frequency (Hz)', fontsize=fs)
        ax1.set_ylabel('Vertical magnetic field (-T)', fontsize=fs)

        ax2.set_xlabel('Time (s)', fontsize=fs)
        ax2.set_ylabel('Vertical magnetic field (-T)', fontsize=fs)

        ax2.legend(("Obs", "Pred"), fontsize=fs)
        ax1.legend(("Obs (real)", "Obs (imag)", "Pred (real)", "Pred (imag)"),
                   fontsize=fs)
        ax1.set_xlim(freqs.max(), freqs.min())

        ax0.set_title("(a) Recovered Models", fontsize=fs)
        ax1.set_title("(b) FDEM observed vs. predicted", fontsize=fs)
        ax2.set_title("(c) TDEM observed vs. predicted", fontsize=fs)

        plt.tight_layout(pad=1.5)
        plt.show()

if __name__ == '__main__':
    run(plotIt=True)
