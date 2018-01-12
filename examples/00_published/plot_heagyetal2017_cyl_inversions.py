"""
Heagy et al., 2017 1D FDEM and TDEM inversions
==============================================

Here, we perform a 1D inversion using both the frequency and time domain
codes. The forward simulations are conducted on a cylindrically symmetric
mesh. The source is a point magnetic dipole source.

This example is used in the paper

    Lindsey J. Heagy, Rowan Cockett, Seogi Kang, Gudni K. Rosenkjaer, Douglas
    W. Oldenburg, A framework for simulation and inversion in electromagnetics,
    Computers & Geosciences, Volume 107, 2017, Pages 1-19, ISSN 0098-3004,
    http://dx.doi.org/10.1016/j.cageo.2017.06.018.

This example is on figshare:
https://doi.org/10.6084/m9.figshare.5035175

"""
from SimPEG import (
    Mesh, Maps, Utils, DataMisfit, Regularization,
    Optimization, Inversion, InvProblem, Directives
)
import numpy as np
from SimPEG.EM import FDEM, TDEM, mu_0
import matplotlib.pyplot as plt
import matplotlib
try:
    from pymatsolver import Pardiso as Solver
except ImportError:
    from SimPEG import SolverLU as Solver


def run(plotIt=True, saveFig=False):

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

    # ----- FDEM problem & survey ----- #
    rxlocs = Utils.ndgrid([np.r_[50.], np.r_[0], np.r_[0.]])
    bzr = FDEM.Rx.Point_bSecondary(rxlocs, 'z', 'real')
    bzi = FDEM.Rx.Point_bSecondary(rxlocs, 'z', 'imag')

    freqs = np.logspace(2, 3, 5)
    srcLoc = np.array([0., 0., 0.])

    print('min skin depth = ', 500./np.sqrt(freqs.max() * sig_half),
          'max skin depth = ', 500./np.sqrt(freqs.min() * sig_half))
    print('max x ', mesh.vectorCCx.max(), 'min z ', mesh.vectorCCz.min(),
          'max z ', mesh.vectorCCz.max())

    srcList = [
        FDEM.Src.MagDipole([bzr, bzi], freq, srcLoc, orientation='Z')
        for freq in freqs
    ]

    surveyFD = FDEM.Survey(srcList)
    prbFD = FDEM.Problem3D_b(mesh, sigmaMap=mapping, Solver=Solver)
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
    directiveList = [beta, betaest, target]

    inv = Inversion.BaseInversion(invProb, directiveList=directiveList)
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
    rx = TDEM.Rx.Point_b(rxlocs, times, 'z')
    src = TDEM.Src.MagDipole(
        [rx],
        waveform=TDEM.Src.StepOffWaveform(),
        loc=srcLoc  # same src location as FDEM problem
    )

    surveyTD = TDEM.Survey([src])
    prbTD = TDEM.Problem3D_b(mesh, sigmaMap=mapping, Solver=Solver)
    prbTD.timeSteps = [(5e-5, 10), (1e-4, 10), (5e-4, 10)]
    prbTD.pair(surveyTD)

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

    # directives
    beta = Directives.BetaSchedule(coolingFactor=4, coolingRate=3)
    betaest = Directives.BetaEstimate_ByEig(beta0_ratio=2.)
    target = Directives.TargetMisfit()
    directiveList = [beta, betaest, target]

    inv = Inversion.BaseInversion(invProb, directiveList=directiveList)
    m0 = np.log(np.ones(mtrue.size)*sig_half)
    reg.alpha_s = 5e-1
    reg.alpha_x = 1.
    prbTD.counter = opt.counter = Utils.Counter()
    opt.remember('xc')
    moptTD = inv.run(m0)

    # Plot the results
    if plotIt:
        plt.figure(figsize=(10, 8))
        ax0 = plt.subplot2grid((2, 2), (0, 0), rowspan=2)
        ax1 = plt.subplot2grid((2, 2), (0, 1))
        ax2 = plt.subplot2grid((2, 2), (1, 1))

        fs = 13  # fontsize
        matplotlib.rcParams['font.size'] = fs

        # Plot the model
        # z_true = np.repeat(mesh.vectorCCz[active][1:], 2, axis=0)
        # z_true = np.r_[mesh.vectorCCz[active][0], z_true, mesh.vectorCCz[active][-1]]
        activeN = mesh.vectorNz <= 0. + cs/2.
        z_true = np.repeat(mesh.vectorNz[activeN][1:-1], 2, axis=0)
        z_true = np.r_[mesh.vectorNz[activeN][0], z_true, mesh.vectorNz[activeN][-1]]
        sigma_true = np.repeat(sigma[active], 2, axis=0)

        ax0.semilogx(
            sigma_true, z_true, 'k-', lw=2, label="True"
        )

        ax0.semilogx(
            np.exp(moptFD), mesh.vectorCCz[active], 'bo', ms=6,
            markeredgecolor='k', markeredgewidth=0.5, label="FDEM"
        )
        ax0.semilogx(
            np.exp(moptTD), mesh.vectorCCz[active], 'r*', ms=10,
            markeredgecolor='k', markeredgewidth=0.5, label="TDEM"
        )
        ax0.set_ylim(-700, 0)
        ax0.set_xlim(5e-3, 1e-1)

        ax0.set_xlabel('Conductivity (S/m)', fontsize=fs)
        ax0.set_ylabel('Depth (m)', fontsize=fs)
        ax0.grid(
            which='both', color='k', alpha=0.5, linestyle='-', linewidth=0.2
        )
        ax0.legend(fontsize=fs, loc=4)

        # plot the data misfits - negative b/c we choose positive to be in the
        # direction of primary

        ax1.plot(freqs, -surveyFD.dobs[::2], 'k-', lw=2, label="Obs (real)")
        ax1.plot(freqs, -surveyFD.dobs[1::2], 'k--', lw=2, label="Obs (imag)")

        dpredFD = surveyFD.dpred(moptTD)
        ax1.loglog(
            freqs, -dpredFD[::2], 'bo', ms=6, markeredgecolor='k',
            markeredgewidth=0.5, label="Pred (real)"
        )
        ax1.loglog(
            freqs, -dpredFD[1::2], 'b+', ms=10, markeredgewidth=2.,
            label="Pred (imag)"
        )

        ax2.loglog(times, surveyTD.dobs, 'k-', lw=2, label='Obs')
        ax2.loglog(
            times, surveyTD.dpred(moptTD), 'r*', ms=10, markeredgecolor='k',
            markeredgewidth=0.5, label='Pred'
        )
        ax2.set_xlim(times.min() - 1e-5, times.max() + 1e-4)

        # Labels, gridlines, etc
        ax2.grid(which='both', alpha=0.5, linestyle='-', linewidth=0.2)
        ax1.grid(which='both', alpha=0.5, linestyle='-', linewidth=0.2)

        ax1.set_xlabel('Frequency (Hz)', fontsize=fs)
        ax1.set_ylabel('Vertical magnetic field (-T)', fontsize=fs)

        ax2.set_xlabel('Time (s)', fontsize=fs)
        ax2.set_ylabel('Vertical magnetic field (T)', fontsize=fs)

        ax2.legend(fontsize=fs, loc=3)
        ax1.legend(fontsize=fs, loc=3)
        ax1.set_xlim(freqs.max() + 1e2, freqs.min() - 1e1)

        ax0.set_title("(a) Recovered Models", fontsize=fs)
        ax1.set_title("(b) FDEM observed vs. predicted", fontsize=fs)
        ax2.set_title("(c) TDEM observed vs. predicted", fontsize=fs)

        plt.tight_layout(pad=1.5)

        if saveFig is True:
            plt.savefig('example1.png', dpi=600)

if __name__ == '__main__':
    run(plotIt=True, saveFig=True)
    plt.show()
