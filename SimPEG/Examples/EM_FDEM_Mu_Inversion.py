from SimPEG import (
    np, Mesh, Maps, Utils, DataMisfit, Regularization,
    Optimization, Inversion, InvProblem, Directives
)
from SimPEG.EM import FDEM, TDEM, mu_0
import matplotlib.pyplot as plt
import matplotlib
try:
    from pymatsolver import PardisoSolver as Solver
except ImportError:
    from SimPEG import SolverLU as Solver


def run(plotIt=True):
    """
    1D FDEM Mu Inversion
    ====================

    1D inversion of Magnetic Susceptibility from FDEM data assuming a fixed
    electrical conductivity

    """

    # Set up cylindrically symmeric mesh
    cs, ncx, ncz, npad = 10., 15, 25, 13  # padded cyl mesh
    hx = [(cs, ncx), (cs, npad, 1.3)]
    hz = [(cs, npad, -1.3), (cs, ncz), (cs, npad, 1.3)]
    mesh = Mesh.CylMesh([hx, 1, hz], '00C')

    # Geologic Parameters model
    layerz = np.r_[-100., -50.]
    layer = (mesh.vectorCCz >= layerz[0]) & (mesh.vectorCCz <= layerz[1])
    active = mesh.vectorCCz < 0.

    # Electrical Conductivity
    sig_half = 1e-2  # Half-space conductivity
    sig_air = 1e-8  # Air conductivity
    sig_layer = 1e-2  # Layer conductivity
    sigma = np.ones(mesh.nCz)*sig_air
    sigma[active] = sig_half
    sigma[layer] = sig_layer

    # mur - relative magnetic permeability
    mur_half = 1.
    mur_air = 1.
    mur_layer = 2.
    mur = np.ones(mesh.nCz)*mur_air
    mur[active] = mur_half
    mur[layer] = mur_layer

    mtrue = mur[active]

    # Maps
    actMap = Maps.InjectActiveCells(mesh, active, mur_air, nC=mesh.nCz)
    surj1Dmap = Maps.SurjectVertical1D(mesh)
    murMap = Maps.MuRelative(mesh)

    # Mapping
    muMap =  murMap * surj1Dmap * actMap

    # ----- FDEM problem & survey -----
    rxlocs = Utils.ndgrid([np.r_[10.], np.r_[0], np.r_[30.]])
    bzr = FDEM.Rx.Point_bSecondary(rxlocs, 'z', 'real')
    # bzi = FDEM.Rx.Point_bSecondary(rxlocs, 'z', 'imag')

    freqs = np.linspace(2000, 10000, 10)  #np.logspace(3, 4, 10)
    srcLoc = np.array([0., 0., 30.])

    print(
        'min skin depth = ', 500./np.sqrt(freqs.max() * sig_half),
        'max skin depth = ', 500./np.sqrt(freqs.min() * sig_half)
    )
    print(
        'max x ', mesh.vectorCCx.max(), 'min z ', mesh.vectorCCz.min(),
        'max z ', mesh.vectorCCz.max()
    )

    srcList = [
        FDEM.Src.MagDipole([bzr], freq, srcLoc, orientation='Z')
        for freq in freqs
    ]

    surveyFD = FDEM.Survey(srcList)
    prbFD = FDEM.Problem3D_b(
        mesh, sigma=surj1Dmap * sigma, muMap=muMap, Solver=Solver
    )
    prbFD.pair(surveyFD)
    std = 0.03
    surveyFD.makeSyntheticData(mtrue, std)
    surveyFD.eps = np.linalg.norm(surveyFD.dtrue)*1e-6

    # FDEM inversion
    np.random.seed(13472)
    dmisfit = DataMisfit.l2_DataMisfit(surveyFD)
    regMesh = Mesh.TensorMesh([mesh.hz[muMap.maps[-1].indActive]])
    reg = Regularization.Simple(regMesh)
    opt = Optimization.InexactGaussNewton(maxIterCG=10)
    invProb = InvProblem.BaseInvProblem(dmisfit, reg, opt)

    # Inversion Directives    betaest = Directives.BetaEstimate_ByEig(beta0_ratio=2.)

    beta = Directives.BetaSchedule(coolingFactor=4, coolingRate=3)
    betaest = Directives.BetaEstimate_ByEig(beta0_ratio=2.)
    target = Directives.TargetMisfit()
    directiveList = [beta, betaest, target]

    inv = Inversion.BaseInversion(invProb, directiveList=directiveList)
    m0 = mur_half * np.ones(mtrue.size)
    reg.alpha_s = 2e-2
    reg.alpha_x = 1.
    prbFD.counter = opt.counter = Utils.Counter()
    opt.remember('xc')
    moptFD = inv.run(m0)

    dpredFD = surveyFD.dpred(moptFD)

    if plotIt:
        fig, ax = plt.subplots(1, 3, figsize=(10, 6))

        fs = 13  # fontsize
        matplotlib.rcParams['font.size'] = fs

        # Plot the conductivity model
        ax[0].semilogx(sigma[active], mesh.vectorCCz[active], 'k-', lw=2)
        ax[0].set_ylim(-500, 0)
        ax[0].set_xlim(5e-3, 1e-1)

        ax[0].set_xlabel('Conductivity (S/m)', fontsize=fs)
        ax[0].set_ylabel('Depth (m)', fontsize=fs)
        ax[0].grid(
            which='both', color='k', alpha=0.5, linestyle='-', linewidth=0.2
        )
        ax[0].legend(['Conductivity Model'], fontsize=fs, loc=4)

        # Plot the permeability model
        ax[1].plot(mur[active], mesh.vectorCCz[active], 'k-', lw=2)
        ax[1].plot(moptFD, mesh.vectorCCz[active], 'b-', lw=2)
        ax[1].set_ylim(-500, 0)
        ax[1].set_xlim(0.5, 2.1)

        ax[1].set_xlabel('Relative Permeability', fontsize=fs)
        ax[1].set_ylabel('Depth (m)', fontsize=fs)
        ax[1].grid(
            which='both', color='k', alpha=0.5, linestyle='-', linewidth=0.2
        )
        ax[1].legend(['True', 'Predicted'], fontsize=fs, loc=4)

        # plot the data misfits - negative b/c we choose positive to be in the
        # direction of primary

        ax[2].plot(freqs, -surveyFD.dobs, 'k-', lw=2)
        # ax[2].plot(freqs, -surveyFD.dobs[1::2], 'k--', lw=2)

        ax[2].loglog(freqs, -dpredFD, 'bo', ms=6)
        # ax[2].loglog(freqs, -dpredFD[1::2], 'b+', markeredgewidth=2., ms=10)


        # Labels, gridlines, etc
        ax[2].grid(which='both', alpha=0.5, linestyle='-', linewidth=0.2)
        ax[2].grid(which='both', alpha=0.5, linestyle='-', linewidth=0.2)

        ax[2].set_xlabel('Frequency (Hz)', fontsize=fs)
        ax[2].set_ylabel('Vertical magnetic field (-T)', fontsize=fs)

        # ax[2].legend(("Obs", "Pred"), fontsize=fs)
        ax[2].legend(
            ("z-Obs (real)", "z-Pred (real)"),
            fontsize=fs
        )
        ax[2].set_xlim(freqs.max(), freqs.min())

        ax[0].set_title("(a) Conductivity Model", fontsize=fs)
        ax[1].set_title("(b) $\mu_r$ Model", fontsize=fs)
        ax[2].set_title("(c) FDEM observed vs. predicted", fontsize=fs)
        # ax[2].set_title("(c) TDEM observed vs. predicted", fontsize=fs)

        plt.tight_layout(pad=1.5)

if __name__ == '__main__':
    run(plotIt=True)
    plt.show()
