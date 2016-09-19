import numpy as np
from SimPEG import (Mesh, Maps, Utils, SolverLU, DataMisfit, Regularization,
                    Optimization, Inversion, InvProblem, Directives)
import SimPEG.EM as EM
from SimPEG.EM import mu_0


def run(plotIt=True):
    """
        EM: FDEM: 1D: Inversion
        =======================

        Here we will create and run a FDEM 1D inversion.

    """

    cs, ncx, ncz, npad = 5., 25, 15, 15
    hx = [(cs, ncx), (cs, npad, 1.3)]
    hz = [(cs, npad, -1.3), (cs, ncz), (cs, npad, 1.3)]
    mesh = Mesh.CylMesh([hx, 1, hz], '00C')

    layerz = -100.

    active = mesh.vectorCCz < 0.
    layer = (mesh.vectorCCz < 0.) & (mesh.vectorCCz >= layerz)
    actMap = Maps.InjectActiveCells(mesh, active, np.log(1e-8), nC=mesh.nCz)
    mapping = Maps.ExpMap(mesh) * Maps.SurjectVertical1D(mesh) * actMap
    sig_half = 2e-2
    sig_air = 1e-8
    sig_layer = 1e-2
    sigma = np.ones(mesh.nCz)*sig_air
    sigma[active] = sig_half
    sigma[layer] = sig_layer
    mtrue = np.log(sigma[active])

    if plotIt:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(3, 6))
        plt.semilogx(sigma[active], mesh.vectorCCz[active])
        ax.set_ylim(-500, 0)
        ax.set_xlim(1e-3, 1e-1)
        ax.set_xlabel('Conductivity (S/m)', fontsize=14)
        ax.set_ylabel('Depth (m)', fontsize=14)
        ax.grid(color='k', alpha=0.5, linestyle='dashed', linewidth=0.5)

    rxOffset = 10.
    bzi = EM.FDEM.Rx.Point_b(np.array([[rxOffset, 0., 1e-3]]), orientation='z',
                             component='imag')

    freqs = np.logspace(1, 3, 10)
    srcLoc = np.array([0., 0., 10.])

    srcList = [EM.FDEM.Src.MagDipole([bzi], freq, srcLoc, orientation='Z')
               for freq in freqs]

    survey = EM.FDEM.Survey(srcList)
    prb = EM.FDEM.Problem3D_b(mesh, mapping=mapping)

    try:
        from pymatsolver import PardisoSolver
        prb.Solver = PardisoSolver
    except ImportError:
        prb.Solver = SolverLU

    prb.pair(survey)

    std = 0.05
    survey.makeSyntheticData(mtrue, std)

    survey.std = std
    survey.eps = np.linalg.norm(survey.dtrue)*1e-5

    if plotIt:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize = (6, 6))
        ax.semilogx(freqs, survey.dtrue[:freqs.size], 'b.-')
        ax.semilogx(freqs, survey.dobs[:freqs.size], 'r.-')
        ax.legend(('Noisefree', '$d^{obs}$'), fontsize = 16)
        ax.set_xlabel('Time (s)', fontsize=14)
        ax.set_ylabel('$B_z$ (T)', fontsize=16)
        ax.set_xlabel('Time (s)', fontsize=14)
        ax.grid(color='k', alpha=0.5, linestyle='dashed', linewidth=0.5)

    dmisfit = DataMisfit.l2_DataMisfit(survey)
    regMesh = Mesh.TensorMesh([mesh.hz[mapping.maps[-1].indActive]])
    reg = Regularization.Tikhonov(regMesh)
    opt = Optimization.InexactGaussNewton(maxIter=6)
    invProb = InvProblem.BaseInvProblem(dmisfit, reg, opt)

    # Create an inversion object
    beta = Directives.BetaSchedule(coolingFactor=5, coolingRate=2)
    betaest = Directives.BetaEstimate_ByEig(beta0_ratio=1e0)
    inv = Inversion.BaseInversion(invProb, directiveList=[beta, betaest])
    m0 = np.log(np.ones(mtrue.size)*sig_half)
    reg.alpha_s = 1e-3
    reg.alpha_x = 1.
    prb.counter = opt.counter = Utils.Counter()
    opt.LSshorten = 0.5
    opt.remember('xc')

    mopt = inv.run(m0)

    if plotIt:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(3, 6))
        plt.semilogx(sigma[active], mesh.vectorCCz[active])
        plt.semilogx(np.exp(mopt), mesh.vectorCCz[active])
        ax.set_ylim(-500, 0)
        ax.set_xlim(1e-3, 1e-1)
        ax.set_xlabel('Conductivity (S/m)', fontsize=14)
        ax.set_ylabel('Depth (m)', fontsize=14)
        ax.grid(color='k', alpha=0.5, linestyle='dashed', linewidth=0.5)
        plt.legend(['$\sigma_{true}$', '$\sigma_{pred}$'], loc='best')
        plt.show()


if __name__ == '__main__':
    run()
