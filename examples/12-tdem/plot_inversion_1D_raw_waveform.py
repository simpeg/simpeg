"""
EM: TDEM: 1D: Inversion with VTEM waveform
==========================================

Here we will create and run a TDEM 1D inversion,
with VTEM waveform of which initial condition
is zero, but have some on- and off-time.
"""
import numpy as np
from SimPEG import (
    Mesh, Maps, SolverLU, DataMisfit, Regularization,
    Optimization, InvProblem, Inversion, Directives, Utils
)
import SimPEG.EM as EM
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
try:
    from pymatsolver import Pardiso as Solver
except ImportError:
    from SimPEG import SolverLU as Solver

def run(plotIt=True):

    cs, ncx, ncz, npad = 5., 25, 24, 15
    hx = [(cs, ncx),  (cs, npad, 1.3)]
    hz = [(cs, npad, -1.3), (cs, ncz), (cs, npad, 1.3)]
    mesh = Mesh.CylMesh([hx, 1, hz], '00C')

    active = mesh.vectorCCz < 0.
    layer = (mesh.vectorCCz < -50.) & (mesh.vectorCCz >= -150.)
    actMap = Maps.InjectActiveCells(mesh, active, np.log(1e-8), nC=mesh.nCz)
    mapping = Maps.ExpMap(mesh) * Maps.SurjectVertical1D(mesh) * actMap
    sig_half = 1e-3
    sig_air = 1e-8
    sig_layer = 1e-2
    sigma = np.ones(mesh.nCz)*sig_air
    sigma[active] = sig_half
    sigma[layer] = sig_layer
    mtrue = np.log(sigma[active])

    x = np.r_[30, 50, 70, 90]
    rxloc = np.c_[x, x*0., np.zeros_like(x)]

    prb = EM.TDEM.Problem3D_b(mesh, sigmaMap=mapping)
    prb.Solver = Solver
    prb.timeSteps = [(1e-3, 5), (1e-4, 5), (5e-5, 10), (5e-5, 5), (1e-4, 10), (5e-4, 10)]

    # Use VTEM waveform
    out = EM.Utils.VTEMFun(prb.times, 0.00595, 0.006, 100)

    # Forming function handle for waveform using 1D linear interpolation
    wavefun = interp1d(prb.times, out)
    t0 = 0.006
    waveform = EM.TDEM.Src.RawWaveform(offTime=t0, waveFct=wavefun)

    rx = EM.TDEM.Rx.Point_dbdt(rxloc, np.logspace(-4, -2.5, 11)+t0, 'z')
    src = EM.TDEM.Src.CircularLoop([rx], waveform=waveform,
                                   loc=np.array([0., 0., 0.]), radius=10.)
    survey = EM.TDEM.Survey([src])
    prb.pair(survey)
    # create observed data
    std = 0.02

    survey.dobs = survey.makeSyntheticData(mtrue, std)
    # dobs = survey.dpred(mtrue)
    survey.std = std
    survey.eps = 1e-11

    dmisfit = DataMisfit.l2_DataMisfit(survey)
    regMesh = Mesh.TensorMesh([mesh.hz[mapping.maps[-1].indActive]])
    reg = Regularization.Simple(regMesh)
    opt = Optimization.InexactGaussNewton(maxIter=5, LSshorten=0.5)
    invProb = InvProblem.BaseInvProblem(dmisfit, reg, opt)
    target = Directives.TargetMisfit()
    # Create an inversion object
    beta = Directives.BetaSchedule(coolingFactor=1., coolingRate=2.)
    betaest = Directives.BetaEstimate_ByEig(beta0_ratio=1e0)
    invProb.beta = 1e2
    inv = Inversion.BaseInversion(invProb, directiveList=[beta, target])
    m0 = np.log(np.ones(mtrue.size)*sig_half)
    prb.counter = opt.counter = Utils.Counter()
    opt.remember('xc')
    mopt = inv.run(m0)

    if plotIt:
        fig, ax = plt.subplots(1, 2, figsize=(10, 6))
        Dobs = survey.dobs.reshape((len(rx.times), len(x)))
        Dpred = invProb.dpred.reshape((len(rx.times), len(x)))
        for i in range (len(x)):
            ax[0].loglog(rx.times-t0, -Dobs[:,i].flatten(), 'k')
            ax[0].loglog(rx.times-t0, -Dpred[:,i].flatten(), 'k.')
            if i==0:
                ax[0].legend(('$d^{obs}$', '$d^{pred}$'), fontsize=16)
        ax[0].set_xlabel('Time (s)', fontsize=14)
        ax[0].set_ylabel('$db_z / dt$ (nT/s)', fontsize=16)
        ax[0].set_xlabel('Time (s)', fontsize=14)
        ax[0].grid(color='k', alpha=0.5, linestyle='dashed', linewidth=0.5)

        plt.semilogx(sigma[active], mesh.vectorCCz[active])
        plt.semilogx(np.exp(mopt), mesh.vectorCCz[active])
        ax[1].set_ylim(-600, 0)
        ax[1].set_xlim(1e-4, 1e-1)
        ax[1].set_xlabel('Conductivity (S/m)', fontsize=14)
        ax[1].set_ylabel('Depth (m)', fontsize=14)
        ax[1].grid(color='k', alpha=0.5, linestyle='dashed', linewidth=0.5)
        plt.legend(['$\sigma_{true}$', '$\sigma_{pred}$'])

if __name__ == '__main__':
    run()
    plt.show()
