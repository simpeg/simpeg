from SimPEG import *
import simpegEM as EM
from scipy.constants import mu_0
import matplotlib.pyplot as plt

plotIt = False

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
    fig, ax = plt.subplots(1,1, figsize = (3, 6))
    plt.semilogx(sigma[active], mesh.vectorCCz[active])
    ax.set_ylim(-600, 0)
    ax.set_xlim(1e-4, 1e-2)
    ax.set_xlabel('Conductivity (S/m)', fontsize = 14)
    ax.set_ylabel('Depth (m)', fontsize = 14)
    ax.grid(color='k', alpha=0.5, linestyle='dashed', linewidth=0.5)
    plt.show()


rxOffset=1e-3
rx = EM.TDEM.RxTDEM(np.array([[rxOffset, 0., 30]]), np.logspace(-5,-3, 31), 'bz')
tx = EM.TDEM.TxTDEM(np.array([0., 0., 80]), 'VMD_MVP', [rx])
survey = EM.TDEM.SurveyTDEM([tx])
prb = EM.TDEM.ProblemTDEM_b(mesh, mapping=mapping)

prb.Solver = SolverLU
prb.timeSteps = [(1e-06, 20),(1e-05, 20), (0.0001, 20)]
prb.pair(survey)
dtrue = survey.dpred(mtrue)


survey.dtrue = dtrue
std = 0.05
noise = std*abs(survey.dtrue)*np.random.randn(*survey.dtrue.shape)
survey.dobs = survey.dtrue+noise
survey.std = survey.dobs*0 + std
survey.Wd = 1/(abs(survey.dobs)*std)

if plotIt:
    fig, ax = plt.subplots(1,1, figsize = (10, 6))
    ax.loglog(rx.times, dtrue, 'b.-')
    ax.loglog(rx.times, survey.dobs, 'r.-')
    ax.legend(('Noisefree', '$d^{obs}$'), fontsize = 16)
    ax.set_xlabel('Time (s)', fontsize = 14)
    ax.set_ylabel('$B_z$ (T)', fontsize = 16)
    ax.set_xlabel('Time (s)', fontsize = 14)
    ax.grid(color='k', alpha=0.5, linestyle='dashed', linewidth=0.5)
    plt.show()


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
