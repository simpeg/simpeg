import numpy as np
from SimPEG import (
    Mesh, Maps, DataMisfit, Regularization,
    Optimization, InvProblem, Inversion, Directives, Utils
)
import SimPEG.EM as EM
import matplotlib.pyplot as plt
import seaborn
from pymatsolver import PardisoSolver
from sklearn.mixture import GaussianMixture

seaborn.set()

np.random.seed(12345)

cs, ncx, ncz, npad = 5., 25, 24, 15
hx = [(cs, ncx),  (cs, npad, 1.3)]
hz = [(cs, npad, -1.3), (cs, ncz), (cs, npad, 1.3)]
mesh = Mesh.CylMesh([hx, 1, hz], '00C')

active = mesh.vectorCCz < 0.
layer = (mesh.vectorCCz < 0.) & (mesh.vectorCCz >= -100.)
Rlayer = (mesh.vectorCCz < -100.) & (mesh.vectorCCz >= -121.)
actMap = Maps.InjectActiveCells(mesh, active, np.log(1e-8), nC=mesh.nCz)
mapping = Maps.ExpMap(mesh) * Maps.SurjectVertical1D(mesh) * actMap
sig_half = 2e-3
sig_air = 1e-8
sig_layer = 1e-3
sig_Rlayer = 1e-4
sigma = np.ones(mesh.nCz)*sig_air
sigma[active] = sig_half
sigma[layer] = sig_layer
#sigma[Rlayer] = sig_Rlayer
mtrue = np.log(sigma[active])

rxOffset = 1e-3
rx = EM.TDEM.Rx.Point_b(
np.array([[rxOffset, 0., 30]]),
np.logspace(-5, -3, 31),
'z'
)
src = EM.TDEM.Src.MagDipole([rx], loc=np.array([0., 0., 80]))
survey = EM.TDEM.Survey([src])
prb = EM.TDEM.Problem3D_b(mesh, sigmaMap=mapping)

prb.Solver = PardisoSolver
prb.timeSteps = [(1e-06, 20), (1e-05, 20), (0.0001, 20)]
prb.pair(survey)

# create observed data
std = 0.01

survey.dobs = survey.makeSyntheticData(mtrue, std)
survey.std = std
survey.eps = 1e-5*np.linalg.norm(survey.dobs)

dmisfit = DataMisfit.l2_DataMisfit(survey)
regMesh = Mesh.TensorMesh([mesh.hz[mapping.maps[-1].indActive]])
reg = Regularization.Tikhonov(regMesh, alpha_s=1e-2, alpha_x=1.)
opt = Optimization.InexactGaussNewton(maxIter=10, LSshorten=0.5)
invProb = InvProblem.BaseInvProblem(dmisfit, reg, opt)

# Tikhonov Inversion
####################

# Create an inversion object
beta = Directives.BetaSchedule(coolingFactor=5, coolingRate=2)
betaest = Directives.BetaEstimate_ByEig(beta0_ratio=1e0)
target = Directives.TargetMisfit()

inv = Inversion.BaseInversion(invProb, directiveList=[beta, betaest,target])
m0 = np.log(np.ones(mtrue.size)*sig_half)
prb.counter = opt.counter = Utils.Counter()
opt.remember('xc')

mnormal = inv.run(m0)

# Petro Inversion
#################

clf = GaussianMixture(n_components=2,
                      covariance_type='full',
                      max_iter=1000, n_init=20, reg_covar=1e-3)
clf.fit(mtrue.reshape(-1, 1))
Utils.order_clusters_GM_weight(clf)

dmisfit = DataMisfit.l2_DataMisfit(survey)
regMesh = Mesh.TensorMesh([mesh.hz[mapping.maps[-1].indActive]])
reg = Regularization.SimplePetroRegularization(GMmref=clf,
                                               mesh=regMesh,
                                               mref=m0)
reg.mrefInSmooth = False
reg.alpha_s = 1e-4
reg.alpha_x = 1.
reg.alpha_z = 1.
opt = Optimization.InexactGaussNewton(maxIter=10, LSshorten=0.5)
invProb = InvProblem.BaseInvProblem(dmisfit, reg, opt)

gamma = np.ones(clf.n_components)*0.75
invProb.reg.gamma = gamma

# Create an inversion object
beta = Directives.PetroBetaReWeighting(rateCooling = 2., rateWarming = 2.)
betaest = Directives.BetaEstimate_ByEig(beta0_ratio=1e0, ninit=10)
#invProb.beta = 1.
target = Directives.PetroTargetMisfit(verbose=True)
petrodir = Directives.GaussianMixtureUpdateModel(verbose=False)
addmref = Directives.AddMrefInSmooth()
inv = Inversion.BaseInversion(
    invProb,
    directiveList=[betaest, target, beta, addmref, petrodir])
m0 = np.log(np.ones(mtrue.size)*sig_half)
prb.counter = opt.counter = Utils.Counter()
opt.remember('xc')

mopt = inv.run(m0)


# Plot it
#########
fig, ax = plt.subplots(1, 3, figsize=(18, 6))
ax[0].loglog(rx.times, survey.dtrue, 'b.-')
#ax[0].loglog(rx.times, survey.dobs, 'r.-')
ax[0].loglog(rx.times, survey.dpred(mopt), 'r.-')
ax[0].legend(('Noisefree', '$d^{pred}$'), fontsize=16)
ax[0].set_xlabel('Time (s)', fontsize=14)
ax[0].set_ylabel('$B_z$ (T)', fontsize=16)
ax[0].set_xlabel('Time (s)', fontsize=14)
ax[0].grid(color='k', alpha=0.5, linestyle='dashed', linewidth=0.5)

ax[1].semilogx(sigma[active], mesh.vectorCCz[active])
ax[1].semilogx(np.exp(mnormal), mesh.vectorCCz[active])

ax[2].semilogx(sigma[active], mesh.vectorCCz[active])
ax[2].semilogx(np.exp(mopt), mesh.vectorCCz[active])
ax[2].semilogx(np.exp(invProb.reg.mref), mesh.vectorCCz[active])

ax[1].set_ylim(-600, 0)
ax[1].set_xlim(1e-4, 1e-2)
ax[1].set_xlabel('Conductivity (S/m)', fontsize=14)
ax[1].set_ylabel('Depth (m)', fontsize=14)
ax[1].grid(color='k', alpha=0.5, linestyle='dashed', linewidth=0.5)
ax[1].legend(['$\sigma_{true}$', '$\sigma_{pred}$'])
ax[1].set_title('Tikhonov', fontsize=15)

ax[2].set_ylim(-600, 0)
ax[2].set_xlim(1e-4, 1e-2)
ax[2].set_xlabel('Conductivity (S/m)', fontsize=14)
ax[2].set_ylabel('Depth (m)', fontsize=14)
ax[2].grid(color='k', alpha=0.5, linestyle='dashed', linewidth=0.5)
ax[2].legend(['$\sigma_{true}$', '$\sigma_{pred}$', 'learned mref'])
ax[2].set_title('Petrophysically\nconstrained', fontsize=15)

plt.show()
