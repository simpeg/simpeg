"""
Petrophysically constrained inversion: Linear example
=====================================================

We do a comparison between the classic Tikhonov inversion
and our formulation of a petrophysically constrained inversion.
We explore it through the UBC linear example.

"""

#####################
# Tikhonov Inversion#
#####################

from SimPEG import (
    Mesh, Problem, Survey, DataMisfit,
    Regularization, Optimization, InvProblem,
    Directives, Inversion, Utils)
import numpy as np
import matplotlib.pyplot as plt

# Better rendering
import seaborn
seaborn.set()

# Random seed for reproductibility
np.random.seed(1)
# Mesh
N = 100
mesh = Mesh.TensorMesh([N])

# Survey design parameters
nk = 20
jk = np.linspace(1., 60., nk)
p = -0.25
q = 0.25


# Physics
def g(k):
    return (
        np.exp(p * jk[k] * mesh.vectorCCx) *
        np.cos(np.pi * q * jk[k] * mesh.vectorCCx)
    )


G = np.empty((nk, mesh.nC))

for i in range(nk):
    G[i, :] = g(i)

# True model
mtrue = np.zeros(mesh.nC)
mtrue[mesh.vectorCCx > 0.3] = 1.
mtrue[mesh.vectorCCx > 0.45] = -0.5
mtrue[mesh.vectorCCx > 0.6] = 0

# SimPEG problem and survey
prob = Problem.LinearProblem(mesh, G=G)
survey = Survey.LinearSurvey()
survey.pair(prob)
survey.makeSyntheticData(mtrue, std=0.01)

M = prob.mesh

# Setup the inverse problem
reg = Regularization.Tikhonov(mesh, alpha_s=1., alpha_x=1.)
dmis = DataMisfit.l2_DataMisfit(survey)
opt = Optimization.ProjectedGNCG(
    maxIter=60, tolX=1e-6, maxIterCG=100, tolCG=1e-3
)
invProb = InvProblem.BaseInvProblem(dmis, reg, opt)
directives = [
    Directives.BetaEstimate_ByEig(beta0_ratio=1e-5),
    Directives.TargetMisfit(),
]

inv = Inversion.BaseInversion(invProb, directiveList=directives)
m0 = np.zeros_like(survey.mtrue)

mnormal = inv.run(m0)

#########################################
# Petrophysically constrained inversion #
#########################################
from SimPEG import Maps
a = 1.
linearmap = Maps.LinearMap(a=a, b=0.)
mpetro = linearmap * mtrue
# fit a Gaussian Mixture Model with n components
# on the true model to simulate the laboratory
# petrophysical measurements
n = 3
clf = Utils.GaussianMixture(n_components=n, covariance_type='spherical', max_iter=1000,
                      n_init=20, reg_covar=(a**2) * 1e-3, warm_start=True)
clf.fit(mpetro.reshape(-1, 1))
# Initial model, same as for Tikhonov
minit = m0

# Petrophyically constrained regularization
#reg = Regularization.SimplePetroRegularization(
#    GMmref=clf, GMmodel=clf, mesh=mesh, mref=m0, maplist=[linearmap])
reg = Utils.MakeSimplePetroRegularization(
    GMmref=clf, GMmodel=clf, mesh=mesh, mref=m0, maplist=[linearmap])

# Include the reference model in the smoothness term
reg.mrefInSmooth = False

# For some reason we need to reinitialize the optimization
opt = Optimization.ProjectedGNCG(
    maxIter=60, tolX=1e-6, maxIterCG=100, tolCG=1e-3)
opt.remember('xc')

# Setup new inverse problem
invProb = InvProblem.BaseInvProblem(dmis, reg, opt)

# Directives
Alphas = Directives.AlphasSmoothEstimate_ByEig(alpha0_ratio=1e-2, ninit=10)
beta = Directives.BetaEstimate_ByEig(beta0_ratio=1e-5, ninit=100)
betaIt = Directives.PetroBetaReWeighting(
    verbose=True, rateCooling=5., rateWarming=1.,
    tolerance=0.02, UpdateRate=1,
    ratio_in_cooling=False,
    progress=0.1,
    update_prior_confidence=True,
    ratio_in_gamma_cooling=False,
    alphadir_rateCooling=1.,
    kappa_rateCooling=2.,
    nu_rateCooling=1.,)
targets = Directives.PetroTargetMisfit(
    TriggerSmall=True, TriggerTheta=False, verbose=True)
betaIt.gamma_min = 0.3
targets.CLtarget = .1
gamma_petro = np.ones(clf.n_components) * 1.
#membership = np.zeros_like(mtrue, dtype='int')
membership = clf.predict(mtrue[:, np.newaxis])
petrodir = Directives.GaussianMixtureUpdateModel(
    verbose=False)  # , fixed_membership=membership)
addmref = Directives.AddMrefInSmooth(verbose=True, wait_till_stable=True)
invProb.reg.gamma = gamma_petro

# Setup Inversion
inv = Inversion.BaseInversion(invProb, directiveList=[Alphas, beta,
                                                      petrodir, targets,
                                                      betaIt, addmref])

mcluster = inv.run(minit)

print('All stopping Criteria: ', targets.AllStop)
print('Final Data Misfit: ', dmis(mcluster))
print('Final Cluster Scorce: ',
      invProb.reg.objfcts[0].objfcts[0](mcluster, externalW=False))
#print('Final DP misfit: ', targets.ThetaTarget())
print('alphas', reg.multipliers)
print('final confidences', inv.directiveList.dList[
      2].alphadir, petrodir.kappa, petrodir.nu)
print(betaIt.mode2_iter)
print(reg.objfcts[1](mcluster))
reg.mrefInSmooth = False
print(reg.objfcts[1].mrefInSmooth)
print(reg.objfcts[1]._delta_m(mcluster))

print(reg.objfcts[0].mref == reg.objfcts[1].objfcts[1].mref)

# Final Plot
fig, axes = plt.subplots(1, 3, figsize=(12 * 1.2, 4 * 1.2))
for i in range(prob.G.shape[0]):
    axes[0].plot(prob.G[i, :])
axes[0].set_title('Columns of matrix G')

axes[1].hist(mtrue, bins=10, linewidth=3.)
axes[1].set_xlabel('Model value')
axes[1].set_xlabel('Occurence')
axes[1].hist(invProb.model, bins=10)
axes[1].legend(['Mtrue Hist.', 'Model Hist.'])

axes[2].plot(M.vectorCCx, survey.mtrue, color='black')
axes[2].plot(M.vectorCCx, mnormal, color='blue')
axes[2].plot(M.vectorCCx, mcluster, 'r-')
axes[2].plot(M.vectorCCx, linearmap.inverse(reg.objfcts[0].mref), 'r--')

axes[2].legend(('True Model', 'L2 Model', 'Petro Model', 'Learned Mref'))
axes[2].set_ylim([-2, 2])

plt.show()
