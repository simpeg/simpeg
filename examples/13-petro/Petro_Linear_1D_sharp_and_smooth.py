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
    Mesh, Problem, Survey, Maps, Utils, EM, DataMisfit,
    Regularization, Optimization, InvProblem,
    Directives, Inversion)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

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
        np.exp(p*jk[k]*mesh.vectorCCx) *
        np.cos(np.pi*q*jk[k]*mesh.vectorCCx)
        )

G = np.empty((nk, mesh.nC))

for i in range(nk):
    G[i, :] = g(i)

# True model
mtrue = np.zeros(mesh.nC)
mtrue[mesh.vectorCCx > 0.2] = 1.
mtrue[mesh.vectorCCx > 0.35] = 0.
t = (mesh.vectorCCx-0.65)/0.25
indx = np.abs(t)<1
mtrue[indx] = -(((1-t**2.)**2.)[indx])

# SimPEG problem and survey
prob = Problem.LinearProblem(mesh, G=G)
survey = Survey.LinearSurvey()
survey.pair(prob)
survey.makeSyntheticData(mtrue, std=0.01)

M = prob.mesh

# Setup the inverse problem
reg = Regularization.Tikhonov(mesh, alpha_s=1., alpha_x=1.)
dmis = DataMisfit.l2_DataMisfit(survey)
opt = Optimization.InexactGaussNewton(maxIter=60)
invProb = InvProblem.BaseInvProblem(dmis, reg, opt)
directives = [
    Directives.BetaEstimate_ByEig(beta0_ratio=1e-3),
    Directives.TargetMisfit(),
    ]

inv = Inversion.BaseInversion(invProb, directiveList=directives)
m0 = np.zeros_like(survey.mtrue)

mnormal = inv.run(m0)

#########################################
# Petrophysically constrained inversion #
#########################################

# fit a Gaussian Mixture Model with n components
# on the true model to simulate the laboratory
# petrophysical measurements
n = 3
clf = GaussianMixture(n_components=n, covariance_type='full', max_iter=1000,
                      n_init=20, reg_covar=10**(-2.5))
clf.fit(mtrue.reshape(-1, 1))

# Initial model, same as for Tikhonov
minit = m0

# Petrophyically constrained regularization
reg = Regularization.PetroRegularization(GMmref=clf, GMmodel=clf, mesh=mesh, mref=m0,
                                         alpha_s=1.,alpha_x=1e-3)

# Include the reference model in the smoothness term
reg.mrefInSmooth = False

# For some reason we need to reinitialize the optimization
opt = Optimization.InexactGaussNewton(maxIter=60, tolX=1e-6)
opt.remember('xc')

# Setup new inverse problem
invProb = InvProblem.BaseInvProblem(dmis, reg, opt)

# Directives
#beta = Directives.BetaEstimate_ByEig(beta0_ratio=1e-3)
invProb.beta = 10.
gamma_petro = np.ones(clf.n_components)*.5
#petrodir = Directives.GaussianMixtureUpdateModel()
K = .05*len(mtrue)
petrodir = Directives.FuzzyGaussianMixtureWithPriorUpdateModel(
    alphadir = 1.+clf.weights_*K,
    kappa = K,
    nu = K)

invProb.reg.gamma = gamma_petro

smoothref = Directives.SmoothUpdateReferenceModel()

# Setup Inversion
inv = Inversion.BaseInversion(invProb, directiveList=[petrodir,]) #smoothref

mcluster = inv.run(minit)

# Final Plot
fig, axes = plt.subplots(1, 3, figsize=(12*1.2, 4*1.2))
for i in range(prob.G.shape[0]):
    axes[0].plot(prob.G[i, :])
axes[0].set_title('Columns of matrix G')

axes[1].hist(mtrue, bins = 10, linewidth =3.);
axes[1].set_xlabel('Model value')
axes[1].set_xlabel('Occurence')
axes[1].hist(invProb.model, bins = 10);
axes[1].legend(['Mtrue Hist.', 'Model Hist.'])

axes[2].plot(M.vectorCCx, survey.mtrue, color='black')
axes[2].plot(M.vectorCCx, mnormal, color='blue')
axes[2].plot(M.vectorCCx, mcluster, 'r-')
axes[2].plot(M.vectorCCx,invProb.reg.mref, 'r--')

axes[2].legend(('True Model', 'L2 Model', 'Petro Model','Learned Mref'))
axes[2].set_ylim([-2, 2])

plt.show()
