from SimPEG import (
    Mesh, Problem, Survey, Maps, Utils, EM, DataMisfit,
    Regularization, Optimization, InvProblem,
    Directives, Inversion)
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
nk = 30
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

m0 = np.zeros(mesh.nC)
m0[20:41] = np.linspace(0., 1., 21)
m0[41:57] = np.linspace(-1, 0., 16)

poly0 = Maps.PolynomialPetroClusterMap(coeffyx=np.r_[0., -2., 2.])
poly1 = Maps.PolynomialPetroClusterMap(coeffyx=np.r_[-0., -2., -2.])
poly0_inverse = Maps.PolynomialPetroClusterMap(coeffyx=-np.r_[-0., -2., 2.])
poly1_inverse = Maps.PolynomialPetroClusterMap(coeffyx=-np.r_[0., -2., -2.])
cluster_mapping = [Maps.IdentityMap(), poly0_inverse, poly1_inverse]

m1 = np.zeros(100)
m1[20:41] = 1. + (poly0 * np.vstack([m0[20:41], m1[20:41]]).T)[:, 1]
m1[41:57] = -1. + (poly1 * np.vstack([m0[41:57], m1[41:57]]).T)[:, 1]

model2d = np.vstack([m0, m1]).T
m = Utils.mkvc(model2d)

clfmapping = Utils.GaussianMixtureWithMapping(
    n_components=3, covariance_type='full', tol=1e-3,
    reg_covar=1e-3, max_iter=100, n_init=10, init_params='kmeans',
    random_state=None, warm_start=False,
    verbose=0, verbose_interval=10, cluster_mapping=cluster_mapping
)
clfmapping = clfmapping.fit(model2d)

wires = Maps.Wires(('m1', mesh.nC), ('m2', mesh.nC))
prob1 = Problem.LinearProblem(mesh, G=G, modelMap=wires.m1)
survey1 = Survey.LinearSurvey()
survey1.pair(prob1)
survey1.makeSyntheticData(m, std=0.01)
survey1.eps=0.

prob2 = Problem.LinearProblem(mesh, G=G, modelMap=wires.m2)
survey2 = Survey.LinearSurvey()
survey2.pair(prob2)
survey2.makeSyntheticData(m, std=0.01)
survey2.eps=0.

dmis1 = DataMisfit.l2_DataMisfit(survey1)
dmis2 = DataMisfit.l2_DataMisfit(survey2)
dmis = dmis1 + dmis2

minit = np.zeros_like(m)
# Distance weighting
wr1 = np.sum(prob1.getJ(minit)**2., axis=0)**0.5
wr1 = wr1 / np.max(wr1)

# Distance weighting
wr2 = np.sum(prob2.getJ(minit)**2., axis=0)**0.5
wr2 = wr2 / np.max(wr2)

wr = wr1 + wr2
W = Utils.sdiag(wr)

reg_simple = Regularization.SimplePetroWithMappingRegularization(
    mesh=mesh,
    #mref = np.zeros_like(m),
    GMmref=clfmapping,
    GMmodel=clfmapping,
    approx_gradient=True, alpha_x=0.,
    wiresmap=wires,
    evaltype='approx')
reg_simple.objfcts[0].cell_weights = wr

opt = Optimization.ProjectedGNCG(
    maxIter=20, tolX=1e-6, maxIterCG=100, tolCG=1e-3
)

invProb = InvProblem.BaseInvProblem(dmis, reg_simple, opt)

# Directives
Alphas = Directives.AlphasSmoothEstimate_ByEig(
    alpha0_ratio=5e-2, ninit=10, verbose=True)
Scales = Directives.ScalingEstimate_ByEig(
    Chi0_ratio=.1, verbose=True, ninit=100)
beta = Directives.BetaEstimate_ByEig(beta0_ratio=1e-5, ninit=100)
betaIt = Directives.PetroBetaReWeighting(
    verbose=True, rateCooling=5., rateWarming=1.,
    tolerance=0.02, UpdateRate=1,
    ratio_in_cooling=False,
    progress=0.1,
    update_prior_confidence=False,
    ratio_in_gamma_cooling=False,
    alphadir_rateCooling=1.,
    kappa_rateCooling=1.,
    nu_rateCooling=1.,)
targets = Directives.PetroTargetMisfit(chiSmall=1.,
                                       TriggerSmall=True, TriggerTheta=False, verbose=True)
gamma_petro = np.ones(clfmapping.n_components) * 1e8
#membership = np.zeros_like(mtrue, dtype='int')
petrodir = Directives.UpdateReference()
invProb.reg.gamma = gamma_petro
#addmref = Directives.AddMrefInSmooth(verbose=True)

# Setup Inversion
inv = Inversion.BaseInversion(invProb, directiveList=[Alphas, Scales, beta,
                                                      petrodir, targets,
                                                      betaIt])

mcluster_map = inv.run(minit)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].plot(mesh.vectorCCx, wires.m1 * mcluster_map, 'b.-',ms=5, marker='v')
axes[0].plot(mesh.vectorCCx, wires.m1 * m, 'k--')
axes[0].set_title('Problem 1')
axes[0].legend(['Recovered Model', 'True Model'])
axes[0].set_xlabel('X')
axes[0].set_ylabel('Property 1')

axes[1].plot(mesh.vectorCCx, wires.m2 * mcluster_map, 'r.-',ms=5, marker='v')
axes[1].plot(mesh.vectorCCx, wires.m2 * m, 'k--')
axes[1].set_title('Problem 2')
axes[1].legend(['Recovered Model', 'True Model'])
axes[1].set_xlabel('X')
axes[1].set_ylabel('Property 2')

x, y = np.mgrid[-1:1:.01, -2:2:.01]
pos = np.empty(x.shape + (2,))
pos[:, :, 0] = x
pos[:, :, 1] = y
CS = axes[2].contour(x, y, np.exp(clfmapping.score_samples(
    pos.reshape(-1, 2)).reshape(x.shape)), 100, alpha=0.25, cmap='viridis')
axes[2].scatter(wires.m1 * mcluster_map, wires.m2 * mcluster_map, marker='v')
axes[2].set_title('Petro Distribution')
CS.collections[0].set_label('')
axes[2].legend(['True Petro Distribution', 'Recovered model crossplot'])
axes[2].set_xlabel('Property 1')
axes[2].set_ylabel('Property 2')

fig.suptitle(
    'Doodling with Mapping:\n'+
    'Joint inversion of 1D Linear Problems '+
     'with nonlinear petrophysical relationships',
     #y=1.
)
plt.subplots_adjust(wspace=0.3, top=0.8)
plt.show()
