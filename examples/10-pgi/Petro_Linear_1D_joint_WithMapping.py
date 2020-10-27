import discretize as Mesh
from SimPEG import (
    simulation,
    maps,
    data_misfit,
    directives,
    optimization,
    regularization,
    inverse_problem,
    inversion,
    utils
)
import numpy as np
import matplotlib.pyplot as plt

# Random seed for reproductibility
np.random.seed(1)
# Mesh
N = 100
mesh = Mesh.TensorMesh([N])

# Survey design parameters
nk = 30
jk = np.linspace(1., 59., nk)
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

poly0 = maps.PolynomialPetroClusterMap(coeffyx=np.r_[0., -4., 4.])
poly1 = maps.PolynomialPetroClusterMap(coeffyx=np.r_[-0., 3, 6, 6.])
poly0_inverse = maps.PolynomialPetroClusterMap(coeffyx=-np.r_[0., -4., 4.])
poly1_inverse = maps.PolynomialPetroClusterMap(coeffyx=-np.r_[0., 3, 6, 6.])
cluster_mapping = [maps.IdentityMap(), poly0_inverse, poly1_inverse]

m1 = np.zeros(100)
m1[20:41] = 1. + (poly0 * np.vstack([m0[20:41], m1[20:41]]).T)[:, 1]
m1[41:57] = -1. + (poly1 * np.vstack([m0[41:57], m1[41:57]]).T)[:, 1]

model2d = np.vstack([m0, m1]).T
m = utils.mkvc(model2d)
print(m0[20:41].mean(),m1[20:41].mean())
clfmapping = utils.GaussianMixtureWithMapping(
    n_components=3, covariance_type='full', tol=1e-6,
    reg_covar=1e-3, max_iter=100, n_init=100, init_params='kmeans',
    random_state=None, warm_start=False,
    means_init=np.array([[0,  0],
       [m0[20:41].mean(), m1[20:41].mean()],
       [m0[41:57].mean(), m1[41:57].mean()]]),
    verbose=0, verbose_interval=10, cluster_mapping=cluster_mapping
)
clfmapping = clfmapping.fit(model2d)
print(clfmapping.covariances_)

clfnomapping = utils.GaussianMixture(
    n_components=3, covariance_type='full', tol=1e-6,
    reg_covar=1e-3, max_iter=100, n_init=100, init_params='kmeans',
    random_state=None, warm_start=False,
    verbose=0, verbose_interval=10,
)
clfnomapping = clfnomapping.fit(model2d)
print('no mapping', clfnomapping.covariances_)

wires = maps.Wires(('m1', mesh.nC), ('m2', mesh.nC))
std=0.01
prob1 = simulation.LinearSimulation(mesh, G=G, model_map=wires.m1)
survey1 = prob1.make_synthetic_data(m, relative_error=std, add_noise=True)
survey1.eps = 0.

prob2 = simulation.LinearSimulation(mesh, G=G, model_map=wires.m2)
survey2 = prob2.make_synthetic_data(m, relative_error=std, add_noise=True)
survey2.eps = 0.

dmis1 = data_misfit.L2DataMisfit(simulation=prob1, data=survey1)
dmis2 = data_misfit.L2DataMisfit(simulation=prob2, data=survey2)
dmis = dmis1 + dmis2
minit = np.zeros_like(m)
print('dmsi1', dmis1(minit))
print('dmsi2', dmis2(minit))

# Distance weighting
wr1 = np.sum(prob1.G**2., axis=0)**0.5
wr1 = wr1 / np.max(wr1)
wr2 = np.sum(prob2.G**2., axis=0)**0.5
wr2 = wr2 / np.max(wr2)
print(wr2.shape)
wr = np.r_[wr1, wr2]
W = utils.sdiag(wr)

reg_simple = regularization.MakeSimplePetroWithMappingRegularization(
    mesh=mesh,
    GMmref=clfmapping,
    GMmodel=clfmapping,
    approx_gradient=True, alpha_x=0.,
    wiresmap=wires,
    evaltype='approx',
    cell_weights_list=[wr1, wr2])

opt = optimization.ProjectedGNCG(
    maxIter=30, tolX=1e-6, maxIterCG=100, tolCG=1e-3,
    lower=-10, upper=10,
)

invProb = inverse_problem.BaseInvProblem(dmis, reg_simple, opt)

# directives
alpha0_ratio = np.r_[
    np.zeros(len(reg_simple.objfcts[0].objfcts)),
    100. * np.ones(len(reg_simple.objfcts[1].objfcts)),
    .4 * np.ones(len(reg_simple.objfcts[2].objfcts))
]
Alphas = directives.AlphasSmoothEstimate_ByEig(alpha0_ratio=alpha0_ratio, ninit=10, verbose=True)
Scales = directives.ScalingEstimate_ByEig(Chi0_ratio=.4, verbose=True, ninit=10)
beta = directives.BetaEstimate_ByEig(beta0_ratio=1e-5, ninit=10)
betaIt = directives.PetroBetaReWeighting(
    verbose=True, rateCooling=2., rateWarming=1.,
    tolerance=0., UpdateRate=1,
    ratio_in_cooling=False,
    progress=0.2,
)
targets = directives.PetroTargetMisfit(verbose=True)
petrodir = directives.UpdateReference()

# Setup Inversion
inv = inversion.BaseInversion(invProb, directiveList=[Alphas, Scales, beta,
                                                      petrodir, targets,
                                                      betaIt])

mcluster_map = inv.run(minit)

# Inversion with no nonlinear mapping
reg_simple_no_map = regularization.MakeSimplePetroRegularization(
    mesh=mesh,
    GMmref=clfnomapping,
    GMmodel=clfnomapping,
    approx_gradient=True, alpha_x=0.,
    wiresmap=wires,
    evaltype='approx',
    cell_weights_list=[wr1, wr2])

opt = optimization.ProjectedGNCG(
    maxIter=20, tolX=1e-6, maxIterCG=100, tolCG=1e-3,
    lower=-10, upper=10,
)

invProb = inverse_problem.BaseInvProblem(dmis, reg_simple_no_map, opt)

# directives
Alphas = directives.AlphasSmoothEstimate_ByEig(
    alpha0_ratio=alpha0_ratio, ninit=10, verbose=True)
Scales = directives.ScalingEstimate_ByEig(
    Chi0_ratio=.4, verbose=True, ninit=100)
beta = directives.BetaEstimate_ByEig(beta0_ratio=1e-5, ninit=100)
betaIt = directives.PetroBetaReWeighting(
    verbose=True, rateCooling=2., rateWarming=1.,
    tolerance=0.0, UpdateRate=1,
    ratio_in_cooling=False,
    progress=0.2,
)
targets = directives.PetroTargetMisfit(chiSmall=1.,
                                       TriggerSmall=True, TriggerTheta=False, verbose=True)
petrodir = directives.UpdateReference()

# Setup Inversion
inv = inversion.BaseInversion(invProb, directiveList=[Alphas, Scales, beta,
                                                      petrodir, targets,
                                                      betaIt])

mcluster_no_map = inv.run(minit)

# Tikhonov Inversion

reg1 = regularization.Tikhonov(
    mesh, alpha_s=1., alpha_x=1., mapping=wires.m1
)
reg1.cell_weights = wr1
reg2 = regularization.Tikhonov(
    mesh, alpha_s=1., alpha_x=1., mapping=wires.m2
)
reg2.cell_weights = wr2
reg = reg1 + reg2

opt = optimization.ProjectedGNCG(
    maxIter=20, tolX=1e-6, maxIterCG=100, tolCG=1e-3,
    lower=-10, upper=10,
)

invProb = inverse_problem.BaseInvProblem(dmis, reg, opt)

# directives
Alphas = directives.AlphasSmoothEstimate_ByEig(
    alpha0_ratio=1e-1, ninit=10, verbose=True)
Scales = directives.ScalingEstimate_ByEig(
    Chi0_ratio=.4, verbose=True, ninit=100)
beta = directives.BetaEstimate_ByEig(beta0_ratio=1e-5, ninit=100)
targets = directives.PetroTargetMisfit(
    chiSmall=1.,
    TriggerSmall=False,
    TriggerTheta=False,
    verbose=False,
)

# Setup Inversion
inv = inversion.BaseInversion(invProb, directiveList=[Scales, beta,
                                                      targets,
                                                      ])

mtik = inv.run(minit)


# Final Plot
fig, axes = plt.subplots(3, 4, figsize=(25, 15))
axes = axes.reshape(12)
left, width = .25, .5
bottom, height = .25, .5
right = left + width
top = bottom + height

axes[0].set_axis_off()
axes[0].text(
    0.5 * (left + right), 0.5 * (bottom + top),
    ('Using true nonlinear\npetrophysical relationships'),
    horizontalalignment='center',
    verticalalignment='center',
    fontsize=20, color='black',
    transform=axes[0].transAxes
)

axes[1].plot(mesh.vectorCCx, wires.m1 * mcluster_map, 'b.-', ms=5, marker='v')
axes[1].plot(mesh.vectorCCx, wires.m1 * m, 'k--')
axes[1].set_title('Problem 1')
axes[1].legend(['Recovered Model', 'True Model'], loc=1)
axes[1].set_xlabel('X')
axes[1].set_ylabel('Property 1')

axes[2].plot(mesh.vectorCCx, wires.m2 * mcluster_map, 'r.-', ms=5, marker='v')
axes[2].plot(mesh.vectorCCx, wires.m2 * m, 'k--')
axes[2].set_title('Problem 2')
axes[2].legend(['Recovered Model', 'True Model'], loc=1)
axes[2].set_xlabel('X')
axes[2].set_ylabel('Property 2')

x, y = np.mgrid[-1:1:.01, -2:2:.01]
pos = np.empty(x.shape + (2,))
pos[:, :, 0] = x
pos[:, :, 1] = y
CS = axes[3].contour(x, y, np.exp(clfmapping.score_samples(
    pos.reshape(-1, 2)).reshape(x.shape)), 100, alpha=0.25, cmap='viridis')
axes[3].scatter(wires.m1 * mcluster_map, wires.m2 * mcluster_map, marker='v')
axes[3].set_title('Petrophysical Distribution')
CS.collections[0].set_label('')
axes[3].legend(['True Petrophysical Distribution', 'Recovered model crossplot'])
axes[3].set_xlabel('Property 1')
axes[3].set_ylabel('Property 2')

#fig.suptitle(
#    'Doodling with Mapping: one mapping per identified rock unit\n' +
#    'Joint inversion of 1D Linear Problems ' +
#    'with nonlinear petrophysical relationships',
#    fontsize=24
#)

axes[4].set_axis_off()
axes[4].text(
    0.5 * (left + right), 0.5 * (bottom + top),
    ('Using a pure\nGaussian distribution'),
    horizontalalignment='center',
    verticalalignment='center',
    fontsize=20, color='black',
    transform=axes[4].transAxes
)

axes[5].plot(mesh.vectorCCx, wires.m1 *
             mcluster_no_map, 'b.-', ms=5, marker='v')
#axes[5].plot(mesh.vectorCCx, wires.m1 * reg_simple_no_map.objfcts[0].mref, 'g--')

axes[5].plot(mesh.vectorCCx, wires.m1 * m, 'k--')
axes[5].set_title('Problem 1')
axes[5].legend(['Recovered Model', 'True Model'], loc=1)
axes[5].set_xlabel('X')
axes[5].set_ylabel('Property 1')

axes[6].plot(mesh.vectorCCx, wires.m2 *
             mcluster_no_map, 'r.-', ms=5, marker='v')
#axes[6].plot(mesh.vectorCCx, wires.m2 * reg_simple_no_map.objfcts[0].mref, 'g--')

axes[6].plot(mesh.vectorCCx, wires.m2 * m, 'k--')
axes[6].set_title('Problem 2')
axes[6].legend(['Recovered Model', 'True Model'], loc=1)
axes[6].set_xlabel('X')
axes[6].set_ylabel('Property 2')

CSF = axes[7].contour(x, y, np.exp(clfmapping.score_samples(
    pos.reshape(-1, 2)).reshape(x.shape)), 100, alpha=0.5)  # , cmap='viridis')
CS = axes[7].contour(x, y, np.exp(clfnomapping.score_samples(
    pos.reshape(-1, 2)).reshape(x.shape)), 500, cmap='viridis')
axes[7].scatter(wires.m1 * mcluster_no_map,
                wires.m2 * mcluster_no_map, marker='v')
axes[7].set_title('Petrophysical Distribution')
CSF.collections[0].set_label('')
CS.collections[0].set_label('')
axes[7].legend(
    [
        'True Petro. Distribution',
        'Modeled Petro. Distribution',
        'Recovered model crossplot'
    ]
)
axes[7].set_xlabel('Property 1')
axes[7].set_ylabel('Property 2')

# Tikonov

axes[8].set_axis_off()
axes[8].text(
    0.5 * (left + right), 0.5 * (bottom + top),
    ('Tikhonov\n~Using a single cluster'),
    horizontalalignment='center',
    verticalalignment='center',
    fontsize=20, color='black',
    transform=axes[8].transAxes
)

axes[9].plot(mesh.vectorCCx, wires.m1 *
             mtik, 'b.-', ms=5, marker='v')
axes[9].plot(mesh.vectorCCx, wires.m1 * m, 'k--')
axes[9].set_title('Problem 1')
axes[9].legend(['Recovered Model', 'True Model'], loc=1)
axes[9].set_xlabel('X')
axes[9].set_ylabel('Property 1')

axes[10].plot(mesh.vectorCCx, wires.m2 *
              mtik, 'r.-', ms=5, marker='v')
axes[10].plot(mesh.vectorCCx, wires.m2 * m, 'k--')
axes[10].set_title('Problem 2')
axes[10].legend(['Recovered Model', 'True Model'], loc=1)
axes[10].set_xlabel('X')
axes[10].set_ylabel('Property 2')

CS = axes[11].contour(x, y, np.exp(clfmapping.score_samples(
    pos.reshape(-1, 2)).reshape(x.shape)), 100, alpha=0.25, cmap='viridis')
axes[11].scatter(wires.m1 * mtik,
                 wires.m2 * mtik, marker='v')
axes[11].set_title('Petro Distribution')
CS.collections[0].set_label('')
axes[11].legend(
    [
        'True Petro Distribution',
        'Recovered model crossplot'
    ]
)
axes[11].set_xlabel('Property 1')
axes[11].set_ylabel('Property 2')
plt.subplots_adjust(wspace=0.3, hspace=0.3, top=0.85)
plt.show()
