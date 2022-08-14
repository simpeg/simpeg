"""
Petrophysically guided inversion: Joint linear example with nonlinear relationships
===================================================================================

We do a comparison between the classic least-squares inversion
and our formulation of a petrophysically guided inversion.
We explore it through coupling two linear problems whose respective physical
properties are linked by polynomial relationships that change between rock units.

"""

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
    utils,
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
jk = np.linspace(1.0, 59.0, nk)
p = -0.25
q = 0.25


# Physics
def g(k):
    return np.exp(p * jk[k] * mesh.cell_centers_x) * np.cos(
        np.pi * q * jk[k] * mesh.cell_centers_x
    )


G = np.empty((nk, mesh.nC))

for i in range(nk):
    G[i, :] = g(i)

m0 = np.zeros(mesh.nC)
m0[20:41] = np.linspace(0.0, 1.0, 21)
m0[41:57] = np.linspace(-1, 0.0, 16)

poly0 = maps.PolynomialPetroClusterMap(coeffyx=np.r_[0.0, -4.0, 4.0])
poly1 = maps.PolynomialPetroClusterMap(coeffyx=np.r_[-0.0, 3.0, 6.0, 6.0])
poly0_inverse = maps.PolynomialPetroClusterMap(coeffyx=-np.r_[0.0, -4.0, 4.0])
poly1_inverse = maps.PolynomialPetroClusterMap(coeffyx=-np.r_[0.0, 3.0, 6.0, 6.0])
cluster_mapping = [maps.IdentityMap(), poly0_inverse, poly1_inverse]

m1 = np.zeros(100)
m1[20:41] = 1.0 + (poly0 * np.vstack([m0[20:41], m1[20:41]]).T)[:, 1]
m1[41:57] = -1.0 + (poly1 * np.vstack([m0[41:57], m1[41:57]]).T)[:, 1]

model2d = np.vstack([m0, m1]).T
m = utils.mkvc(model2d)

clfmapping = utils.GaussianMixtureWithNonlinearRelationships(
    mesh=mesh,
    n_components=3,
    covariance_type="full",
    tol=1e-8,
    reg_covar=1e-3,
    max_iter=1000,
    n_init=100,
    init_params="kmeans",
    random_state=None,
    warm_start=False,
    means_init=np.array(
        [
            [0, 0],
            [m0[20:41].mean(), m1[20:41].mean()],
            [m0[41:57].mean(), m1[41:57].mean()],
        ]
    ),
    verbose=0,
    verbose_interval=10,
    cluster_mapping=cluster_mapping,
)
clfmapping = clfmapping.fit(model2d)

clfnomapping = utils.WeightedGaussianMixture(
    mesh=mesh,
    n_components=3,
    covariance_type="full",
    tol=1e-8,
    reg_covar=1e-3,
    max_iter=1000,
    n_init=100,
    init_params="kmeans",
    random_state=None,
    warm_start=False,
    verbose=0,
    verbose_interval=10,
)
clfnomapping = clfnomapping.fit(model2d)

wires = maps.Wires(("m1", mesh.nC), ("m2", mesh.nC))

relatrive_error = 0.01
noise_floor = 0.0

prob1 = simulation.LinearSimulation(mesh, G=G, model_map=wires.m1)
survey1 = prob1.make_synthetic_data(
    m, relative_error=relatrive_error, noise_floor=noise_floor, add_noise=True
)

prob2 = simulation.LinearSimulation(mesh, G=G, model_map=wires.m2)
survey2 = prob2.make_synthetic_data(
    m, relative_error=relatrive_error, noise_floor=noise_floor, add_noise=True
)


dmis1 = data_misfit.L2DataMisfit(simulation=prob1, data=survey1)
dmis2 = data_misfit.L2DataMisfit(simulation=prob2, data=survey2)
dmis = dmis1 + dmis2
minit = np.zeros_like(m)

# Distance weighting
wr1 = np.sum(prob1.G ** 2.0, axis=0) ** 0.5 / mesh.cell_volumes
wr1 = wr1 / np.max(wr1)
wr2 = np.sum(prob2.G ** 2.0, axis=0) ** 0.5 / mesh.cell_volumes
wr2 = wr2 / np.max(wr2)

reg_simple = regularization.PGI(
    mesh=mesh,
    gmmref=clfmapping,
    gmm=clfmapping,
    approx_gradient=True,
    wiresmap=wires,
    non_linear_relationships=True,
    weights_list=[wr1, wr2],
)

opt = optimization.ProjectedGNCG(
    maxIter=50,
    tolX=1e-6,
    maxIterCG=100,
    tolCG=1e-3,
    lower=-10,
    upper=10,
)

invProb = inverse_problem.BaseInvProblem(dmis, reg_simple, opt)

# directives
scales = directives.ScalingMultipleDataMisfits_ByEig(
    chi0_ratio=np.r_[1.0, 1.0], verbose=True, n_pw_iter=10
)
scaling_schedule = directives.JointScalingSchedule(verbose=True)
alpha0_ratio = np.r_[1e6, 1e4, 1, 1]
alphas = directives.AlphasSmoothEstimate_ByEig(
    alpha0_ratio=alpha0_ratio, n_pw_iter=10, verbose=True
)
beta = directives.BetaEstimate_ByEig(beta0_ratio=1e-5, n_pw_iter=10)
betaIt = directives.PGI_BetaAlphaSchedule(
    verbose=True,
    coolingFactor=2.0,
    progress=0.2,
)
targets = directives.MultiTargetMisfits(verbose=True)
petrodir = directives.PGI_UpdateParameters(update_gmm=False)

# Setup Inversion
inv = inversion.BaseInversion(
    invProb,
    directiveList=[alphas, scales, beta, petrodir, targets, betaIt, scaling_schedule],
)

mcluster_map = inv.run(minit)

# Inversion with no nonlinear mapping
reg_simple_no_map = regularization.PGI(
    mesh=mesh,
    gmmref=clfnomapping,
    gmm=clfnomapping,
    approx_gradient=True,
    wiresmap=wires,
    non_linear_relationships=False,
    weights_list=[wr1, wr2],
)

opt = optimization.ProjectedGNCG(
    maxIter=50,
    tolX=1e-6,
    maxIterCG=100,
    tolCG=1e-3,
    lower=-10,
    upper=10,
)

invProb = inverse_problem.BaseInvProblem(dmis, reg_simple_no_map, opt)

# directives
scales = directives.ScalingMultipleDataMisfits_ByEig(
    chi0_ratio=np.r_[1.0, 1.0], verbose=True, n_pw_iter=10
)
scaling_schedule = directives.JointScalingSchedule(verbose=True)
alpha0_ratio = np.r_[
    100.0 * np.ones(2), 1, 1
]
alphas = directives.AlphasSmoothEstimate_ByEig(
    alpha0_ratio=alpha0_ratio, n_pw_iter=10, verbose=True
)
beta = directives.BetaEstimate_ByEig(beta0_ratio=1e-5, n_pw_iter=10)
betaIt = directives.PGI_BetaAlphaSchedule(
    verbose=True,
    coolingFactor=2.0,
    progress=0.2,
)
targets = directives.MultiTargetMisfits(
    chiSmall=1.0, TriggerSmall=True, TriggerTheta=False, verbose=True
)
petrodir = directives.PGI_UpdateParameters(update_gmm=False)

# Setup Inversion
inv = inversion.BaseInversion(
    invProb,
    directiveList=[alphas, scales, beta, petrodir, targets, betaIt, scaling_schedule],
)

mcluster_no_map = inv.run(minit)

# WeightedLeastSquares Inversion

reg1 = regularization.WeightedLeastSquares(
    mesh, alpha_s=1.0, alpha_x=1.0, mapping=wires.m1
)
reg1.cell_weights = wr1
reg2 = regularization.WeightedLeastSquares(
    mesh, alpha_s=1.0, alpha_x=1.0, mapping=wires.m2
)
reg2.cell_weights = wr2
reg = reg1 + reg2

opt = optimization.ProjectedGNCG(
    maxIter=50,
    tolX=1e-6,
    maxIterCG=100,
    tolCG=1e-3,
    lower=-10,
    upper=10,
)

invProb = inverse_problem.BaseInvProblem(dmis, reg, opt)

# directives
alpha0_ratio = np.r_[1, 1, 1, 1]
alphas = directives.AlphasSmoothEstimate_ByEig(
    alpha0_ratio=alpha0_ratio, n_pw_iter=10, verbose=True
)
scales = directives.ScalingMultipleDataMisfits_ByEig(
    chi0_ratio=np.r_[1.0, 1.0], verbose=True, n_pw_iter=10
)
scaling_schedule = directives.JointScalingSchedule(verbose=True)
beta = directives.BetaEstimate_ByEig(beta0_ratio=1e-5, n_pw_iter=10)
beta_schedule = directives.BetaSchedule(coolingFactor=5.0, coolingRate=1)
targets = directives.MultiTargetMisfits(
    TriggerSmall=False,
    verbose=True,
)

# Setup Inversion
inv = inversion.BaseInversion(
    invProb,
    directiveList=[alphas, scales, beta, targets, beta_schedule, scaling_schedule],
)

mtik = inv.run(minit)


# Final Plot
fig, axes = plt.subplots(3, 4, figsize=(25, 15))
axes = axes.reshape(12)
left, width = 0.25, 0.5
bottom, height = 0.25, 0.5
right = left + width
top = bottom + height

axes[0].set_axis_off()
axes[0].text(
    0.5 * (left + right),
    0.5 * (bottom + top),
    ("Using true nonlinear\npetrophysical relationships"),
    horizontalalignment="center",
    verticalalignment="center",
    fontsize=20,
    color="black",
    transform=axes[0].transAxes,
)

axes[1].plot(mesh.cell_centers_x, wires.m1 * mcluster_map, "b.-", ms=5, marker="v")
axes[1].plot(mesh.cell_centers_x, wires.m1 * m, "k--")
axes[1].set_title("Problem 1")
axes[1].legend(["Recovered Model", "True Model"], loc=1)
axes[1].set_xlabel("X")
axes[1].set_ylabel("Property 1")

axes[2].plot(mesh.cell_centers_x, wires.m2 * mcluster_map, "r.-", ms=5, marker="v")
axes[2].plot(mesh.cell_centers_x, wires.m2 * m, "k--")
axes[2].set_title("Problem 2")
axes[2].legend(["Recovered Model", "True Model"], loc=1)
axes[2].set_xlabel("X")
axes[2].set_ylabel("Property 2")

x, y = np.mgrid[-1:1:0.01, -4:2:0.01]
pos = np.empty(x.shape + (2,))
pos[:, :, 0] = x
pos[:, :, 1] = y
CS = axes[3].contour(
    x,
    y,
    np.exp(clfmapping.score_samples(pos.reshape(-1, 2)).reshape(x.shape)),
    100,
    alpha=0.25,
    cmap="viridis",
)
axes[3].scatter(wires.m1 * mcluster_map, wires.m2 * mcluster_map, marker="v")
axes[3].set_title("Petrophysical Distribution")
CS.collections[0].set_label("")
axes[3].legend(["True Petrophysical Distribution", "Recovered model crossplot"])
axes[3].set_xlabel("Property 1")
axes[3].set_ylabel("Property 2")

axes[4].set_axis_off()
axes[4].text(
    0.5 * (left + right),
    0.5 * (bottom + top),
    ("Using a pure\nGaussian distribution"),
    horizontalalignment="center",
    verticalalignment="center",
    fontsize=20,
    color="black",
    transform=axes[4].transAxes,
)

axes[5].plot(mesh.cell_centers_x, wires.m1 * mcluster_no_map, "b.-", ms=5, marker="v")
axes[5].plot(mesh.cell_centers_x, wires.m1 * m, "k--")
axes[5].set_title("Problem 1")
axes[5].legend(["Recovered Model", "True Model"], loc=1)
axes[5].set_xlabel("X")
axes[5].set_ylabel("Property 1")

axes[6].plot(mesh.cell_centers_x, wires.m2 * mcluster_no_map, "r.-", ms=5, marker="v")
axes[6].plot(mesh.cell_centers_x, wires.m2 * m, "k--")
axes[6].set_title("Problem 2")
axes[6].legend(["Recovered Model", "True Model"], loc=1)
axes[6].set_xlabel("X")
axes[6].set_ylabel("Property 2")

CSF = axes[7].contour(
    x,
    y,
    np.exp(clfmapping.score_samples(pos.reshape(-1, 2)).reshape(x.shape)),
    100,
    alpha=0.5,
    label="True Petro. Distribution",
)
CS = axes[7].contour(
    x,
    y,
    np.exp(clfnomapping.score_samples(pos.reshape(-1, 2)).reshape(x.shape)),
    500,
    cmap="viridis",
    linestyles="--",
    label="Modeled Petro. Distribution",
)
axes[7].scatter(
    wires.m1 * mcluster_no_map,
    wires.m2 * mcluster_no_map,
    marker="v",
    label="Recovered model crossplot",
)
axes[7].set_title("Petrophysical Distribution")
axes[7].legend()
axes[7].set_xlabel("Property 1")
axes[7].set_ylabel("Property 2")

# Tikonov

axes[8].set_axis_off()
axes[8].text(
    0.5 * (left + right),
    0.5 * (bottom + top),
    ("Least-Squares\n~Using a single cluster"),
    horizontalalignment="center",
    verticalalignment="center",
    fontsize=20,
    color="black",
    transform=axes[8].transAxes,
)

axes[9].plot(mesh.cell_centers_x, wires.m1 * mtik, "b.-", ms=5, marker="v")
axes[9].plot(mesh.cell_centers_x, wires.m1 * m, "k--")
axes[9].set_title("Problem 1")
axes[9].legend(["Recovered Model", "True Model"], loc=1)
axes[9].set_xlabel("X")
axes[9].set_ylabel("Property 1")

axes[10].plot(mesh.cell_centers_x, wires.m2 * mtik, "r.-", ms=5, marker="v")
axes[10].plot(mesh.cell_centers_x, wires.m2 * m, "k--")
axes[10].set_title("Problem 2")
axes[10].legend(["Recovered Model", "True Model"], loc=1)
axes[10].set_xlabel("X")
axes[10].set_ylabel("Property 2")

CS = axes[11].contour(
    x,
    y,
    np.exp(clfmapping.score_samples(pos.reshape(-1, 2)).reshape(x.shape)),
    100,
    alpha=0.25,
    cmap="viridis",
)
axes[11].scatter(wires.m1 * mtik, wires.m2 * mtik, marker="v")
axes[11].set_title("Petro Distribution")
CS.collections[0].set_label("")
axes[11].legend(["True Petro Distribution", "Recovered model crossplot"])
axes[11].set_xlabel("Property 1")
axes[11].set_ylabel("Property 2")
plt.subplots_adjust(wspace=0.3, hspace=0.3, top=0.85)
plt.show()
