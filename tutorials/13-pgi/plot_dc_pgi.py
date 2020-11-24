"""
DC: Petrophysically Guided Inversion (PGI) with various levels of information
=============================================================================

Invert dc data with petrophysical information

Example inspired by: Thibaut Astic, Douglas W Oldenburg, A framework for petrophysically and geologically guided geophysical inversion using a dynamic Gaussian mixture model prior, Geophysical Journal International, Volume 219, Issue 3, December 2019, Pages 1989–2012, https://doi.org/10.1093/gji/ggz389

A DC resistivity profile is acquired over two cylinders. We illustrate the performance of this framework when no physical property mean values are available, and compared it to the result with full petrophysical information. We highlight then how geological information from borehole logs can be incorporated into this framework.

For that purpose, we first start by running a PGI with full petrophysical information to set benchmarks. We then run a PGI without providing any information about the physical property mean values nor the proportions. We finally run another PGI, still without means information, but with added geological information included through the use of local proportions. All inversions share the same starting weighting of the geophysical objective function terms.

"""

# Import
import discretize as Mesh
from SimPEG import (
    maps,
    data,
    utils,
    data_misfit,
    regularization,
    optimization,
    inverse_problem,
    directives,
    inversion,
)
from SimPEG.electromagnetics.static import resistivity as DC, utils as DCutils
import numpy as np
import copy
import matplotlib.pyplot as plt
from scipy.stats import norm

try:
    from pymatsolver import Pardiso as Solver
except ImportError:
    from SimPEG import SolverLU as Solver

## Reproducible science
seed = 12345
np.random.seed(seed)

## Function to plot cylinder border
def getCylinderPoints(xc, zc, r):
    xLocOrig1 = np.arange(-r, r + r / 10.0, r / 10.0)
    xLocOrig2 = np.arange(r, -r - r / 10.0, -r / 10.0)
    # Top half of cylinder
    zLoc1 = np.sqrt(-(xLocOrig1 ** 2.0) + r ** 2.0) + zc
    # Bottom half of cylinder
    zLoc2 = -np.sqrt(-(xLocOrig2 ** 2.0) + r ** 2.0) + zc
    # Shift from x = 0 to xc
    xLoc1 = xLocOrig1 + xc * np.ones_like(xLocOrig1)
    xLoc2 = xLocOrig2 + xc * np.ones_like(xLocOrig2)

    topHalf = np.vstack([xLoc1, zLoc1]).T
    topHalf = topHalf[0:-1, :]
    bottomHalf = np.vstack([xLoc2, zLoc2]).T
    bottomHalf = bottomHalf[0:-1, :]

    cylinderPoints = np.vstack([topHalf, bottomHalf])
    cylinderPoints = np.vstack([cylinderPoints, topHalf[0, :]])
    return cylinderPoints


# Setup
#######

# 2D Mesh
## Cell sizes
csx, csy, csz = 0.25, 0.25, 0.25
## Number of core cells in each direction
ncx, ncz = 123, 61
## Number of padding cells to add in each direction
npad = 12
## Vectors of cell lengthts in each direction
hx = [(csx, npad, -1.5), (csx, ncx), (csx, npad, 1.5)]
hz = [(csz, npad, -1.5), (csz, ncz)]
## Create mesh
mesh = Mesh.TensorMesh([hx, hz], x0="CN")
mesh.x0[1] = mesh.x0[1] + csz / 2.0

# 2-cylinders Model Creation
## cylinder parameters
x0, z0, r0 = -6.0, -5.0, 3.0
x1, z1, r1 = 6.0, -5.0, 3.0

## units ln-conductivities
ln_sigback = -np.log(100.0)
ln_sigc = -np.log(50.0)
ln_sigr = -np.log(250.0)

## Add some variability to the physical property model
noisemean = 0.0
noisevar = np.sqrt(0.001)
ln_over = -2.0

## model creation: bakcground
mtrue = ln_sigback * np.ones(mesh.nC) + norm(noisemean, noisevar).rvs(mesh.nC)

## model creation: add conductive cylinder
csph = (np.sqrt((mesh.gridCC[:, 1] - z0) ** 2.0 + (mesh.gridCC[:, 0] - x0) ** 2.0)) < r0
mtrue[csph] = ln_sigc * np.ones_like(mtrue[csph]) + norm(noisemean, noisevar).rvs(
    np.prod((mtrue[csph]).shape)
)

## model creation: add resistive cylinder
rsph = (np.sqrt((mesh.gridCC[:, 1] - z1) ** 2.0 + (mesh.gridCC[:, 0] - x1) ** 2.0)) < r1
mtrue[rsph] = ln_sigr * np.ones_like(mtrue[rsph]) + norm(noisemean, noisevar).rvs(
    np.prod((mtrue[rsph]).shape)
)

## Area of interest, define Core Mesh
xmin, xmax = -15.0, 15
ymin, ymax = -15.0, 0.0
xyzlim = np.r_[[[xmin, xmax], [ymin, ymax]]]
actcore, meshCore = utils.ExtractCoreMesh(xyzlim, mesh)
actind = np.ones_like(actcore)

# Survey
########

# Setup a Dipole-Dipole Survey with 1m and 2m dipoles
xmin, xmax = -15.0, 15.0
ymin, ymax = 0.0, 0.0
zmin, zmax = 0, 0

endl = np.array([[xmin, ymin, zmin], [xmax, ymax, zmax]])
survey1 = DCutils.gen_DCIPsurvey(
    endl, survey_type="dipole-dipole", dim=mesh.dim, a=1, b=1, n=16, d2flag="2.5D"
)
survey2 = DCutils.gen_DCIPsurvey(
    endl, survey_type="dipole-dipole", dim=mesh.dim, a=2, b=2, n=16, d2flag="2.5D"
)

survey = DC.Survey_ky(survey1.srcList + survey2.srcList)

# Setup Problem with exponential mapping and Active cells only in the core mesh
expmap = maps.ExpMap(mesh)
mapactive = maps.InjectActiveCells(
    mesh=mesh, indActive=actcore, valInactive=-np.log(100)
)
mapping = expmap * mapactive
simulation = DC.simulation_2d.Simulation2DNodal(
    mesh, survey=survey, sigmaMap=mapping, Solver=Solver, nky=8
)

relative_measurement_error = 0.02
dc_data = simulation.make_synthetic_data(
    mtrue[actcore],
    relative_error=relative_measurement_error,
    force=True,
    add_noise=True,
)
survey.eps = 1e-4

# Gaussian Mixture Model (GMM) representing the petrophysical distribution
##########################################################################

# Generate the GMM petrophysical distribution
n = 3
clf = utils.WeightedGaussianMixture(
    n_components=n,
    mesh=meshCore,
    covariance_type="full",
    reg_covar=1e-3,
    means_init=np.r_[-np.log(100.0), -np.log(50.0), -np.log(250.0)][:, np.newaxis],
)
## Validate the GMM object (sklearn requirement)
clf.fit(mtrue[actcore].reshape(-1, 1))

# Manually setting the GMM parameters
## Order cluster by order of importance
clf.order_clusters_GM_weight()
## Set cluster means
clf.means_ = np.r_[-np.log(100.0), -np.log(50.0), -np.log(250.0)][:, np.newaxis]
## Set clusters variance
clf.covariances_ = np.array([[[0.001]], [[0.001]], [[0.001]],])
##Set clusters precision and Cholesky decomposition from variances
clf.compute_clusters_precisions()

# PGI with full petrophysical information and Least-Squares Approximation of the Regularizer
############################################################################################

# Set the initial model to the true background mean
m0 = -np.log(100.0) * np.ones(mapping.nP)

# Create data misfit object
dmis = data_misfit.L2DataMisfit(data=dc_data, simulation=simulation)

# Create the regularization with GMM information
idenMap = maps.IdentityMap(nP=m0.shape[0])
wires = maps.Wires(("m", m0.shape[0]))
## By default the PGI regularization uses the least-squares approximation.
## It requires then the directives.PGI_UpdateParameters()
reg_mean = regularization.SimplePGI(
    gmmref=clf, mesh=mesh, wiresmap=wires, maplist=[idenMap], mref=m0, indActive=actcore
)

# Regularization Weighting
betavalue = 100.0
alpha_s = 0.016
alpha_x = 100.0
alpha_y = 100.0
reg_mean.alpha_s = alpha_s
reg_mean.alpha_x = alpha_x
reg_mean.alpha_y = alpha_y
reg_mean.mrefInSmooth = False

# Optimization
opt = optimization.ProjectedGNCG(
    maxIter=30, lower=-10, upper=10, maxIterLS=20, maxIterCG=100, tolCG=1e-5
)
opt.remember("xc")

# Set the inverse problem
invProb = inverse_problem.BaseInvProblem(dmis, reg_mean, opt)
## Starting beta
invProb.beta = betavalue

# Inversion directives
## Beta Strategy with Beta and Alpha
beta_alpha_iteration = directives.PGI_BetaAlphaSchedule(
    verbose=True,
    coolingFactor=5.0,
    tolerance=0.05,  # Tolerance on Phi_d for beta-cooling
    progress=0.1,  # Minimum progress, else beta-cooling
)
## PGI multi-target misfits
targets = directives.MultiTargetMisfits(verbose=True,)
## Put learned reference model in Smoothness once stable
MrefInSmooth = directives.PGI_AddMrefInSmooth(verbose=True)
## PGI update to the GMM and Smallness reference model and weights
## **This one is required when using the Least-Squares approximation of PGI (default)
petrodir = directives.PGI_UpdateParameters()
## Sensitivity weights based on the starting half-space
updateSensW = directives.UpdateSensitivityWeights(threshold=1e-3, everyIter=False)
## Preconditioner
update_Jacobi = directives.UpdatePreconditioner()

# Inversion
inv = inversion.BaseInversion(
    invProb,
    # directives list: the order matters!
    directiveList=[
        updateSensW,
        petrodir,
        targets,
        beta_alpha_iteration,
        MrefInSmooth,
        update_Jacobi,
    ],
)

# Run the inversion
#m_pgi = inv.run(m0)


# PGI without mean information
##############################

# We start from the best fitting half-space value
m0 = -np.log(np.median((DCutils.apparent_resistivity(dc_data)))) * np.ones(mapping.nP)

# We now learn a suitable GMM
# Create an initial GMM with guest values
clfnomean = utils.WeightedGaussianMixture(
    n_components=n, mesh=meshCore, covariance_type="full", reg_covar=1e-3, n_init=20,
)
clfnomean.fit(mtrue[actcore].reshape(-1, 1))
clfnomean.order_clusters_GM_weight()
# Set qualitative initial means; the value chosen here are based on
# the range of apparent conductivities from the data
clfnomean.means_ = np.r_[
    # background is guest as the best fitting half-space value
    -np.log(np.median((DCutils.apparent_resistivity(dc_data)))),
    # conductive and resistive are min and max of the apparent conducitivities
    -np.log(np.min((DCutils.apparent_resistivity(dc_data)))),
    -np.log(np.max((DCutils.apparent_resistivity(dc_data)))),
][:, np.newaxis]
clfnomean.covariances_ = np.array([[[0.001]], [[0.001]], [[0.001]],])
clfnomean.compute_clusters_precisions()

# Create the PGI regularization
reg_nomean = regularization.SimplePGI(
    gmmref=clfnomean,
    mesh=mesh,
    wiresmap=wires,
    maplist=[idenMap],
    mref=m0,
    indActive=actcore,
)
reg_nomean.mrefInSmooth = False
reg_nomean.alpha_s = alpha_s
reg_nomean.alpha_x = alpha_x
reg_nomean.alpha_y = alpha_y

# Optimization
opt = optimization.ProjectedGNCG(
    maxIter=30, lower=-10, upper=10, maxIterLS=20, maxIterCG=100, tolCG=1e-5
)
opt.remember("xc")

# Set the inverse problem
invProb = inverse_problem.BaseInvProblem(dmis, reg_nomean, opt)
invProb.beta = betavalue

# Inversion directives
betaIt = directives.PGI_BetaAlphaSchedule(
    verbose=True, coolingFactor=5.0, tolerance=0.05, progress=0.1
)
targets = directives.MultiTargetMisfits(verbose=True,)
# kappa, nu and zeta set the learning of the GMM
petrodir = directives.PGI_UpdateParameters(
    kappa=0.0,  # No influence from Prior means in the learning
    nu=1e8,  # Fixed variances
    zeta=0.0,  # Prior GMM proportions have no influeance
)
updateSensW = directives.UpdateSensitivityWeights(threshold=1e-3, everyIter=False)
update_Jacobi = directives.UpdatePreconditioner()
MrefInSmooth = directives.PGI_AddMrefInSmooth(verbose=True)

inv = inversion.BaseInversion(
    invProb,
    directiveList=[
        updateSensW,
        petrodir,
        targets,
        betaIt,
        MrefInSmooth,
        update_Jacobi,
    ],
)

# Run the inversion
#m_pgi_nomean = inv.run(m0)


# PGI with local proportions
############################

# In this section, we force the occurrences of the resistive and conductive
# anomalies within a certain depth range through the use of local GMM proportions

## Build the local proportions over the mesh
## Algorithm is in Log-probability, we need a minimum proportions to avoid infinities warning
tol_log = 1e-16
## define proportions over the mesh
## default is only background
proportions_mesh = np.ones((meshCore.nC, 3)) * np.r_[1.0, tol_log, tol_log]
## Force anomalous units between depth of 2 m and 8 m
below_2m = meshCore.gridCC[:, 1] <= -2
above_8m = meshCore.gridCC[:, 1] >= -8
between_2m_8m = np.logical_and(below_2m, above_8m)
proportions_mesh[between_2m_8m] = np.ones(3) / 3.0  # equal probabilities


# initialize the GMM with the one learned from PGI without mean information
clf_with_depth_info = copy.deepcopy(reg_nomean.objfcts[0].gmm)
clf_with_depth_info.order_clusters_GM_weight()
# Include the local proportions
clf_with_depth_info.weights_ = proportions_mesh

# Re-initiliaze a PGI
# Create the regularization with GMM information
reg_nomean_geo = regularization.SimplePGI(
    gmmref=clf_with_depth_info,
    mesh=mesh,
    wiresmap=wires,
    maplist=[idenMap],
    mref=m0,
    indActive=actcore,
)
# Weighting
reg_nomean_geo.alpha_s = alpha_s
reg_nomean_geo.alpha_x = alpha_x
reg_nomean_geo.alpha_y = alpha_y
reg_nomean_geo.mrefInSmooth = False

# Optimization
opt = optimization.ProjectedGNCG(
    maxIter=30, lower=-10, upper=10, maxIterLS=20, maxIterCG=50, tolCG=1e-4
)
opt.remember("xc")

# Set the inverse problem
invProb = inverse_problem.BaseInvProblem(dmis, reg_nomean_geo, opt)
invProb.beta = betavalue

# Inversion directives
betaIt = directives.PGI_BetaAlphaSchedule(
    verbose=True, coolingFactor=5.0, tolerance=0.05, progress=0.1
)
targets = directives.MultiTargetMisfits(
    chifact=1.0, TriggerSmall=True, TriggerTheta=False, verbose=True,
)
MrefInSmooth = directives.PGI_AddMrefInSmooth(verbose=True)
petrodir = directives.PGI_UpdateParameters(kappa=0, nu=1e8, zeta=1e8)

update_Jacobi = directives.UpdatePreconditioner()
updateSensW = directives.UpdateSensitivityWeights(threshold=1e-3, everyIter=False)

inv = inversion.BaseInversion(
    invProb,
    directiveList=[
        updateSensW,
        petrodir,
        targets,
        betaIt,
        MrefInSmooth,
        update_Jacobi,
    ],
)

# Run
#m_pgi_nomean_depth = inv.run(m0)


# Final Plot
############
# fig, axx = plt.subplots(4, 1, figsize=(15, 15), sharex=True)
# fig.subplots_adjust(wspace=0.1, hspace=0.3)
# clim = [mtrue.min(), mtrue.max()]
# cyl0 = getCylinderPoints(x0, z0, r0)
# cyl1 = getCylinderPoints(x1, z1, r1)
#
# title_list = [
#     "a) True model",
#     "b) PGI with full petrophysical info.",
#     "c) PGI with no mean info.",
#     "d) PGI with depth but no mean info",
# ]
#
# model_list = [mtrue[actcore], m_pgi, m_pgi_nomean, m_pgi_nomean_depth]
#
# for i, ax in enumerate(axx):
#     cyl0 = getCylinderPoints(x0, z0, r0)
#     cyl1 = getCylinderPoints(x1, z1, r1)
#     dat = meshCore.plotImage(
#         model_list[i], ax=ax, clim=clim, pcolorOpts={"cmap": "viridis"}
#     )
#     ax.set_title(title_list[i], fontsize=24, loc="left")
#     ax.set_aspect("equal")
#     ax.set_ylim([-15, 0])
#     ax.set_xlim([-15, 15])
#     ax.set_xlabel("", fontsize=22)
#     ax.set_ylabel("z (m)", fontsize=22)
#     ax.tick_params(labelsize=20)
#     ax.plot(cyl0[:, 0], cyl0[:, 1], "k--")
#     ax.plot(cyl1[:, 0], cyl1[:, 1], "k--")
#
#
# cbaxes_geo = fig.add_axes([0.8, 0.2, 0.02, 0.6])
# ticks = np.r_[10, 50, 100, 150, 200, 250]
# cbargeo = fig.colorbar(dat[0], cbaxes_geo, ticks=-np.log(ticks))
# cbargeo.ax.invert_yaxis()
# cbargeo.set_ticklabels(ticks)
# cbargeo.ax.tick_params(labelsize=20)
# cbargeo.set_label("Electrical resistivity ($\Omega$-m)", fontsize=24)
#
# plt.show()
