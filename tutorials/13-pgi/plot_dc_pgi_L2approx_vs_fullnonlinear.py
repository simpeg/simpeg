"""
DC: Petrophysically Guided Inversion (PGI): Approximation vs full-evaluation
============================================================================

Invert dc data with petrophysical information

Example inspired by: Thibaut Astic, Douglas W Oldenburg, A framework for petrophysically and geologically guided geophysical inversion using a dynamic Gaussian mixture model prior, Geophysical Journal International, Volume 219, Issue 3, December 2019, Pages 1989–2012, https://doi.org/10.1093/gji/ggz389

A DC resistivity profile is acquired over two cylinders.

We compare the use of the full nonlinear PGI regularizer and its Least-Squares approximation.

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
from pymatsolver import PardisoSolver
from scipy.stats import norm

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
    mesh, survey=survey, sigmaMap=mapping, Solver=PardisoSolver, nky=8
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
utils.order_clusters_GM_weight(clf)
## Set cluster means
clf.means_ = np.r_[-np.log(100.0), -np.log(50.0), -np.log(250.0)][:, np.newaxis]
## Set clusters variance
clf.covariances_ = np.array([[[0.001]], [[0.001]], [[0.001]],])
##Set clusters precision and Cholesky decomposition from variances
utils.compute_clusters_precision(clf)

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
MrefInSmooth = directives.AddMrefInSmooth(verbose=True)
## PGI update to the GMM, Smallness reference model and weights: 
## **This one is required when using the Least-Squares approximation of PGI 
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
m_pgi = inv.run(m0)


# PGI with Full Nonlinear Regularizer
#####################################

# Set the initial model to the true background mean
m0 = -np.log(100.0) * np.ones(mapping.nP)

# Create data misfit object
dmis = data_misfit.L2DataMisfit(data=dc_data, simulation=simulation)

# Create the regularization with GMM information
idenMap = maps.IdentityMap(nP=m0.shape[0])
wires = maps.Wires(("m", m0.shape[0]))
## Use the non-approximated Smallness and derivatives 
## The directives.PGI_UpdateParameters() is not necessary if the GMM stays fix.
reg_mean = regularization.SimplePGI(
    gmmref=clf, mesh=mesh, wiresmap=wires, maplist=[idenMap], mref=m0, indActive=actcore,
    approx_eval=False, approx_gradient=False,
)

# Regularization Weighting
betavalue = 25.0
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
MrefInSmooth = directives.AddMrefInSmooth(verbose=True)
## No directives.GaussianUpdateModel()
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
        targets,
        beta_alpha_iteration,
        MrefInSmooth,
        update_Jacobi,
    ],
)

# Run the inversion
m_pgi_full = inv.run(m0)


# Final Plot
############
fig, axx = plt.subplots(3, 1, figsize=(15, 15), sharex=True)
fig.subplots_adjust(wspace=0.1, hspace=0.3)
clim = [mtrue.min(), mtrue.max()]
cyl0 = getCylinderPoints(x0, z0, r0)
cyl1 = getCylinderPoints(x1, z1, r1)

title_list = [
    "a) True model",
    "b) PGI with Least-Squares Approximation",
    "c) PGI with full nonlinear Regularizer",
]

model_list = [mtrue[actcore], m_pgi, m_pgi_full]

for i, ax in enumerate(axx):
    cyl0 = getCylinderPoints(x0, z0, r0)
    cyl1 = getCylinderPoints(x1, z1, r1)
    dat = meshCore.plotImage(
        model_list[i], ax=ax, clim=clim, pcolorOpts={"cmap": "viridis"}
    )
    ax.set_title(title_list[i], fontsize=24, loc="left")
    ax.set_aspect("equal")
    ax.set_ylim([-15, 0])
    ax.set_xlim([-15, 15])
    ax.set_xlabel("", fontsize=22)
    ax.set_ylabel("z (m)", fontsize=22)
    ax.tick_params(labelsize=20)
    ax.plot(cyl0[:, 0], cyl0[:, 1], "k--")
    ax.plot(cyl1[:, 0], cyl1[:, 1], "k--")


cbaxes_geo = fig.add_axes([0.8, 0.2, 0.02, 0.6])
ticks = np.r_[10, 50, 100, 150, 200, 250]
cbargeo = fig.colorbar(dat[0], cbaxes_geo, ticks=-np.log(ticks))
cbargeo.ax.invert_yaxis()
cbargeo.set_ticklabels(ticks)
cbargeo.ax.tick_params(labelsize=20)
cbargeo.set_label("Electrical resistivity ($\Omega$-m)", fontsize=24)

plt.show()