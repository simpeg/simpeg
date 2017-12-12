from SimPEG import DC
from SimPEG import (Mesh, Maps, Utils, DataMisfit, Regularization,
                    Optimization, Inversion, InvProblem, Directives)
from pymatsolver import PardisoSolver
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import colors
import numpy as np
from pylab import hist
try:
    from pymatsolver import Pardiso as Solver
except ImportError:
    from SimPEG import SolverLU as Solver

showIt = True

# Initiate I/O class for DC
IO = DC.IO()
# Obtain ABMN locations
A, B, M, N = IO.genLocs_2D('dipole-dipole', 0., 200., 10., 10)

# Generate DC survey object
survey = IO.fromABMN_to_survey(
    A, B, M, N, 'dipole-dipole', dataType='volt'
)

# Obtain 2D TensorMesh
mesh, actind = IO.setMesh()

# Build a conductivity model
inds = Utils.ModelBuilder.getIndicesSphere(np.r_[100., -25.], 15., mesh.gridCC)
sigma = np.ones(mesh.nC)*1e-2
sigma[inds] = 0.1
rho = 1./sigma

# Show the true conductivity model
if showIt:
    fig = plt.figure(figsize = (12, 3))
    ax = plt.subplot(111)
    mesh.plotImage(1./sigma, grid=True, ax=ax, gridOpts={'alpha':0.2}, pcolorOpts={"cmap":"jet"})
    plt.plot(IO.uniqElecLocs[:,0], IO.uniqElecLocs[:,1], 'k.')
    plt.show()

# Use Exponential Map: m = log(rho)
mapping = Maps.ExpMap(mesh)
# Generate mtrue
mtrue = np.ones(mesh.nC)*np.log(rho)

# Generate 2.5D DC problem
# "N" means potential is defined at nodes
prb = DC.Problem2D_N(
    mesh, rhoMap=mapping, storeJ=True,
    Solver = PardisoSolver
)
# Pair problem with survey
try:
    prb.pair(survey)
except:
    survey.unpair()
    prb.pair(survey)

# Make synthetic DC data with 5% Gaussian noise
dtrue = survey.makeSyntheticData(mtrue, std=0.05)

# Show apparent resisitivty pseudo-section
if showIt:
    IO.plotPseudoSection(dobs=survey.dobs)

# Show apparent resisitivty histogram
if showIt:
    fig = plt.figure()
    out = hist(survey.dobs/IO.G, bins=20)
    plt.show()

# Set initial model based upon histogram
m0 = np.ones(mesh.nC)*np.log(110.)

# Set uncertainty
eps = 10**-3.2 # floor
std = 0.05     # percentage
dmisfit = DataMisfit.l2_DataMisfit(survey)
uncert = abs(survey.dobs) * std + eps
dmisfit.W = 1./uncert

# Map for a regularization
regmap = Maps.IdentityMap(nP=int(actind.sum()))

# Related to inversion
reg = Regularization.Simple(mesh, indActive=actind, mapping=regmap)
opt = Optimization.InexactGaussNewton(maxIter=15)
invProb = InvProblem.BaseInvProblem(dmisfit, reg, opt)
beta = Directives.BetaSchedule(coolingFactor=5, coolingRate=2)
betaest = Directives.BetaEstimate_ByEig(beta0_ratio=1e0)
target = Directives.TargetMisfit()
inv = Inversion.BaseInversion(invProb, directiveList=[beta, betaest, target])
prb.counter = opt.counter = Utils.Counter()
opt.LSshorten = 0.5
opt.remember('xc')

# Run inversion
mopt = inv.run(m0)

# Convert obtained inversion model to resistivity
# rho = M(m), where M(.) is a mapping

rho_est = mapping*mopt
rho_est[~actind] = np.nan

# show recovered conductivity
if showIt:
    vmin, vmax = rho.min(), rho.max()
    fig, ax = plt.subplots(figsize=(20, 3))
    out = mesh.plotImage(rho_est, clim=(vmin, vmax), pcolorOpts={"cmap":"jet", "norm":colors.LogNorm()}, ax=ax)
    ax.set_xlim(IO.grids[:,0].min(), IO.grids[:,0].max())
    ax.set_ylim(-IO.grids[:,1].max(), IO.grids[:,1].min())
    cb = plt.colorbar(out[0])
    cb.set_label("Resistivity ($\Omega$m)")
    ax.set_xlabel("Northing (m)")
    ax.set_ylabel("Elevation (m)")
    plt.show()

# show predicted data
if showIt:
    IO.plotPseudoSection(dobs=invProb.dpred)
