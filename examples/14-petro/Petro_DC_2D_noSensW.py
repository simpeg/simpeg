from SimPEG import (
    Mesh,  Maps,  Utils, DataMisfit,  Regularization,
    Optimization, InvProblem,  Directives,  Inversion
)
from SimPEG.EM.Static import DC, Utils as DCUtils
import numpy as np
import matplotlib.pyplot as plt
from pymatsolver import PardisoSolver
from scipy.stats import norm
from sklearn.mixture import GaussianMixture
import copy

# Reproducible science!
np.random.seed(518936)

# 2D Mesh
#########
csx, csy, csz = 0.25, 0.25, 0.25
# Number of core cells in each direction
ncx, ncz = 123, 41
# Number of padding cells to add in each direction
npad = 12
# Vectors of cell lengthts in each direction
hx = [(csx, npad, -1.5), (csx, ncx), (csx, npad, 1.5)]
hz = [(csz, npad, -1.5), (csz, ncz)]
# Create mesh
mesh = Mesh.TensorMesh([hx, hz], x0="CN")
mesh.x0[1] = mesh.x0[1] + csz / 2.

# 2-cylinders Model Creation
##########################
# Spheres parameters
x0, z0, r0 = -6., -5., 3.
x1, z1, r1 = 6., -5., 3.

ln_sigback = -5.
ln_sigc = -3.
ln_sigr = -6.

# Add some variability to the physical property model
noisemean = 0.
noisevar = 0.05
ln_over = -2.

mtrue = ln_sigback * np.ones(mesh.nC) + norm(noisemean, noisevar).rvs(mesh.nC)
mprim = copy.deepcopy(mtrue)

csph = (np.sqrt((mesh.gridCC[:, 1] - z0) **
                2. + (mesh.gridCC[:, 0] - x0)**2.)) < r0
mtrue[csph] = ln_sigc * np.ones_like(mtrue[csph]) + norm(
    noisemean, noisevar).rvs(np.prod((mtrue[csph]).shape))

# Define the sphere limit
rsph = (np.sqrt((mesh.gridCC[:, 1] - z1) **
                2. + (mesh.gridCC[:, 0] - x1)**2.)) < r1
mtrue[rsph] = ln_sigr * np.ones_like(mtrue[rsph]) + norm(
    noisemean, noisevar).rvs(np.prod((mtrue[rsph]).shape))

mtrue = Utils.mkvc(mtrue)
xmin, xmax = -15., 15
ymin, ymax = -15., 0.
xyzlim = np.r_[[[xmin, xmax], [ymin, ymax]]]
actind, meshCore = Utils.meshutils.ExtractCoreMesh(xyzlim, mesh)


# Function to plot cylinder border
def getCylinderPoints(xc, zc, r):
    xLocOrig1 = np.arange(-r, r + r / 10., r / 10.)
    xLocOrig2 = np.arange(r, -r - r / 10., -r / 10.)
    # Top half of cylinder
    zLoc1 = np.sqrt(-xLocOrig1**2. + r**2.) + zc
    # Bottom half of cylinder
    zLoc2 = -np.sqrt(-xLocOrig2**2. + r**2.) + zc
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


# Setup a Dipole-Dipole Survey
xmin, xmax = -15., 15.
ymin, ymax = 0., 0.
zmin, zmax = 0, 0
endl = np.array([[xmin, ymin, zmin], [xmax, ymax, zmax]])
survey = DCUtils.gen_DCIPsurvey(
    endl, survey_type="dipole-dipole", dim=mesh.dim,
    a=1, b=1, n=10, d2flag='2D'
)

# Setup Problem with exponential mapping and Active cells only in the core mesh
expmap = Maps.ExpMap(mesh)
mapactive = Maps.InjectActiveCells(mesh=mesh, indActive=actind,
                                   valInactive=-5.)
mapping = expmap * mapactive
problem = DC.Problem3D_CC(mesh, sigmaMap=mapping)
problem.pair(survey)
problem.Solver = PardisoSolver

survey.dpred(mtrue[actind])
survey.makeSyntheticData(mtrue[actind], std=0.05, force=True)


#####################
# Tikhonov Inversion#
#####################
m0 = np.median(ln_sigback) * np.ones(mapping.nP)
dmis = DataMisfit.l2_DataMisfit(survey)
regT = Regularization.Simple(
    mesh,
    alpha_s=1.,
    alpha_x=1.,
    alpha_y=1., indActive=actind
)
opt = Optimization.InexactGaussNewton(maxIter=20, tolX=1e-6)
opt.remember('xc')
invProb = InvProblem.BaseInvProblem(dmis, regT, opt)

beta = Directives.BetaEstimate_ByEig(beta0_ratio=1e-3)
betaSched = Directives.BetaSchedule(coolingFactor=5., coolingRate=3)
target = Directives.TargetMisfit()

inv = Inversion.BaseInversion(invProb, directiveList=[beta, betaSched, target])

mnormal = inv.run(m0)


#########################################
# Petrophysically constrained inversion #
#########################################

# fit a Gaussian Mixture Model with n components
# on the true model to simulate the laboratory
# petrophysical measurements
n = 3
clf = GaussianMixture(n_components=n, covariance_type='tied', reg_covar=5e-3)
clf.fit(mtrue[actind].reshape(-1, 1))
Utils.order_clusters_GM_weight(clf)
print(clf.covariances_)

regmesh = mesh
m0 = np.median(clf.means_) * np.ones(mapping.nP)
dmis = DataMisfit.l2_DataMisfit(survey)
Utils.order_clusters_GM_weight(clf)

idenMap = Maps.IdentityMap(nP=m0.shape[0])
wires = Maps.Wires(('m', m0.shape[0]))
reg = Regularization.SimplePetroRegularization(
    GMmref=clf, mesh=mesh,
    wiresmap=wires,
    maplist=[idenMap],
    alpha_s=1.,
    alpha_x=1.,
    alpha_y=1.,
    indActive=actind
)
reg.mrefInSmooth = False
reg.approx_gradient = True
gamma_petro = np.r_[1., 1., 1.]
reg.gamma = gamma_petro

opt = Optimization.ProjectedGNCG(
    maxIter=20, lower=-10, upper=10,
    maxIterLS=20, maxIterCG=50, tolCG=1e-4
)
opt.remember('xc')
invProb = InvProblem.BaseInvProblem(dmis, reg, opt)
Alphas = Directives.AlphasSmoothEstimate_ByEig(
    alpha0_ratio=1., ninit=10, verbose=True
)
beta = Directives.BetaEstimate_ByEig(beta0_ratio=1., ninit=10)
betaIt = Directives.PetroBetaReWeighting(
    verbose=True, rateCooling=5.,
    rateWarming=1., progress=0.1
)
targets = Directives.PetroTargetMisfit(
    TriggerSmall=True,
    TriggerTheta=False,
    verbose=True,
)
MrefInSmooth = Directives.AddMrefInSmooth(verbose=True, wait_till_stable=True)
petrodir = Directives.GaussianMixtureUpdateModel()

inv = Inversion.BaseInversion(
    invProb, directiveList=[Alphas, beta, petrodir, targets, betaIt, MrefInSmooth])

mcluster = inv.run(m0)

print(meshCore.nC / 2)
print(invProb.reg.objfcts[0](mcluster))

# Final Plot
fig, ax = plt.subplots(2, 2, figsize=(12, 6))
ax = Utils.mkvc(ax)

cyl0 = getCylinderPoints(x0, z0, r0)
cyl1 = getCylinderPoints(x1, z1, r1)

clim = [mtrue.min(), mtrue.max()]

dat = meshCore.plotImage(mtrue[actind], ax=ax[0], clim=clim)
ax[0].set_title('Ground Truth')
ax[0].set_aspect('equal')

meshCore.plotImage(mnormal, ax=ax[1], clim=clim)
ax[1].set_aspect('equal')
ax[1].set_title('Tikhonov')

meshCore.plotImage(mcluster, ax=ax[2], clim=clim)
ax[2].set_title('Petrophysically constrained')
ax[2].set_aspect('equal')

meshCore.plotImage(invProb.reg.mref, ax=ax[3], clim=clim)
ax[3].set_title('Learned Reference model')
ax[3].set_aspect('equal')

for i in range(4):
    ax[i].plot(cyl0[:, 0], cyl0[:, 1], 'k--')
    ax[i].plot(cyl1[:, 0], cyl1[:, 1], 'k--')

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
cb = plt.colorbar(dat[0], ax=cbar_ax)
cb.set_label('ln conductivity')

cbar_ax.axis('off')


plt.show()
