"""
2D DC inversion of Dipole Dipole array
======================================

This is an example for 2D DC Inversion. The model consists of 2 cylinders,
one conductive, the other one resistive compared to the background.

We restrain the inversion to the Core Mesh through the use an Active Cells
mapping that we combine with an exponetial mapping to invert
in log conductivity space. Here mapping,  :math:`\\mathcal{M}`,
indicates transformation of our model to a different space:

.. math::
    \\sigma = \\mathcal{M}(\\mathbf{m})

Following example will show you how user can implement a 2D DC inversion.
"""

from SimPEG import (
    Mesh, Maps, Utils,
    DataMisfit, Regularization, Optimization,
    InvProblem, Directives, Inversion
)
from SimPEG.EM.Static import DC, Utils as DCUtils
import numpy as np
import matplotlib.pyplot as plt
try:
    from pymatsolver import Pardiso as Solver
except ImportError:
    from SimPEG import SolverLU as Solver

# Reproducible science
np.random.seed(12345)

# 2D Mesh
#########

# Cells size
csx, csz = 0.25, 0.25
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
############################

# Spheres parameters
x0, z0, r0 = -6., -5., 3.
x1, z1, r1 = 6., -5., 3.

ln_sigback = -5.
ln_sigc = -3.
ln_sigr = -6.

mtrue = ln_sigback * np.ones(mesh.nC)

csph = (np.sqrt((mesh.gridCC[:, 1] - z0) **
                2. + (mesh.gridCC[:, 0] - x0)**2.)) < r0
mtrue[csph] = ln_sigc * np.ones_like(mtrue[csph])

# Define the sphere limit
rsph = (np.sqrt((mesh.gridCC[:, 1] - z1) **
                2. + (mesh.gridCC[:, 0] - x1)**2.)) < r1
mtrue[rsph] = ln_sigr * np.ones_like(mtrue[rsph])

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
zmin, zmax = mesh.vectorCCy[-1], mesh.vectorCCy[-1]
endl = np.array([[xmin, ymin, zmin], [xmax, ymax, zmax]])
survey = DCUtils.gen_DCIPsurvey(endl, "dipole-dipole", dim=mesh.dim,
                                a=1, b=1, n=10, d2flag='2D')

# Setup Problem with exponential mapping and Active cells only in the core mesh
expmap = Maps.ExpMap(mesh)
mapactive = Maps.InjectActiveCells(mesh=mesh, indActive=actind,
                                   valInactive=-5.)
mapping = expmap * mapactive
problem = DC.Problem3D_CC(mesh, sigmaMap=mapping)
problem.pair(survey)
problem.Solver = Solver

survey.dpred(mtrue[actind])
survey.makeSyntheticData(mtrue[actind], std=0.05, force=True)

# Tikhonov Inversion
####################

m0 = np.median(ln_sigback) * np.ones(mapping.nP)
dmis = DataMisfit.l2_DataMisfit(survey)
regT = Regularization.Simple(mesh, indActive=actind)

# Personal preference for this solver with a Jacobi preconditioner
opt = Optimization.ProjectedGNCG(maxIter=20, lower=-10, upper=10,
                                 maxIterLS=20, maxIterCG=30, tolCG=1e-4)

opt.remember('xc')
invProb = InvProblem.BaseInvProblem(dmis, regT, opt)

beta = Directives.BetaEstimate_ByEig(beta0_ratio=1.)
Target = Directives.TargetMisfit()
betaSched = Directives.BetaSchedule(coolingFactor=5., coolingRate=2)
updateSensW = Directives.UpdateSensitivityWeights(threshold=1e-3)
update_Jacobi = Directives.UpdatePreconditioner()

inv = Inversion.BaseInversion(invProb, directiveList=[beta, Target,
                                                      betaSched, updateSensW,
                                                      update_Jacobi])

minv = inv.run(m0)

# Final Plot
############

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax = Utils.mkvc(ax)

cyl0v = getCylinderPoints(x0, z0, r0)
cyl1v = getCylinderPoints(x1, z1, r1)

clim = [(mtrue[actind]).min(), (mtrue[actind]).max()]

dat = meshCore.plotImage(((mtrue[actind])), ax=ax[0], clim=clim)
ax[0].set_title('Ground Truth')
ax[0].set_aspect('equal')

meshCore.plotImage((minv), ax=ax[1], clim=clim)
ax[1].set_aspect('equal')
ax[1].set_title('Inverted Model')

for i in range(2):
    ax[i].plot(cyl0v[:, 0], cyl0v[:, 1], 'k--')
    ax[i].plot(cyl1v[:, 0], cyl1v[:, 1], 'k--')

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
cb = plt.colorbar(dat[0], ax=cbar_ax)
cb.set_label('ln conductivity')

cbar_ax.axis('off')

plt.show()
