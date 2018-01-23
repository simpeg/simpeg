"""
3D DC inversion of Dipole Dipole array
======================================

This is an example for 3D DC Inversion. The model consists of 2 spheres,
one conductive, the other one resistive compared to the background.

We restrain the inversion to the Core Mesh through the use an Active Cells
mapping that we combine with an exponetial mapping to invert
in log conductivity space. Here mapping,  :math:`\\mathcal{M}`,
indicates transformation of our model to a different space:

.. math::
    \\sigma = \\mathcal{M}(\\mathbf{m})

Following example will show you how user can implement a 3D DC inversion.
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

np.random.seed(12345)

# 3D Mesh
#########

# Cell sizes
csx, csy, csz = 1., 1., 0.5
# Number of core cells in each direction
ncx, ncy, ncz = 41, 31, 21
# Number of padding cells to add in each direction
npad = 7
# Vectors of cell lengths in each direction with padding
hx = [(csx, npad, -1.5), (csx, ncx), (csx, npad, 1.5)]
hy = [(csy, npad, -1.5), (csy, ncy), (csy, npad, 1.5)]
hz = [(csz, npad, -1.5), (csz, ncz)]
# Create mesh and center it
mesh = Mesh.TensorMesh([hx, hy, hz], x0="CCN")

# 2-spheres Model Creation
##########################

# Spheres parameters
x0, y0, z0, r0 = -6., 0., -3.5, 3.
x1, y1, z1, r1 = 6., 0., -3.5, 3.

# ln conductivity
ln_sigback = -5.
ln_sigc = -3.
ln_sigr = -6.

# Define model
# Background
mtrue = ln_sigback * np.ones(mesh.nC)

# Conductive sphere
csph = (np.sqrt((mesh.gridCC[:, 0] - x0)**2. + (mesh.gridCC[:, 1] - y0)**2. +
                (mesh.gridCC[:, 2] - z0)**2.)) < r0
mtrue[csph] = ln_sigc * np.ones_like(mtrue[csph])

# Resistive Sphere
rsph = (np.sqrt((mesh.gridCC[:, 0] - x1)**2. + (mesh.gridCC[:, 1] - y1)**2. +
                (mesh.gridCC[:, 2] - z1)**2.)) < r1
mtrue[rsph] = ln_sigr * np.ones_like(mtrue[rsph])

# Extract Core Mesh
xmin, xmax = -20., 20.
ymin, ymax = -15., 15.
zmin, zmax = -10., 0.
xyzlim = np.r_[[[xmin, xmax], [ymin, ymax], [zmin, zmax]]]
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


# Setup a synthetic Dipole-Dipole Survey
# Line 1
xmin, xmax = -15., 15.
ymin, ymax = 0., 0.
zmin, zmax = 0, 0
endl = np.array([[xmin, ymin, zmin], [xmax, ymax, zmax]])
survey1 = DCUtils.gen_DCIPsurvey(endl, "dipole-dipole", dim=mesh.dim,
                                 a=3, b=3, n=8)

# Line 2
xmin, xmax = -15., 15.
ymin, ymax = 5., 5.
zmin, zmax = 0, 0
endl = np.array([[xmin, ymin, zmin], [xmax, ymax, zmax]])
survey2 = DCUtils.gen_DCIPsurvey(endl, "dipole-dipole", dim=mesh.dim,
                                 a=3, b=3, n=8)

# Line 3
xmin, xmax = -15., 15.
ymin, ymax = -5., -5.
zmin, zmax = 0, 0
endl = np.array([[xmin, ymin, zmin], [xmax, ymax, zmax]])
survey3 = DCUtils.gen_DCIPsurvey(endl, "dipole-dipole", dim=mesh.dim,
                                 a=3, b=3, n=8)

# Concatenate lines
survey = DC.Survey(survey1.srcList + survey2.srcList + survey3.srcList)

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

# Initial Model
m0 = np.median(ln_sigback) * np.ones(mapping.nP)
# Data Misfit
dmis = DataMisfit.l2_DataMisfit(survey)
# Regularization
regT = Regularization.Simple(mesh, indActive=actind, alpha_s=1e-6,
                             alpha_x=1., alpha_y=1., alpha_z=1.)

# Optimization Scheme
opt = Optimization.InexactGaussNewton(maxIter=10)

# Form the problem
opt.remember('xc')
invProb = InvProblem.BaseInvProblem(dmis, regT, opt)

# Directives for Inversions
beta = Directives.BetaEstimate_ByEig(beta0_ratio=1e+1)
Target = Directives.TargetMisfit()
betaSched = Directives.BetaSchedule(coolingFactor=5., coolingRate=2)

inv = Inversion.BaseInversion(invProb, directiveList=[beta, Target,
                                                      betaSched])
# Run Inversion
minv = inv.run(m0)

# Final Plot
############

fig, ax = plt.subplots(2, 2, figsize=(12, 6))
ax = Utils.mkvc(ax)

cyl0v = getCylinderPoints(x0, z0, r0)
cyl1v = getCylinderPoints(x1, z1, r1)

cyl0h = getCylinderPoints(x0, y0, r0)
cyl1h = getCylinderPoints(x1, y1, r1)

clim = [(mtrue[actind]).min(), (mtrue[actind]).max()]

dat = meshCore.plotSlice(((mtrue[actind])), ax=ax[0], normal='Y', clim=clim,
                         ind=int(ncy / 2))
ax[0].set_title('Ground Truth, Vertical')
ax[0].set_aspect('equal')

meshCore.plotSlice((minv), ax=ax[1], normal='Y', clim=clim, ind=int(ncy / 2))
ax[1].set_aspect('equal')
ax[1].set_title('Inverted Model, Vertical')

meshCore.plotSlice(((mtrue[actind])), ax=ax[2], normal='Z', clim=clim,
                   ind=int(ncz / 2))
ax[2].set_title('Ground Truth, Horizontal')
ax[2].set_aspect('equal')

meshCore.plotSlice((minv), ax=ax[3], normal='Z', clim=clim, ind=int(ncz / 2))
ax[3].set_title('Inverted Model, Horizontal')
ax[3].set_aspect('equal')

for i in range(2):
    ax[i].plot(cyl0v[:, 0], cyl0v[:, 1], 'k--')
    ax[i].plot(cyl1v[:, 0], cyl1v[:, 1], 'k--')
for i in range(2, 4):
    ax[i].plot(cyl1h[:, 0], cyl1h[:, 1], 'k--')
    ax[i].plot(cyl0h[:, 0], cyl0h[:, 1], 'k--')

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
cb = plt.colorbar(dat[0], ax=cbar_ax)
cb.set_label('ln conductivity')

cbar_ax.axis('off')

plt.show()
