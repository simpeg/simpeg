"""
Magnetic Amplitude inversion on a TreeMesh
==========================================

In this example, we demonstrate the use of magnetic amplitude
inversion on 3D TreeMesh for the inversion of Total Magnetic Intensity
(TMI) data affected by remanence. The original idea must be credited to
Shearer and Li (2005) @ CSM

First we invert the TMI for an equivalent source layer, from which we
recover 3-component magnetic data. This data is then transformed to amplitude

Secondly, we invert the non-linear inverse problem with
:class:'SimPEG.Directive.UpdateSensitivityWeights`. We also
uses the :class:'SimPEG.Regularization.Sparse' to apply sparsity
assumption in order to improve the recovery of a cube prism.

"""


from SimPEG import (Mesh, Directives, Maps,
                    InvProblem, Optimization, DataMisfit,
                    Inversion, Utils, Regularization)

import SimPEG.PF as PF
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import NearestNDInterpolator
from SimPEG.Utils import mkvc
###############################################################################
# Setup
# -----
#
# Define the survey and model parameters
#
# First we need to define the direction of the inducing field
# As a simple case, we pick a vertical inducing field of magnitude 50,000 nT.
#
#

# We will assume a vertical inducing field
H0 = (50000., 90., 0.)

# The magnetization is set along a different direction (induced + remanence)
M = np.array([45., 90.])

# Block with an effective susceptibility
chi_e = 0.05

# Create grid of points for topography
# Lets create a simple Gaussian topo and set the active cells
[xx, yy] = np.meshgrid(np.linspace(-200, 200, 50), np.linspace(-200, 200, 50))
b = 100
A = 50
zz = A*np.exp(-0.5*((xx/b)**2. + (yy/b)**2.))
topo = np.c_[Utils.mkvc(xx), Utils.mkvc(yy), Utils.mkvc(zz)]

# Create and array of observation points
xr = np.linspace(-100., 100., 20)
yr = np.linspace(-100., 100., 20)
X, Y = np.meshgrid(xr, yr)
Z = A*np.exp(-0.5*((X/b)**2. + (Y/b)**2.)) + 5

# Create a MAGsurvey
rxLoc = np.c_[Utils.mkvc(X.T), Utils.mkvc(Y.T), Utils.mkvc(Z.T)]
Rx = PF.BaseMag.RxObs(rxLoc)
srcField = PF.BaseMag.SrcField([Rx], param=H0)
survey = PF.BaseMag.LinearSurvey(srcField)

# Here how the topography looks with a quick interpolation, just a Gaussian...
tri = sp.spatial.Delaunay(topo)
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.plot_trisurf(
    topo[:, 0], topo[:, 1], topo[:, 2],
    triangles=tri.simplices, cmap=plt.cm.Spectral
)
ax.scatter3D(rxLoc[:, 0], rxLoc[:, 1], rxLoc[:, 2], c='k')
plt.show()

###############################################################################
# Inversion Mesh
# --------------
#
# Here, we create a TreeMesh with base cell size of 5 m. We reated a small
# utility function to center the mesh around points and to figure out the
# outer most dimension for adequate padding distance.
# The second stage allows to refine the mesh around points or surfaces
# (point assumed to follow some horiontal trend)
# The refinement process is repeated twice to allow for a finer level around
# the survey locations.
#

# Create a mesh
h = [5, 5, 5]
padDist = np.ones((3, 2)) * 100
nCpad = [2, 4, 2]

# Get extent of points
limx = np.r_[topo[:, 0].max(), topo[:, 0].min()]
limy = np.r_[topo[:, 1].max(), topo[:, 1].min()]
limz = np.r_[topo[:, 2].max(), topo[:, 2].min()]

# Get center of the mesh
midX = np.mean(limx)
midY = np.mean(limy)
midZ = np.mean(limz)

nCx = int(limx[0]-limx[1]) / h[0]
nCy = int(limy[0]-limy[1]) / h[1]
nCz = int(limz[0]-limz[1]+int(np.min(np.r_[nCx, nCy])/3)) / h[2]
# Figure out full extent required from input
extent = np.max(np.r_[nCx * h[0] + padDist[0, :].sum(),
                      nCy * h[1] + padDist[1, :].sum(),
                      nCz * h[2] + padDist[2, :].sum()])

maxLevel = int(np.log2(extent/h[0]))+1

# Number of cells at the small octree level
# For now equal in 3D
nCx, nCy, nCz = 2**(maxLevel), 2**(maxLevel), 2**(maxLevel)

# Define the mesh and origin
mesh = Mesh.TreeMesh([np.ones(nCx)*h[0],
                      np.ones(nCx)*h[1],
                      np.ones(nCx)*h[2]])

# Set origin
mesh.x0 = np.r_[-nCx*h[0]/2.+midX, -nCy*h[1]/2.+midY, -nCz*h[2]/2.+midZ]

# Refine the mesh around topography
# Get extent of points
F = NearestNDInterpolator(topo[:, :2], topo[:, 2])
zOffset = 0
# Cycle through the first 3 octree levels
for ii in range(3):

    dx = mesh.hx.min()*2**ii

    nCx = int((limx[0]-limx[1]) / dx)
    nCy = int((limy[0]-limy[1]) / dx)

    # Create a grid at the octree level in xy
    CCx, CCy = np.meshgrid(
        np.linspace(limx[1], limx[0], nCx),
        np.linspace(limy[1], limy[0], nCy)
    )

    z = F(mkvc(CCx), mkvc(CCy))

    # level means number of layers in current OcTree level
    for level in range(int(nCpad[ii])):

        mesh.insert_cells(
            np.c_[mkvc(CCx), mkvc(CCy), z-zOffset], np.ones_like(z)*maxLevel-ii,
            finalize=False
        )

        zOffset += dx

mesh.finalize()

# Define an active cells from topo
actv = Utils.surface2ind_topo(mesh, topo)
nC = int(actv.sum())

###########################################################################
# Forward modeling data
# ---------------------
#
# We can now generate TMI data
#

# Convert the inclination declination to vector in Cartesian
M_xyz = Utils.matutils.dipazm2xyz(np.ones(nC)*M[0], np.ones(nC)*M[1])

# Get the indicies of the magnetized block
ind = Utils.ModelBuilder.getIndicesBlock(
    np.r_[-20, -20, -10], np.r_[20, 20, 25],
    mesh.gridCC,
)[0]

# Assign magnetization value, inducing field strength will
# be applied in by the :class:`SimPEG.PF.Magnetics` problem
model = np.zeros(mesh.nC)
model[ind] = chi_e

# Remove air cells
model = model[actv]

# Create active map to go from reduce set to full
actvPlot = Maps.InjectActiveCells(mesh, actv, np.nan)

# Creat reduced identity map
idenMap = Maps.IdentityMap(nP=nC)

# Create the forward model operator
prob = PF.Magnetics.MagneticIntegral(
    mesh, M=M_xyz, chiMap=idenMap, actInd=actv
)

# Pair the survey and problem
survey.pair(prob)

# Compute some data and add some random noise
data = prob.fields(model)

# Split the data in components
nD = rxLoc.shape[0]

std = 5  # nT
data += np.random.randn(nD)*std
wd = np.ones(nD)*std

# Assigne data and uncertainties to the survey
survey.dobs = data
survey.std = wd


# Plot the model and data
plt.figure()
ax = plt.subplot(2, 1, 1)
im = Utils.PlotUtils.plot2Ddata(rxLoc, data, ax=ax)
plt.colorbar(im[0])
ax.set_title('Predicted data.')
plt.gca().set_aspect('equal', adjustable='box')

# Plot the vector model
ax = plt.subplot(2, 1, 2)
mesh.plotSlice(actvPlot*model, ax=ax, normal='Y', ind=66,
    pcolorOpts={"vmin":0., "vmax":0.01}
)
ax.set_xlim([-200, 200])
ax.set_ylim([-100, 75])
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.gca().set_aspect('equal', adjustable='box')

plt.show()


######################################################################
# Equivalent Source
# -----------------
#
# We first need to convert the TMI data into amplitude. We do this by
# for an effective susceptibility layer, from which we can forward component
# data
#

# Get the active cells for equivalent source is the top only
surf = Utils.modelutils.surfaceLayerIndex(mesh, topo)

# Get the layer of cells directyl below topo
#surf = Utils.actIndFull2layer(mesh, active)
nC = np.count_nonzero(surf)  # Number of active cells
print(nC)

# Create active map to go from reduce set to full
surfMap = Maps.InjectActiveCells(mesh, surf, np.nan)

# Create identity map
idenMap = Maps.IdentityMap(nP=nC)

# Create static map
prob = PF.Magnetics.MagneticIntegral(
        mesh, chiMap=idenMap, actInd=surf, 
        parallelized=False, equiSourceLayer = True)

prob.solverOpts['accuracyTol'] = 1e-4

# Pair the survey and problem
if survey.ispaired:
    survey.unpair()
survey.pair(prob)


# Create a regularization function, in this case l2l2
reg = Regularization.Sparse(mesh, indActive=surf, mapping=Maps.IdentityMap(nP=nC), scaledIRLS=False)
reg.mref = np.zeros(nC)

# Specify how the optimization will proceed, set susceptibility bounds to inf
opt = Optimization.ProjectedGNCG(maxIter=20, lower=-np.inf,
                                 upper=np.inf, maxIterLS=20,
                                 maxIterCG=20, tolCG=1e-3)

# Define misfit function (obs-calc)
dmis = DataMisfit.l2_DataMisfit(survey)
dmis.W = 1./survey.std

# Create the default L2 inverse problem from the above objects
invProb = InvProblem.BaseInvProblem(dmis, reg, opt)

# Specify how the initial beta is found
betaest = Directives.BetaEstimate_ByEig()

# Target misfit to stop the inversion,
# try to fit as much as possible of the signal, we don't want to lose anything
IRLS = Directives.Update_IRLS(f_min_change=1e-3, minGNiter=1,
                              beta_tol = 1e-1)
update_Jacobi = Directives.UpdatePreconditioner()
# Put all the parts together
inv = Inversion.BaseInversion(invProb,
                              directiveList=[betaest, IRLS, update_Jacobi])

# Run the equivalent source inversion
mstart = np.ones(nC)*1e-4
mrec = inv.run(mstart)

# Now that we have an equialent source layer, we can forward model alh three
# components of the field and add them up: |B| = ( Bx**2 + Bx**2 + Bx**2 )**0.5

# Won't store the sensitivity and output 'xyz' data.
prob.forwardOnly=True
prob.rx_type='xyz'
prob._G = None
prob.modelType = 'amplitude'
prob.model = mrec
pred = prob.fields(mrec)

bx = pred[:nD]
by = pred[nD:2*nD]
bz = pred[2*nD:]

bAmp = (bx**2. + by**2. + bz**2.)**0.5


# Plot the layer model and data
plt.figure()
ax = plt.subplot(2, 1, 1)
im = Utils.PlotUtils.plot2Ddata(rxLoc, bAmp, ax=ax)
plt.colorbar(im[0])
ax.set_title('Predicted data.')
plt.gca().set_aspect('equal', adjustable='box')

# Plot the vector model
ax = plt.subplot(2, 1, 2)
mesh.plotSlice(surfMap*mrec, ax=ax, normal='Y', ind=66,
    pcolorOpts={"vmin":0., "vmax":0.01}
)
ax.set_xlim([-200, 200])
ax.set_ylim([-100, 75])
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.gca().set_aspect('equal', adjustable='box')

plt.show()

#%% STEP 3: RUN AMPLITUDE INVERSION
# Now that we have |B| data, we can invert. This is a non-linear inversion,
# which requires some special care for the sensitivity weights (see Directives)

# Create active map to go from reduce space to full
actvMap = Maps.InjectActiveCells(mesh, actv, -100)
nC = len(actv)

# Create identity map
idenMap = Maps.IdentityMap(nP=nC)

mstart= np.ones(len(actv))*1e-4

# Create the forward model operator
prob = PF.Magnetics.MagneticIntegral(mesh, chiMap=idenMap,
                                     actInd=actv, modelType='amplitude',
                                    rxType='xyz')
prob.model = mstart
# Change the survey to xyz components
survey_xyz = PF.BaseMag.LinearSurvey(survey.srcField)
survey_xyz.srcField.rxList[0].rxType = 'xyz'

# Pair the survey and problem
survey_xyz.pair(prob)
# Create a regularization function, in this case l2l2
wr = np.sum(prob.G**2., axis=0)**0.5
wr = (wr/np.max(wr))
# Re-set the observations to |B|
survey_xyz.dobs = damp

# Create a regularization function, in this case l2l2

# Create a regularization function, in this case l2l2
wr = np.sum(prob.G**2., axis=0)**0.5
wr = (wr/np.max(wr))

# Create a sparse regularization
reg = Regularization.Sparse(mesh, indActive=actv, mapping=idenMap)
reg.norms = np.c_[1,2,2,2]
reg.mref = np.zeros(nC)
reg.cell_weights= wr
# Data misfit function
dmis = DataMisfit.l2_DataMisfit(survey_xyz)
dmis.W = 1/survey.std

# Add directives to the inversion
opt = Optimization.ProjectedGNCG(maxIter=10, lower=0., upper=1.,
                                 maxIterLS=20, maxIterCG=20,
                                 tolCG=1e-3)

invProb = InvProblem.BaseInvProblem(dmis, reg, opt)

# Here is the list of directives
betaest = Directives.BetaEstimate_ByEig()

# Specify the sparse norms
IRLS = Directives.Update_IRLS(f_min_change=1e-3,
                              minGNiter=3, coolingRate=1)

# Special directive specific to the mag amplitude problem. The sensitivity
# weights are update between each iteration.
update_SensWeight = Directives.UpdateSensitivityWeights(everyIter=True)
update_Jacobi = Directives.UpdatePreconditioner(epsilon=1e-3)

# Put all together
inv = Inversion.BaseInversion(invProb,
                                   directiveList=[betaest, IRLS, update_SensWeight, update_Jacobi,])

# Invert
mrec_Amp = inv.run(mstart)

#############################################################
# Final Plot
# ----------
#
# Let's compare the smooth and compact model
#
#
#

plt.figure(figsize=(8, 8))
ax = plt.subplot(2, 1, 1)
plotVectorSectionsOctree(
    mesh, mrec_MVIC.reshape((nC, 3), order="F"),
    axs=ax, normal='Y', ind=65, actvMap=actvPlot,
    scale=0.05, vmin=0., vmax=0.005)

ax.set_xlim([-200, 200])
ax.set_ylim([-100, 75])
ax.set_title('Smooth model (Cartesian)')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.gca().set_aspect('equal', adjustable='box')

ax = plt.subplot(2, 1, 2)
vec_xyz = Utils.matutils.spherical2xyz(
    invProb.model.reshape((nC, 3), order='F')).reshape((nC, 3), order='F')

plotVectorSectionsOctree(
    mesh, vec_xyz, axs=ax, normal='Y', ind=65,
    actvMap=actvPlot, scale=0.4, vmin=0., vmax=0.01
)
ax.set_xlim([-200, 200])
ax.set_ylim([-100, 75])
ax.set_title('Sparse model (Spherical)')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.gca().set_aspect('equal', adjustable='box')

plt.show()

# Plot the final predicted data and the residual
plt.figure()
ax = plt.subplot(1, 2, 1)
Utils.PlotUtils.plot2Ddata(xyzLoc, invProb.dpred, ax=ax)
ax.set_title('Predicted data.')
plt.gca().set_aspect('equal', adjustable='box')

ax = plt.subplot(1, 2, 2)
Utils.PlotUtils.plot2Ddata(xyzLoc, data-invProb.dpred, ax=ax)
ax.set_title('Data residual.')
plt.gca().set_aspect('equal', adjustable='box')
