"""
Magnetic inversion on a TreeMesh
================================

In this example, we demonstrate the use of a Magnetic Vector Inverison
on 3D TreeMesh for the inversion of magnetic affected by remanence.
The mesh is auto-generate based
on the position of the observation locations and topography.

We invert the data twice, first for a smooth starting model using the
Cartesian coordinate system, and second for a compact model using
the Spherical formulation.

The inverse problem uses the :class:'SimPEG.Regularization.Sparse'
that

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

# sphinx_gallery_thumbnail_number = 3

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
sp.random.seed(1)
# We will assume a vertical inducing field
H0 = (50000., 90., 0.)

# The magnetization is set along a different direction (induced + remanence)
M = np.array([45., 90.])

# Create grid of points for topography
# Lets create a simple Gaussian topo and set the active cells
[xx, yy] = np.meshgrid(np.linspace(-200, 200, 50), np.linspace(-200, 200, 50))
b = 100
A = 50
zz = A*np.exp(-0.5*((xx/b)**2. + (yy/b)**2.))

# We would usually load a topofile
topo = np.c_[Utils.mkvc(xx), Utils.mkvc(yy), Utils.mkvc(zz)]

# Create and array of observation points
xr = np.linspace(-100., 100., 20)
yr = np.linspace(-100., 100., 20)
X, Y = np.meshgrid(xr, yr)
Z = A*np.exp(-0.5*((X/b)**2. + (Y/b)**2.)) + 5

# Create a MAGsurvey
xyzLoc = np.c_[Utils.mkvc(X.T), Utils.mkvc(Y.T), Utils.mkvc(Z.T)]
rxLoc = PF.BaseMag.RxObs(xyzLoc)
srcField = PF.BaseMag.SrcField([rxLoc], param=H0)
survey = PF.BaseMag.LinearSurvey(srcField)

# Here how the topography looks with a quick interpolation, just a Gaussian...
tri = sp.spatial.Delaunay(topo)
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.plot_trisurf(
    topo[:, 0], topo[:, 1], topo[:, 2],
    triangles=tri.simplices, cmap=plt.cm.Spectral
)
ax.scatter3D(xyzLoc[:, 0], xyzLoc[:, 1], xyzLoc[:, 2], c='k')
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
# nCy = 2**(int(np.log2(extent/h[1]))+1)
# nCz = 2**(int(np.log2(extent/h[2]))+1)

# Define the mesh and origin
# For now cubic cells
mesh = Mesh.TreeMesh([np.ones(nCx)*h[0],
                      np.ones(nCx)*h[1],
                      np.ones(nCx)*h[2]])

# Set origin
mesh.x0 = np.r_[-nCx*h[0]/2.+midX, -nCy*h[1]/2.+midY, -nCz*h[2]/2.+midZ]

# mesh = Utils.modelutils.meshBuilder(topo, h, padDist,
#                                     meshType='TREE',
#                                     verticalAlignment='center')


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
# A simple function to plot vectors in TreeMesh
#
# Should eventually end up on discretize
#


def plotVectorSectionsOctree(
    mesh, m, normal='X', ind=0, vmin=None, vmax=None,
    subFact=2, scale=1., xlim=None, ylim=None, vec='k',
    title=None, axs=None, actvMap=None, contours=None, fill=True,
        orientation='vertical', cmap='pink_r'
):

    """
    Plot section through a 3D tensor model
    """
    # plot recovered model
    normalInd = {'X': 0, 'Y': 1, 'Z': 2}[normal]
    antiNormalInd = {'X': [1, 2], 'Y': [0, 2], 'Z': [0, 1]}[normal]

    h2d = (mesh.h[antiNormalInd[0]], mesh.h[antiNormalInd[1]])
    x2d = (mesh.x0[antiNormalInd[0]], mesh.x0[antiNormalInd[1]])

    #: Size of the sliced dimension
    szSliceDim = len(mesh.h[normalInd])
    if ind is None:
        ind = int(szSliceDim//2)

    cc_tensor = [None, None, None]
    for i in range(3):
        cc_tensor[i] = np.cumsum(np.r_[mesh.x0[i], mesh.h[i]])
        cc_tensor[i] = (cc_tensor[i][1:] + cc_tensor[i][:-1])*0.5
    slice_loc = cc_tensor[normalInd][ind]

    # Create a temporary TreeMesh with the slice through
    temp_mesh = Mesh.TreeMesh(h2d, x2d)
    level_diff = mesh.max_level - temp_mesh.max_level

    XS = [None, None, None]
    XS[antiNormalInd[0]], XS[antiNormalInd[1]] = np.meshgrid(
        cc_tensor[antiNormalInd[0]], cc_tensor[antiNormalInd[1]]
    )
    XS[normalInd] = np.ones_like(XS[antiNormalInd[0]])*slice_loc
    loc_grid = np.c_[XS[0].reshape(-1), XS[1].reshape(-1), XS[2].reshape(-1)]
    inds = np.unique(mesh._get_containing_cell_indexes(loc_grid))

    grid2d = mesh.gridCC[inds][:, antiNormalInd]
    levels = mesh._cell_levels_by_indexes(inds) - level_diff
    temp_mesh.insert_cells(grid2d, levels)
    tm_gridboost = np.empty((temp_mesh.nC, 3))
    tm_gridboost[:, antiNormalInd] = temp_mesh.gridCC
    tm_gridboost[:, normalInd] = slice_loc

    # Interpolate values to mesh.gridCC if not 'CC'
    mx = (actvMap*m[:, 0])
    my = (actvMap*m[:, 1])
    mz = (actvMap*m[:, 2])

    m = np.c_[mx, my, mz]

    # Interpolate values from mesh.gridCC to grid2d
    ind_3d_to_2d = mesh._get_containing_cell_indexes(tm_gridboost)
    v2d = m[ind_3d_to_2d, :]
    amp = np.sum(v2d**2., axis=1)**0.5

    if axs is None:
        axs = plt.subplot(111)

    if fill:
        temp_mesh.plotImage(amp, ax=axs, clim=[vmin, vmax], grid=True)

    axs.quiver(temp_mesh.gridCC[:, 0],
               temp_mesh.gridCC[:, 1],
               v2d[:, antiNormalInd[0]],
               v2d[:, antiNormalInd[1]],
               pivot='mid',
               scale_units="inches", scale=scale, linewidths=(1,),
               edgecolors=(vec),
               headaxislength=0.1, headwidth=10, headlength=30)

###########################################################################
# Forward modeling data
# ---------------------
#
# We can now create a magnetization model and generate data
# Lets start with a block below topography
#

model = np.zeros((mesh.nC, 3))

# Convert the inclination declination to vector in Cartesian
M_xyz = Utils.matutils.dip_azimuth2cartesian(M[0], M[1])

# Get the indicies of the magnetized block
ind = Utils.ModelBuilder.getIndicesBlock(
    np.r_[-20, -20, -10], np.r_[20, 20, 25],
    mesh.gridCC,
)[0]

# Assign magnetization values
model[ind, :] = np.kron(
    np.ones((ind.shape[0], 1)), M_xyz*0.05
)

# Remove air cells
model = model[actv, :]

# Create active map to go from reduce set to full
actvMap = Maps.InjectActiveCells(mesh, actv, np.nan)

# Creat reduced identity map
idenMap = Maps.IdentityMap(nP=nC*3)

# Create the forward model operator
prob = PF.Magnetics.MagneticIntegral(
    mesh, chiMap=idenMap, actInd=actv,
    modelType='vector'
)

# Pair the survey and problem
survey.pair(prob)

# Compute some data and add some random noise
data = prob.fields(Utils.mkvc(model))
std = 5  # nT
data += np.random.randn(len(data))*std
wd = np.ones(len(data))*std

# Assigne data and uncertainties to the survey
survey.dobs = data
survey.std = wd

# Create an projection matrix for plotting later
actvPlot = Maps.InjectActiveCells(mesh, actv, np.nan)

# Plot the model and data
plt.figure()
ax = plt.subplot(2, 1, 1)
im = Utils.PlotUtils.plot2Ddata(xyzLoc, data, ax=ax)
plt.colorbar(im[0])
ax.set_title('Predicted data.')
plt.gca().set_aspect('equal', adjustable='box')

# Plot the vector model
ax = plt.subplot(2, 1, 2)
plotVectorSectionsOctree(
    mesh, model, axs=ax, normal='Y', ind=66,
    actvMap=actvPlot, scale=0.5, vmin=0., vmax=0.01
)
ax.set_xlim([-200, 200])
ax.set_ylim([-100, 75])
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.gca().set_aspect('equal', adjustable='box')

plt.show()


######################################################################
# Inversion
# ---------
#
# We can now attempt the inverse calculations. We put some great care
# in design an inversion methology that would yield geologically
# reasonable solution for the non-induced problem.
# The inversion is done in two stages. First we compute a smooth
# solution using a Cartesian coordinate system, then a sparse
# inversion in the Spherical domain.
#

# Create sensitivity weights from our linear forward operator
rxLoc = survey.srcField.rxList[0].locs

# This Mapping connects the regularizations for the three-component
# vector model
wires = Maps.Wires(('p', nC), ('s', nC), ('t', nC))

# Create sensitivity weights from our linear forward operator
# so that all cells get equal chance to contribute to the solution
wr = np.sum(prob.G**2., axis=0)**0.5
wr = (wr/np.max(wr))

# Create three regularization for the different components
# of magnetization
reg_p = Regularization.Sparse(mesh, indActive=actv, mapping=wires.p)
reg_p.mref = np.zeros(3*nC)
reg_p.cell_weights = (wires.p * wr)

reg_s = Regularization.Sparse(mesh, indActive=actv, mapping=wires.s)
reg_s.mref = np.zeros(3*nC)
reg_s.cell_weights = (wires.s * wr)

reg_t = Regularization.Sparse(mesh, indActive=actv, mapping=wires.t)
reg_t.mref = np.zeros(3*nC)
reg_t.cell_weights = (wires.t * wr)

reg = reg_p + reg_s + reg_t
reg.mref = np.zeros(3*nC)

# Data misfit function
dmis = DataMisfit.l2_DataMisfit(survey)
dmis.W = 1./survey.std

# Add directives to the inversion
opt = Optimization.ProjectedGNCG(maxIter=30, lower=-10, upper=10.,
                                 maxIterLS=20, maxIterCG=20, tolCG=1e-4)

invProb = InvProblem.BaseInvProblem(dmis, reg, opt)

# A list of directive to control the inverson
betaest = Directives.BetaEstimate_ByEig()

# Here is where the norms are applied
# Use pick a treshold parameter empirically based on the distribution of
#  model parameters
IRLS = Directives.Update_IRLS(
    f_min_change=1e-3, maxIRLSiter=0, beta_tol=5e-1
)

# Pre-conditioner
update_Jacobi = Directives.UpdatePreconditioner()

inv = Inversion.BaseInversion(invProb,
                              directiveList=[IRLS, update_Jacobi, betaest])

# Run the inversion
m0 = np.ones(3*nC) * 1e-4  # Starting model
mrec_MVIC = inv.run(m0)

###############################################################
# Sparse Vector Inversion
# -----------------------
#
# Re-run the MVI in spherical domain so we can impose
# sparsity in the vectors.
#
#

mstart = Utils.matutils.cartesian2spherical(mrec_MVIC.reshape((nC, 3), order='F'))
beta = invProb.beta
dmis.prob.coordinate_system = 'spherical'
dmis.prob.model = mstart

# Create a block diagonal regularization
wires = Maps.Wires(('amp', nC), ('theta', nC), ('phi', nC))

# Create a Combo Regularization
# Regularize the amplitude of the vectors
reg_a = Regularization.Sparse(mesh, indActive=actv,
                              mapping=wires.amp)
reg_a.norms = np.c_[0., 0., 0., 0.]  # Sparse on the model and its gradients
reg_a.mref = np.zeros(3*nC)

# Regularize the vertical angle of the vectors
reg_t = Regularization.Sparse(mesh, indActive=actv,
                              mapping=wires.theta)
reg_t.alpha_s = 0.  # No reference angle
reg_t.space = 'spherical'
reg_t.norms = np.c_[2., 0., 0., 0.]  # Only norm on gradients used

# Regularize the horizontal angle of the vectors
reg_p = Regularization.Sparse(mesh, indActive=actv,
                              mapping=wires.phi)
reg_p.alpha_s = 0.  # No reference angle
reg_p.space = 'spherical'
reg_p.norms = np.c_[2., 0., 0., 0.]  # Only norm on gradients used

reg = reg_a + reg_t + reg_p
reg.mref = np.zeros(3*nC)

Lbound = np.kron(np.asarray([0, -np.inf, -np.inf]), np.ones(nC))
Ubound = np.kron(np.asarray([10, np.inf, np.inf]), np.ones(nC))

# Add directives to the inversion
opt = Optimization.ProjectedGNCG(maxIter=20,
                                 lower=Lbound,
                                 upper=Ubound,
                                 maxIterLS=20,
                                 maxIterCG=30,
                                 tolCG=1e-3,
                                 stepOffBoundsFact=1e-3,
                                 )
opt.approxHinv = None

invProb = InvProblem.BaseInvProblem(dmis, reg, opt, beta=beta)

# Here is where the norms are applied
IRLS = Directives.Update_IRLS(f_min_change=1e-4, maxIRLSiter=20,
                              minGNiter=1, beta_tol=0.5,
                              coolingRate=1, coolEps_q=True,
                              betaSearch=False)

# Special directive specific to the mag amplitude problem. The sensitivity
# weights are update between each iteration.
ProjSpherical = Directives.ProjectSphericalBounds()
update_SensWeight = Directives.UpdateSensitivityWeights()
update_Jacobi = Directives.UpdatePreconditioner()

inv = Inversion.BaseInversion(
    invProb,
    directiveList=[
        ProjSpherical, IRLS, update_SensWeight, update_Jacobi
    ]
)

mrec_MVI_S = inv.run(mstart)

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
vec_xyz = Utils.matutils.spherical2cartesian(
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
