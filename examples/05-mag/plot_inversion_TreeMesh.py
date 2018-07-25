from SimPEG import (Mesh, Directives, Maps,
                    InvProblem, Optimization, DataMisfit,
                    Inversion, Utils, Regularization)

import SimPEG.PF as PF
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt

###############################################################################
# Setup
# -----
#
# Define the survey and model parameters
#
# First we need to define the direction of the inducing field
# As a simple case, we pick a vertical inducing field of magnitude 50,000 nT.
# From old convention, field orientation is given as an azimuth from North
# (positive clockwise) and dip from the horizontal (positive downward).

H0 = (50000., 90., 0.)

# Assume all induced so the magnetization M is also in the same direction
M = np.array([90, 0])

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

# Create a topography
mesh = Utils.modelutils.meshBuilder(topo, h, padDist,
                                    meshType='TREE',
                                    verticalAlignment='center')

# Discretize around topography, only 2th and 3th level for now
mesh = Utils.modelutils.refineTree(mesh, topo, dtype='surface',
                                   nCpad=[0, 3, 1], finalize=False)

# Dsicretize finer around the observation locations...finalize here
mesh = Utils.modelutils.refineTree(mesh, xyzLoc, dtype='surface',
                                   nCpad=[4, 0, 0], finalize=True)

# Define an active cells from topo
actv = Utils.surface2ind_topo(mesh, topo)
nC = int(actv.sum())

###########################################################################
# Forward modeling data
# ---------------------
#
# We can now create a susceptibility model and generate data
# Lets start with a simple block below topography
#

model = Utils.ModelBuilder.addBlock(
    mesh.gridCC, np.zeros(mesh.nC),
    np.r_[-40, -40, -50], np.r_[40, 40, 0], 0.05
)[actv]

# Create active map to go from reduce set to full
actvMap = Maps.InjectActiveCells(mesh, actv, np.nan)

# Creat reduced identity map
idenMap = Maps.IdentityMap(nP=nC)

# Create the forward model operator
prob = PF.Magnetics.MagneticIntegral(mesh, chiMap=idenMap, actInd=actv)

# Pair the survey and problem
survey.pair(prob)

# Compute linear forward operator and compute some data
data = prob.fields(model)

data += np.random.randn(len(data))
wd = np.ones(len(data))*1.

survey.dobs = data
survey.std = wd


actvPlot = Maps.InjectActiveCells(mesh, actv, np.nan)
# Create a few models
plt.figure()
ax = plt.subplot()
mesh.plotSlice(actvPlot*model, ax=ax, normal='Y', ind=65)
ax.set_xlim([-300, 300])
ax.set_ylim([-300, 75])
ax.set_title('A simple block model.')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.gca().set_aspect('equal', adjustable='box')
plt.show()


Utils.PlotUtils.plot2Ddata(xyzLoc, data)
ax.set_title('Predicted data.')
plt.gca().set_aspect('equal', adjustable='box')

######################################################################
# Inversion
# ---------
#
# We can now attempt the inverse calculations. We put some great care
# in design an inversion methology that would yield geologically
# reasonable solution, even when non-uniform discretization
# is used, such as TreeMesh.
#
# We will run the sparse inversion.
#

# Create sensitivity weights from our linear forward operator
rxLoc = survey.srcField.rxList[0].locs
wr = np.zeros(prob.G.shape[1])
for ii in range(survey.nD):
    wr += (prob.G[ii, :]/survey.std[ii])**2.

# wr = (wr/np.max(wr))
wr = wr**0.5

# Create a regularization with sparse norms on the model and its
# gradients.
reg = Regularization.Sparse(mesh, indActive=actv, mapping=idenMap)
reg.norms = np.c_[0, 0, 0, 0]
reg.cell_weights = wr  # Apply sensitivity weights

reg.mref = np.zeros(nC)

# Data misfit function
dmis = DataMisfit.l2_DataMisfit(survey)
dmis.W = 1./survey.std

# Add directives to the inversion
opt = Optimization.ProjectedGNCG(maxIter=30, lower=0., upper=10.,
                                 maxIterLS=20, maxIterCG=20, tolCG=1e-4)
invProb = InvProblem.BaseInvProblem(dmis, reg, opt)
betaest = Directives.BetaEstimate_ByEig()

# Here is where the norms are applied
# Use pick a treshold parameter empirically based on the distribution of
#  model parameters
IRLS = Directives.Update_IRLS(f_min_change=1e-3, maxIRLSiter=30, beta_tol=5e-1)
update_Jacobi = Directives.UpdatePreconditioner()

saveOuput = Directives.SaveOutputEveryIteration()
# saveModel.fileName = work_dir + out_dir + 'ModelSus'

inv = Inversion.BaseInversion(
    invProb, directiveList=[betaest, saveOuput, IRLS, update_Jacobi]
)

# Run the inversion
m0 = np.ones(nC) * 1e-4  # Starting model
prob.model = m0
mrec = inv.run(m0)


plt.figure()
ax = plt.subplot()
mesh.plotSlice(actvMap*mrec, ax=ax, normal='Y', ind=65)
ax.set_xlim([-300, 300])
ax.set_ylim([-300, 75])
ax.set_title('Recovered model.')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.gca().set_aspect('equal', adjustable='box')
plt.show()

saveOuput.plot_misfit_curves()
