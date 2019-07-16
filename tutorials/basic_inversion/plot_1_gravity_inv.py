
"""
Gravity: Least-Squares Inversion
================================

Here we invert gravity anomaly data to recover a density contrast model. We
formulate the inverse problem as a least-squares optimization problem. For
this tutorial, we focus on the following:

    - Defining the survey from xyz formatted data
    - Generating a mesh based on survey geometry
    - Including surface topography
    - Defining the inverse problem (data misfit, regularization, directives)
    - Applying sensitivity weighting
    - Plotting the recovered model and data misfit

Although we consider gravity anomaly data in this tutorial, the same approach
can be used to invert other types of geophysical data.
    

"""

#########################################################################
# Import modules
# --------------
#

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from discretize import TensorMesh
from SimPEG.Utils import plot2Ddata, ModelBuilder, surface2ind_topo
from SimPEG import (
    Maps, PF, InvProblem, DataMisfit, Regularization, Optimization, Directives,
    Inversion, Utils
    )

# sphinx_gallery_thumbnail_number = 2

#############################################
# Load Data and Plot
# ------------------
#
# Here we load and plot synthetic gravity anomaly data.
#

topo_name = os.path.dirname(PF.__file__) + '\\..\\..\\tutorials\\assets\\gravity_topo.txt'
data_name = os.path.dirname(PF.__file__) + '\\..\\..\\tutorials\\assets\\gravity_data.txt'
topo = np.loadtxt(str(topo_name))
dobs = np.loadtxt(str(data_name))

rx_locs = dobs[:, 0:3]
dobs = dobs[:, -1]

# Plot
fig = plt.figure(figsize=(7, 5))

ax1 = fig.add_axes([0.05, 0.05, 0.78, 0.9])
plot2Ddata(rx_locs, dobs, ax=ax1, contourOpts={"cmap": "RdBu_r"})
ax1.set_title('Gravity Anomaly')

ax2 = fig.add_axes([0.82, 0.05, 0.03, 0.9])
norm = mpl.colors.Normalize(
    vmin=-np.max(np.abs(dobs)), vmax=np.max(np.abs(dobs))
)
cbar = mpl.colorbar.ColorbarBase(
        ax2, norm=norm, orientation='vertical', cmap='RdBu_r', format='%.1e'
        )
cbar.set_label('$mgal$', rotation=270, labelpad=15, size=12)

plt.show()

#############################################
# Assign Uncertainty
# ------------------

dunc = 5e-4*np.ones(len(dobs))

#############################################
# Defining the Survey
# -------------------
#
# Here, we define survey that will be used for this tutorial. Gravity
# surveys are simple to create. The user only needs an (N, 3) array to define
# the xyz locations of the observation locations. From this, the user can
# define the receivers and the source field.
#

# Define the survey
rx_list = PF.BaseGrav.RxObs(rx_locs)            # Define receivers
src_field = PF.BaseGrav.SrcField([rx_list])     # Define the source field
survey = PF.BaseGrav.LinearSurvey(src_field)    # Define the survey

#############################################
# Defining a Tensor Mesh
# ----------------------
#
# Here, we create the tensor mesh that will be used to invert gravity anomaly
# data. If desired, we could define an OcTree mesh.
#

dh = 5.
hx = [(dh, 5, -1.3), (dh, 40), (dh, 5, 1.3)]
hy = [(dh, 5, -1.3), (dh, 40), (dh, 5, 1.3)]
hz = [(dh, 5, -1.3), (dh, 15)]
mesh = TensorMesh([hx, hy, hz], 'CCN')

########################################################
# Starting/Reference Model and Mapping on Tensor Mesh
# ---------------------------------------------------
#
# Here, we would create starting and/or reference models for the inversion as
# well as the mapping from the model space to the active cells. Starting and
# reference models can be a constant background value or contain a-priori
# structures. Here, the background is 1e-6 g/cc.
#

# Define density contrast values for each unit in g/cc. Don't make this 0!
# Otherwise the gradient for the 1st iteration is zero and the inversion will
# not converge.
background_val = 1e-6

# Find the indecies of the active cells in forward model (ones below surface)
ind_active = surface2ind_topo(mesh, topo)

# Define mapping from model to active cells
nC = int(ind_active.sum())
mod_map = Maps.IdentityMap(nP=nC)  # model consists of a value for each active cell

# Define and plot starting model
m0 = background_val*np.ones(nC)
fig = plt.figure(figsize=(9, 4))
plotting_map = Maps.InjectActiveCells(mesh, ind_active, np.nan)

ax1 = fig.add_axes([0.05, 0.05, 0.78, 0.9])
mesh.plotSlice(
    plotting_map*m0, normal='Y', ax=ax1, ind=int(mesh.nCy/2), grid=True,
    clim=(-0.1, 0.1), pcolorOpts={'cmap': 'jet'}
    )
ax1.set_title('Starting model slice at y = 0 m')

ax2 = fig.add_axes([0.85, 0.05, 0.05, 0.9])
norm = mpl.colors.Normalize(vmin=-0.1, vmax=0.1)
cbar = mpl.colorbar.ColorbarBase(
    ax2, norm=norm, orientation='vertical', cmap='jet', format='%.1e'
    )
cbar.set_label('$g/cm^3$', rotation=270, labelpad=15, size=12)

plt.show()


##############################################
# Define the Physics
# ------------------
#
# Here, we define the physics of the gravity problem.
# 

# Define the problem. Define the cells below topography and the mapping
prob = PF.Gravity.GravityIntegral(mesh, rhoMap=mod_map, actInd=ind_active)

# Pair the survey and problem
survey.pair(prob)

# Define the observed data and uncertainties
survey.dobs = dobs
survey.std = dunc

#######################################################################
# Define Inverse Problem
# ----------------------
#
# Here we define the inverse problem.
#

# Define the data misfit (Here we use weighted L2-norm)
dmis = DataMisfit.l2_DataMisfit(survey)
dmis.W = Utils.sdiag(1/dunc)

# Define the regularization (model objective function)
reg = Regularization.Simple(
    mesh, indActive=ind_active, mapping=mod_map,
    alpha_s=0.04, alpha_x=1, alpha_y=1, alpha_z=1
)

# Create model weights based on sensitivity matrix (sensitivity weighting)
wr = np.sum(prob.G**2., axis=0)**0.5
wr = (wr/np.max(np.abs(wr)))
reg.cell_weights = wr  # include in regularization

# Define how the optimization problem is solved.
opt = Optimization.ProjectedGNCG(
    maxIter=5, lower=-1., upper=1.,
    maxIterLS=20, maxIterCG=10, tolCG=1e-3
)

# Here we define the inverse problem that is to be solved
inv_prob = InvProblem.BaseInvProblem(dmis, reg, opt)

# Here we define any directive that are carried out during the inversion
betaest = Directives.BetaEstimate_ByEig(beta0_ratio=1e-1)
saveDict = Directives.SaveOutputEveryIteration(save_txt=False)
update_Jacobi = Directives.UpdatePreconditioner()

# Here we combine the inverse problem and the set of directives
inv = Inversion.BaseInversion(
    inv_prob, directiveList=[betaest, update_Jacobi, saveDict]
)

# Run inversion
mrec = inv.run(m0)


############################################################
# Plotting True Model and Recovered Model
# ---------------------------------------
#

# Construct True Model

# Define density contrast values for each unit in g/cc
background_val = 0.
block_val = -0.1
sphere_val = 0.1

mtrue = background_val*np.ones(nC)

ind_block = (
    (mesh.gridCC[ind_active, 0] > -50.) & (mesh.gridCC[ind_active, 0] < -20.) &
    (mesh.gridCC[ind_active, 1] > -15.) & (mesh.gridCC[ind_active, 1] < 15.) &
    (mesh.gridCC[ind_active, 2] > -50.) & (mesh.gridCC[ind_active, 2] < -30.)
)
mtrue[ind_block] = block_val

ind_sphere = ModelBuilder.getIndicesSphere(
    np.r_[35., 0., -40.], 15., mesh.gridCC
)
ind_sphere = ind_sphere[ind_active]
mtrue[ind_sphere] = sphere_val

# Plot True Model
fig = plt.figure(figsize=(9, 4))
plotting_map = Maps.InjectActiveCells(mesh, ind_active, np.nan)

ax1 = fig.add_axes([0.05, 0.05, 0.78, 0.9])
mesh.plotSlice(
    plotting_map*mtrue, normal='Y', ax=ax1, ind=int(mesh.nCy/2), grid=True,
    clim=(np.min(mtrue), np.max(mtrue)), pcolorOpts={'cmap': 'jet'}
    )
ax1.set_title('Model slice at y = 0 m')

ax2 = fig.add_axes([0.85, 0.05, 0.05, 0.9])
norm = mpl.colors.Normalize(vmin=np.min(mtrue), vmax=np.max(mtrue))
cbar = mpl.colorbar.ColorbarBase(
    ax2, norm=norm, orientation='vertical', cmap='jet', format='%.1e'
)
cbar.set_label(
    '$g/cm^3$',
    rotation=270, labelpad=15, size=12
)

plt.show()

# Plot Recovered Model
fig = plt.figure(figsize=(9, 4))
plotting_map = Maps.InjectActiveCells(mesh, ind_active, np.nan)

ax1 = fig.add_axes([0.05, 0.05, 0.78, 0.9])
mesh.plotSlice(
    plotting_map*mrec, normal='Y', ax=ax1, ind=int(mesh.nCy/2), grid=True,
    clim=(np.min(mrec), np.max(mrec)), pcolorOpts={'cmap': 'jet'}
)
ax1.set_title('Model slice at y = 0 m')

ax2 = fig.add_axes([0.85, 0.05, 0.05, 0.9])
norm = mpl.colors.Normalize(vmin=np.min(mrec), vmax=np.max(mrec))
cbar = mpl.colorbar.ColorbarBase(
        ax2, norm=norm, orientation='vertical', cmap='jet'
        )
cbar.set_label('$g/cm^3$',rotation=270, labelpad=15, size=12)

plt.show()

###################################################################
# Plotting Predicted Data and Misfit
# ----------------------------------
#

dpred = inv_prob.dpred

data_array = np.c_[dobs, dpred, (dobs-dpred)/dunc]

fig = plt.figure(figsize=(17, 4))
plot_title=['Observed', 'Predicted', 'Normalized Misfit']
plot_units=['mgal', 'mgal', '']

ax1 = 3*[None]
ax2 = 3*[None]
norm = 3*[None]
cbar = 3*[None]
cplot = 3*[None]
v_lim = [np.max(np.abs(dobs)), np.max(np.abs(dobs)), 2]

for ii in range(0, 3):
    
    ax1[ii] = fig.add_axes([0.33*ii+0.03, 0.05, 0.25, 0.9])
    cplot[ii] = plot2Ddata(
        rx_list.locs, data_array[:, ii], ax=ax1[ii], ncontour=30,
        clim=(-v_lim[ii], v_lim[ii]), contourOpts={"cmap": "RdBu_r"}
    )
    ax1[ii].set_title(plot_title[ii])
    
    ax2[ii] = fig.add_axes([0.33*ii+0.27, 0.05, 0.01, 0.9])
    norm[ii] = mpl.colors.Normalize(vmin=-v_lim[ii], vmax=v_lim[ii])
    cbar[ii] = mpl.colorbar.ColorbarBase(
        ax2[ii], norm=norm[ii], orientation='vertical', cmap='RdBu_r'
    )
    cbar[ii].set_label(plot_units[ii], rotation=270, labelpad=15, size=12)

plt.show()

