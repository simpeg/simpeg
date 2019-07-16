
"""
Magnetics: Sparse Norm Inversion
================================

Here we invert total magnetic intensity (TMI) data to recover a magnetic
susceptibility model. We formulate the inverse problem as an iteratively
re-weighted least-squares (IRLS) optimization problem. For this tutorial, we
focus on the following:

    - Defining the survey from xyz formatted data
    - Generating a mesh based on survey geometry
    - Including surface topography
    - Defining the inverse problem (data misfit, regularization, directives)
    - Applying sensitivity weighting
    - Setting sparse and blocky norms
    - Plotting the recovered model and data misfit

Although we consider TMI data in this tutorial, the same approach
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
# Here we load and plot synthetic TMI data.
#

topo_name = os.path.dirname(PF.__file__) + '\\..\\..\\tutorials\\assets\\magnetic_topo.txt'
data_name = os.path.dirname(PF.__file__) + '\\..\\..\\tutorials\\assets\\magnetic_data.txt'
topo = np.loadtxt(str(topo_name))
dobs = np.loadtxt(str(data_name))

rx_locs = dobs[:, 0:3]
dobs = dobs[:, -1]

# Plot
fig = plt.figure(figsize=(6, 5))
v_max = np.max(np.abs(dobs))

ax1 = fig.add_axes([0.05, 0.05, 0.8, 0.9])
plot2Ddata(
    rx_locs, dobs, ax=ax1, ncontour=30, clim=(-v_max, v_max),
    contourOpts={"cmap": "RdBu_r"}
)
ax1.set_title('TMI Anomaly')

ax2 = fig.add_axes([0.85, 0.05, 0.05, 0.9])
norm = mpl.colors.Normalize(
        vmin=-np.max(np.abs(dobs)), vmax=np.max(np.abs(dobs))
)
cbar = mpl.colorbar.ColorbarBase(
        ax2, norm=norm, orientation='vertical', cmap='RdBu_r'
)
cbar.set_label('$nT$', rotation=270, labelpad=15, size=12)

plt.show()

#############################################
# Assign Uncertainty
# ------------------

dunc = 0.5*np.ones(len(dobs))

#############################################
# Defining the Survey
# -------------------
#
# Here, we define survey that will be used for all tutorial examples. Magnetic
# surveys are simple to create. The user needs an (N, 3) array to define
# the xyz positions of the observation locations. The user also needs to
# define the Earth's magnetic field intensity and orientation. Here, we
# create a basic airborne survey with a flight height of 10 m above the
# surface topography.
#

# Define the receivers
rx_list = PF.BaseMag.RxObs(rx_locs)

# Define the inducing field H0 = (intensity [nT], inclination [deg], declination [deg])
H0 = (50000, 90, 0)
src_field = PF.BaseMag.SrcField([rx_list], param=H0)   # Define the source field

# Define the survey
survey = PF.BaseMag.LinearSurvey(src_field)              # Define the survey

#############################################
# Defining a Tensor Mesh
# ----------------------
#
# Here, we create the tensor mesh that will be used to invert TMI data.
# If desired, we could define an OcTree mesh.
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
# structures. Here, the background is 1e-4 SI.
#

# Define density contrast values for each unit in g/cc
background_val = 1e-4

# Find the indecies of the active cells in forward model (ones below surface)
ind_active = surface2ind_topo(mesh, topo)

# Define mapping from model to active cells
nC = int(ind_active.sum())
mod_map = Maps.IdentityMap(nP=nC)  # model consists of a value for each cell

# Define starting model
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
cbar.set_label('$SI$', rotation=270, labelpad=15, size=12)

plt.show()

##############################################
# Define the Physics
# ------------------
#
# Here, we define the physics of the gravity problem.
# 

# Define the problem. Define the cells below topography and the mapping
prob = PF.Magnetics.MagneticIntegral(
    mesh, chiMap=mod_map, actInd=ind_active, rx_type='tmi'
)
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
reg = Regularization.Sparse(
    mesh, indActive=ind_active, mapping=mod_map,
    alpha_s=0.04, alpha_x=1, alpha_y=1, alpha_z=1
)
reg.norms = np.c_[1, 0, 0, 0]  # Define sparse and blocky norms p=(0, 2)

# Create model weights based on sensitivity matrix (sensitivity weighting)
wr = np.sum(prob.G**2., axis=0)**0.5
wr = (wr/np.max(np.abs(wr)))
reg.cell_weights = wr  # include sensitivity weighting in regularization

# Define how the optimization problem is solved.
opt = Optimization.ProjectedGNCG(
    maxIter=5, lower=-1., upper=1.,
    maxIterLS=20, maxIterCG=10, tolCG=1e-3
)

# Here we define the inverse problem that is to be solved
inv_prob = InvProblem.BaseInvProblem(dmis, reg, opt)

# Here we define any directive that are carried out during the inversion. Here,
# we apply the itertively re-weighted least squares.
betaest = Directives.BetaEstimate_ByEig(beta0_ratio=1e-1)
saveDict = Directives.SaveOutputEveryIteration(save_txt=False)
update_Jacobi = Directives.UpdatePreconditioner()
IRLS = Directives.Update_IRLS(
    f_min_change=1e-4, maxIRLSiter=30, coolEpsFact=1.5, beta_tol=1e-1,
)

# Here we combine the inverse problem and the set of directives
inv = Inversion.BaseInversion(
    inv_prob, directiveList=[IRLS, betaest, update_Jacobi, saveDict]
)

# Run the inversion
mrec = inv.run(m0)

############################################################
# Plotting True Model and Recovered Model
# ---------------------------------------
#

# Construct True Model

# Define susceptibility values for each unit in SI
background_val = 0.0001
sphere_val = 0.01

mtrue = background_val*np.ones(ind_active.sum())
ind_sphere = ModelBuilder.getIndicesSphere(
    np.r_[0., 0., -45.], 15., mesh.gridCC
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
cbar.set_label('SI', rotation=270, labelpad=15, size=12)

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
    ax2, norm=norm, orientation='vertical', cmap='jet', format='%.1e'
)
cbar.set_label('SI',rotation=270, labelpad=15, size=12)

plt.show()

###################################################################
# Plotting Predicted Data and Misfit
# ----------------------------------
#

dpred = inv_prob.dpred
data_array = np.c_[dobs, dpred, (dobs-dpred)/dunc]

fig = plt.figure(figsize=(17, 4))
plot_title=['Observed', 'Predicted', 'Normalized Misfit']
plot_units=['nT', 'nT', '']

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





