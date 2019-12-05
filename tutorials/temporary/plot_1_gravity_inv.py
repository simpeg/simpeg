
"""
Gravity: Least-Squares inversion
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

from SimPEG.utils import plot2Ddata, ModelBuilder, surface2ind_topo
from SimPEG.potential_fields import gravity
from SimPEG import (
    maps, data, data_misfit, inverse_problem, regularization, optimization,
    directives, inversion, utils
    )

# sphinx_gallery_thumbnail_number = 4

#############################################
# Load Data and Plot
# ------------------
#
# Here we load and plot synthetic gravity anomaly data.
#

topo_filename = os.path.dirname(gravity.__file__) + '\\..\\..\\..\\tutorials\\assets\\gravity\\gravity_topo.txt'
data_filename = os.path.dirname(gravity.__file__) + '\\..\\..\\..\\tutorials\\assets\\gravity\\gravity_data.obs'
xyz_topo = np.loadtxt(str(topo_filename))
dobs = np.loadtxt(str(data_filename))

receiver_locations = dobs[:, 0:3]
dobs = dobs[:, -1]

# Plot
fig = plt.figure(figsize=(7, 5))

ax1 = fig.add_axes([0.05, 0.05, 0.78, 0.9])
plot2Ddata(receiver_locations, dobs, ax=ax1, contourOpts={"cmap": "RdBu_r"})
ax1.set_title('Gravity Anomaly')

ax2 = fig.add_axes([0.82, 0.05, 0.03, 0.9])
norm = mpl.colors.Normalize(
    vmin=-np.max(np.abs(dobs)), vmax=np.max(np.abs(dobs))
)
cbar = mpl.colorbar.ColorbarBase(
        ax2, norm=norm, orientation='vertical', cmap=mpl.cm.RdBu_r, format='%.1e'
        )
cbar.set_label('$mgal$', rotation=270, labelpad=15, size=12)

plt.show()

#############################################
# Assign Uncertainties
# --------------------

uncertainties = 5e-4*np.ones(np.shape(dobs))

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
receiver_list = [gravity.receivers.point_receiver(receiver_locations, components="gz")]            # Define receivers
source_field = gravity.sources.SourceField(receiver_list=receiver_list)     # Define the source field
survey = gravity.survey.GravitySurvey(source_field)    # Define the survey

#############################################
# Defining the Data
# -----------------
#

data_object = data.Data(survey, dobs=dobs, noise_floor=uncertainties)


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
background_density = 1e-6

# Find the indecies of the active cells in forward model (ones below surface)
ind_active = surface2ind_topo(mesh, xyz_topo)

# Define mapping from model to active cells
nC = int(ind_active.sum())
model_map = maps.IdentityMap(nP=nC)  # model consists of a value for each active cell

# Define and plot starting model
starting_model = background_density*np.ones(nC)
fig = plt.figure(figsize=(9, 4))
plotting_map = maps.InjectActiveCells(mesh, ind_active, np.nan)

ax1 = fig.add_axes([0.05, 0.05, 0.78, 0.9])
mesh.plotSlice(
    plotting_map*starting_model, normal='Y', ax=ax1, ind=int(mesh.nCy/2), grid=True,
    clim=(-0.1, 0.1), pcolorOpts={'cmap': 'jet'}
    )
ax1.set_title('Starting model slice at y = 0 m')

ax2 = fig.add_axes([0.85, 0.05, 0.05, 0.9])
norm = mpl.colors.Normalize(vmin=-0.1, vmax=0.1)
cbar = mpl.colorbar.ColorbarBase(
    ax2, norm=norm, orientation='vertical', cmap=mpl.cm.jet, format='%.1e'
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
simulation = gravity.simulation.GravityIntegralSimulation(
    survey=survey, mesh=mesh, rhoMap=model_map,
    actInd=ind_active, forward_only=False
)




#######################################################################
# Define Inverse Problem
# ----------------------
#
# Here we define the inverse problem.
#

# Define the data misfit (Here we use weighted L2-norm)
dmis = data_misfit.L2DataMisfit(data=data_object, simulation=simulation)
dmis.W = utils.sdiag(1/uncertainties)

# Define the regularization (model objective function)
reg = regularization.Simple(
    mesh, indActive=ind_active, mapping=model_map,
    alpha_s=1, alpha_x=1, alpha_y=1, alpha_z=1
)

# Create model weights based on sensitivity matrix (sensitivity weighting)
wr = simulation.getJtJdiag(starting_model)**0.5
wr = (wr/np.max(np.abs(wr)))
reg.cell_weights = wr  # include in regularization

# Define how the optimization problem is solved.
opt = optimization.ProjectedGNCG(
    maxIter=10, lower=-1., upper=1.,
    maxIterLS=20, maxIterCG=10, tolCG=1e-3
)

# Here we define the inverse problem that is to be solved
inv_prob = inverse_problem.BaseInvProblem(dmis, reg, opt)

# Here we define any directive that are carried out during the inversion
beta_estimation = directives.BetaEstimate_ByEig(beta0_ratio=1e-1)
save_iteration = directives.SaveOutputEveryIteration(save_txt=False)
update_jacobi = directives.UpdatePreconditioner()

# Here we combine the inverse problem and the set of directives
inv = inversion.BaseInversion(
    inv_prob, [beta_estimation, update_jacobi, save_iteration]
)

# Run inversion
recovered_model = inv.run(starting_model)


############################################################
# Plotting True Model and Recovered Model
# ---------------------------------------
#

# Construct True Model

# Define density contrast values for each unit in g/cc
background_density = 0.
block_density = -0.1
sphere_density = 0.1

mtrue = background_density*np.ones(nC)

ind_block = (
    (mesh.gridCC[ind_active, 0] > -50.) & (mesh.gridCC[ind_active, 0] < -20.) &
    (mesh.gridCC[ind_active, 1] > -15.) & (mesh.gridCC[ind_active, 1] < 15.) &
    (mesh.gridCC[ind_active, 2] > -50.) & (mesh.gridCC[ind_active, 2] < -30.)
)
mtrue[ind_block] = block_density

ind_sphere = ModelBuilder.getIndicesSphere(
    np.r_[35., 0., -40.], 15., mesh.gridCC
)
ind_sphere = ind_sphere[ind_active]
mtrue[ind_sphere] = sphere_density

# Plot True Model
fig = plt.figure(figsize=(9, 4))
plotting_map = maps.InjectActiveCells(mesh, ind_active, np.nan)

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
plotting_map = maps.InjectActiveCells(mesh, ind_active, np.nan)

ax1 = fig.add_axes([0.05, 0.05, 0.78, 0.9])
mesh.plotSlice(
    plotting_map*recovered_model, normal='Y', ax=ax1, ind=int(mesh.nCy/2), grid=True,
    clim=(np.min(recovered_model), np.max(recovered_model)), pcolorOpts={'cmap': 'jet'}
)
ax1.set_title('Model slice at y = 0 m')

ax2 = fig.add_axes([0.85, 0.05, 0.05, 0.9])
norm = mpl.colors.Normalize(vmin=np.min(recovered_model), vmax=np.max(recovered_model))
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

data_array = np.c_[dobs, dpred, (dobs-dpred)/uncertainties]

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
        receiver_list.locs, data_array[:, ii], ax=ax1[ii], ncontour=30,
        clim=(-v_lim[ii], v_lim[ii]), contourOpts={"cmap": "RdBu_r"}
    )
    ax1[ii].set_title(plot_title[ii])
    
    ax2[ii] = fig.add_axes([0.33*ii+0.27, 0.05, 0.01, 0.9])
    norm[ii] = mpl.colors.Normalize(vmin=-v_lim[ii], vmax=v_lim[ii])
    cbar[ii] = mpl.colorbar.ColorbarBase(
        ax2[ii], norm=norm[ii], orientation='vertical', cmap=mpl.cm.RdBu_r
    )
    cbar[ii].set_label(plot_units[ii], rotation=270, labelpad=15, size=12)

plt.show()

