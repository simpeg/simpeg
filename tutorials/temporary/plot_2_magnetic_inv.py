
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

import os, shutil
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from discretize import TensorMesh

from SimPEG.potential_fields import magnetics
from SimPEG.utils import plot2Ddata, surface2ind_topo
from SimPEG import (
    maps, data, inverse_problem, data_misfit, regularization, optimization,
    directives, inversion, utils
    )

# sphinx_gallery_thumbnail_number = 2

#############################################
# Load Data and Plot
# ------------------
#
# File names for assets we are loading. Here we load the topography, observed
# data and the true model define on the whole mesh.
#

topo_filename = os.path.dirname(magnetics.__file__) + '\\..\\..\\..\\tutorials\\assets\\magnetics\\magnetics_topo.txt'
data_filename = os.path.dirname(magnetics.__file__) + '\\..\\..\\..\\tutorials\\assets\\magnetics\\magnetics_data.obs'
model_filename = os.path.dirname(magnetics.__file__) + '\\..\\..\\..\\tutorials\\assets\\magnetics\\true_model.txt'


#############################################
# Load Data and Plot
# ------------------
#
# Here we load and plot synthetic TMI data.
#

xyz_topo = np.loadtxt(str(topo_filename))
dobs = np.loadtxt(str(data_filename))

receiver_locations = dobs[:, 0:3]
dobs = dobs[:, -1]

# Plot
fig = plt.figure(figsize=(6, 5))
v_max = np.max(np.abs(dobs))

ax1 = fig.add_axes([0.05, 0.05, 0.8, 0.9])
plot2Ddata(
    receiver_locations, dobs, ax=ax1, ncontour=30, clim=(-v_max, v_max),
    contourOpts={"cmap": "RdBu_r"}
)
ax1.set_title('TMI Anomaly')

ax2 = fig.add_axes([0.85, 0.05, 0.05, 0.9])
norm = mpl.colors.Normalize(
        vmin=-np.max(np.abs(dobs)), vmax=np.max(np.abs(dobs))
)
cbar = mpl.colorbar.ColorbarBase(
        ax2, norm=norm, orientation='vertical', cmap=mpl.cm.RdBu_r
)
cbar.set_label('$nT$', rotation=270, labelpad=15, size=12)

plt.show()

#############################################
# Assign Uncertainty
# ------------------

maximum_anomaly = np.max(np.abs(dobs))

uncertainties = 0.02*maximum_anomaly*np.ones(len(dobs))

#############################################
# Defining the Survey
# -------------------
#
# Here, we define survey that will be used for the simulation. Magnetic
# surveys are simple to create. The user only needs an (N, 3) array to define
# the xyz locations of the observation locations, the list of field components
# which are to be modeled and the properties of the Earth's field.
#

# Define the component(s) of the field we are inverting as a list. Here we will
# invert total magnetic intensity data.
components = ["tmi"]

# Use the observation locations and components to define the receivers. To
# simulate data, the receivers must be defined as a list.
receiver_list = magnetics.receivers.point_receiver(
        receiver_locations, components=components
        )

receiver_list = [receiver_list]

# Define the inducing field H0 = (intensity [nT], inclination [deg], declination [deg])
inclination = 90
declination = 0
strength = 50000
inducing_field = (strength, inclination, declination)

source_field = magnetics.sources.SourceField(
    receiver_list=receiver_list, parameters=inducing_field
    )

# Define the survey
survey = magnetics.survey.MagneticSurvey(source_field)

#############################################
# Defining the Data
# -----------------
#
# Here is where we define the data that is inverted. The data is defined by
# the survey, the observation values and the uncertainties.
#

data_object = data.Data(survey, dobs=dobs, noise_floor=uncertainties)


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

# Define background susceptibility model in SI
background_susceptibility = 1e-4

# Find the indecies of the active cells in forward model (ones below surface)
ind_active = surface2ind_topo(mesh, xyz_topo)

# Define mapping from model to active cells
nC = int(ind_active.sum())
model_map = maps.IdentityMap(nP=nC)  # model consists of a value for each cell

# Define starting model
starting_model = background_susceptibility*np.ones(nC)

##############################################
# Define the Physics
# ------------------
#
# Here, we define the physics of the gravity problem.
# 

# Define the problem. Define the cells below topography and the mapping
simulation = magnetics.simulation.MagneticIntegralSimulation(
    survey=survey, mesh=mesh,
    modelType='susceptibility', chiMap=model_map,
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
reg = regularization.Sparse(
    mesh, indActive=ind_active, mapping=model_map, mref=starting_model,
    alpha_s=1, alpha_x=1, alpha_y=1, alpha_z=1
)
reg.norms = np.c_[0.1, 0.1, 0.1, 0.1]  # Define sparse and blocky norms p=(0, 2)

# Create model weights based on sensitivity matrix (sensitivity weighting)
wr = simulation.getJtJdiag(starting_model)**0.5
wr = (wr/np.max(np.abs(wr)))
reg.cell_weights = wr  # include in regularization

# Define how the optimization problem is solved.
opt = optimization.ProjectedGNCG(
    maxIter=10, lower=0., upper=1.,
    maxIterLS=20, maxIterCG=10, tolCG=1e-3
)

# Here we define the inverse problem that is to be solved
inv_prob = inverse_problem.BaseInvProblem(dmis, reg, opt)

# Here we define any directive that are carried out during the inversion. Here,
# we apply the itertively re-weighted least squares.
starting_beta = directives.BetaEstimate_ByEig(beta0_ratio=1)
beta_schedule = directives.BetaSchedule(coolingFactor=2, coolingRate=3)
save_iteration = directives.SaveOutputEveryIteration(save_txt=False)
update_IRLS = directives.Update_IRLS(
    f_min_change=1e-4, max_irls_iterations=30,
    coolEpsFact=1.5, beta_tol=1e-2,
)
update_jacobi = directives.UpdatePreconditioner()
target_misfit = directives.TargetMisfit(chifact=1)

directives_list = [
    starting_beta, save_iteration, update_IRLS, update_jacobi,
    ]

# Here we combine the inverse problem and the set of directives
inv = inversion.BaseInversion(inv_prob, directives_list)

# Print target misfit to compare with convergence
# print("Target misfit is " + str(target_misfit.target))

# Run the inversion
recovered_model = inv.run(starting_model)

# Remove directory storing sensitivities
shutil.rmtree(".\\sensitivity.zarr")

############################################################
# Plotting True Model and Recovered Model
# ---------------------------------------
#

# Load the true model and keep only active cells
true_model = np.loadtxt(str(model_filename))
true_model = true_model[ind_active]

# Plot True Model
fig = plt.figure(figsize=(9, 4))
plotting_map = maps.InjectActiveCells(mesh, ind_active, np.nan)

ax1 = fig.add_axes([0.05, 0.05, 0.78, 0.9])
mesh.plotSlice(
    plotting_map*true_model, normal='Y', ax=ax1, ind=int(mesh.nCy/2), grid=True,
    clim=(np.min(true_model), np.max(true_model)), pcolorOpts={'cmap': 'jet'}
)
ax1.set_title('Model slice at y = 0 m')

ax2 = fig.add_axes([0.85, 0.05, 0.05, 0.9])
norm = mpl.colors.Normalize(vmin=np.min(true_model), vmax=np.max(true_model))
cbar = mpl.colorbar.ColorbarBase(
    ax2, norm=norm, orientation='vertical', cmap=mpl.cm.jet, format='%.1e'
)
cbar.set_label('SI', rotation=270, labelpad=15, size=12)

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
    ax2, norm=norm, orientation='vertical', cmap=mpl.cm.jet, format='%.1e'
)
cbar.set_label('SI',rotation=270, labelpad=15, size=12)

plt.show()

###################################################################
# Plotting Predicted Data and Misfit
# ----------------------------------
#

dpred = inv_prob.dpred
data_array = np.c_[dobs, dpred, (dobs-dpred)/uncertainties]

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
        receiver_list[0].locations, data_array[:, ii], ax=ax1[ii], ncontour=30,
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





