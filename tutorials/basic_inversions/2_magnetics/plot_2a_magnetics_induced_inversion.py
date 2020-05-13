
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
    - Defining the inverse problem (data misfit, regularization, optimization)
    - Specifying directives for the inversion
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
import tarfile

from discretize import TensorMesh

from SimPEG.potential_fields import magnetics
from SimPEG.utils import plot2Ddata, surface2ind_topo
from SimPEG import (
    maps, data, inverse_problem, data_misfit, regularization, optimization,
    directives, inversion, utils
)

# sphinx_gallery_thumbnail_number = 3

#############################################
# Load Data and Plot
# ------------------
#
# File paths for assets we are loading. To set up the inversion, we require
# topography and field observations. The true model defined on the whole mesh
# is loaded to compare with the inversion result. These files are stored as a
# tar-file on our google cloud bucket:
# "https://storage.googleapis.com/simpeg/doc-assets/magnetics.tar.gz"
#

# storage bucket where we have the data
data_source = "https://storage.googleapis.com/simpeg/doc-assets/magnetics.tar.gz"

# download the data
downloaded_data = utils.download(data_source, overwrite=True)

# unzip the tarfile
tar = tarfile.open(downloaded_data, "r")
tar.extractall()
tar.close()

# path to the directory containing our data
dir_path = downloaded_data.split(".")[0] + os.path.sep

# files to work with
topo_filename = dir_path + 'magnetics_topo.txt'
data_filename = dir_path + 'magnetics_data.obs'
model_filename = dir_path + 'true_model.txt'


#############################################
# Load Data and Plot
# ------------------
#
# Here we load and plot synthetic TMI data. Topography is generally
# defined as an (N, 3) array. TMI data is generally defined with 4 columns:
# x, y, z and data.
#

xyz_topo = np.loadtxt(str(topo_filename))
dobs = np.loadtxt(str(data_filename))

receiver_locations = dobs[:, 0:3]
dobs = dobs[:, -1]

# Plot
fig = plt.figure(figsize=(6, 5))
v_max = np.max(np.abs(dobs))

ax1 = fig.add_axes([0.1, 0.1, 0.75, 0.85])
plot2Ddata(
    receiver_locations, dobs, ax=ax1, ncontour=30, clim=(-v_max, v_max),
    contourOpts={"cmap": "bwr"}
)
ax1.set_title('TMI Anomaly')
ax1.set_xlabel('x (m)')
ax1.set_ylabel('y (m)')

ax2 = fig.add_axes([0.85, 0.05, 0.05, 0.9])
norm = mpl.colors.Normalize(
    vmin=-np.max(np.abs(dobs)), vmax=np.max(np.abs(dobs))
)
cbar = mpl.colorbar.ColorbarBase(
    ax2, norm=norm, orientation='vertical', cmap=mpl.cm.bwr
)
cbar.set_label('$nT$', rotation=270, labelpad=15, size=12)

plt.show()

#############################################
# Assign Uncertainty
# ------------------
#
# Inversion with SimPEG requires that we define uncertainties on our data. The
# uncertainty represents our estimate of the standard deviation of the noise on
# our data. For magnetic inversions, a constant floor value is generall applied to
# all data. For this tutorial, the uncertainty on each datum will be 2% of the
# maximum observed gravity anomaly value.
#

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
receiver_list = magnetics.receivers.Point(
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

# Define background susceptibility model in SI. Don't make this 0!
# Otherwise the gradient for the 1st iteration is zero and the inversion will
# not converge.
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
# Here, we define the physics of the magnetics problem by using the simulation
# class.
#

# Define the problem. Define the cells below topography and the mapping
simulation = magnetics.simulation.Simulation3DIntegral(
    survey=survey, mesh=mesh,
    modelType='susceptibility', chiMap=model_map,
    actInd=ind_active
)


#######################################################################
# Define Inverse Problem
# ----------------------
#
# The inverse problem is defined by 3 things:
#
#     1) Data Misfit: a measure of how well our recovered model explains the field data
#     2) Regularization: constraints placed on the recovered model and a priori information
#     3) Optimization: the numerical approach used to solve the inverse problem
#

# Define the data misfit. Here the data misfit is the L2 norm of the weighted
# residual between the observed data and the data predicted for a given model.
# The weighting is defined by the reciprocal of the uncertainties.
dmis = data_misfit.L2DataMisfit(data=data_object, simulation=simulation)
dmis.W = utils.sdiag(1/uncertainties)

# Define the regularization (model objective function)
reg = regularization.Sparse(
    mesh, indActive=ind_active, mapping=model_map, mref=starting_model, gradientType='total',
    alpha_s=1, alpha_x=1, alpha_y=1, alpha_z=1
)

# Define sparse and blocky norms p, qx, qy, qz
reg.norms = np.c_[0, 2, 2, 2]

# Define how the optimization problem is solved. Here we will use a projected
# Gauss-Newton approach that employs the conjugate gradient solver.
opt = optimization.ProjectedGNCG(
    maxIter=10, lower=0., upper=1.,
    maxIterLS=20, maxIterCG=10, tolCG=1e-3
)

# Here we define the inverse problem that is to be solved
inv_prob = inverse_problem.BaseInvProblem(dmis, reg, opt)

#######################################################################
# Define Inversion Directives
# ---------------------------
#
# Here we define any directiveas that are carried out during the inversion. This
# includes the cooling schedule for the trade-off parameter (beta), stopping
# criteria for the inversion and saving inversion results at each iteration.
#

# Defining a starting value for the trade-off parameter (beta) between the data
# misfit and the regularization.
starting_beta = directives.BetaEstimate_ByEig(beta0_ratio=1)

# Options for outputting recovered models and predicted data for each beta.
save_iteration = directives.SaveOutputEveryIteration(save_txt=False)

# Defines the directives for the IRLS regularization. This includes setting
# the cooling schedule for the trade-off parameter.
update_IRLS = directives.Update_IRLS(
    f_min_change=1e-4, max_irls_iterations=30,
    coolEpsFact=1.5, beta_tol=1e-2,
)

# Updating the preconditionner if it is model dependent.
update_jacobi = directives.UpdatePreconditioner()

# Setting a stopping criteria for the inversion.
target_misfit = directives.TargetMisfit(chifact=1)

# Add sensitivity weights
sensitivity_weights = directives.UpdateSensitivityWeights(everyIter=False)

# The directives are defined as a list.
directives_list = [
    sensitivity_weights, starting_beta, save_iteration, update_IRLS, update_jacobi,
]

#####################################################################
# Running the Inversion
# ---------------------
#
# To define the inversion object, we need to define the inversion problem and
# the set of directives. We can then run the inversion.
#

# Here we combine the inverse problem and the set of directives
inv = inversion.BaseInversion(inv_prob, directives_list)

# Print target misfit to compare with convergence
# print("Target misfit is " + str(target_misfit.target))

# Run the inversion
recovered_model = inv.run(starting_model)

############################################################
# Plotting True Model and Recovered Model
# ---------------------------------------
#

# Load the true model (was defined on the whole mesh) and extract only the
# values on active cells.
true_model = np.loadtxt(str(model_filename))
true_model = true_model[ind_active]

# Plot True Model
fig = plt.figure(figsize=(9, 4))
plotting_map = maps.InjectActiveCells(mesh, ind_active, np.nan)

ax1 = fig.add_axes([0.08, 0.1, 0.75, 0.8])
mesh.plotSlice(
    plotting_map*true_model, normal='Y', ax=ax1, ind=int(mesh.nCy/2), grid=True,
    clim=(np.min(true_model), np.max(true_model)), pcolorOpts={'cmap': 'viridis'}
)
ax1.set_title('Model slice at y = 0 m')

ax2 = fig.add_axes([0.85, 0.1, 0.05, 0.8])
norm = mpl.colors.Normalize(vmin=np.min(true_model), vmax=np.max(true_model))
cbar = mpl.colorbar.ColorbarBase(
    ax2, norm=norm, orientation='vertical', cmap=mpl.cm.viridis, format='%.1e'
)
cbar.set_label('SI', rotation=270, labelpad=15, size=12)

plt.show()

# Plot Recovered Model
fig = plt.figure(figsize=(9, 4))
plotting_map = maps.InjectActiveCells(mesh, ind_active, np.nan)

ax1 = fig.add_axes([0.08, 0.1, 0.75, 0.8])
mesh.plotSlice(
    plotting_map*recovered_model, normal='Y', ax=ax1, ind=int(mesh.nCy/2), grid=True,
    clim=(np.min(recovered_model), np.max(recovered_model)), pcolorOpts={'cmap': 'viridis'}
)
ax1.set_title('Model slice at y = 0 m')

ax2 = fig.add_axes([0.85, 0.1, 0.05, 0.8])
norm = mpl.colors.Normalize(vmin=np.min(recovered_model), vmax=np.max(recovered_model))
cbar = mpl.colorbar.ColorbarBase(
    ax2, norm=norm, orientation='vertical', cmap=mpl.cm.viridis, format='%.1e'
)
cbar.set_label('SI',rotation=270, labelpad=15, size=12)

plt.show()

###################################################################
# Plotting Predicted Data and Misfit
# ----------------------------------
#

# Predicted data with final recovered model
dpred = inv_prob.dpred

# Observed data | Predicted data | Normalized data misfit
data_array = np.c_[dobs, dpred, (dobs-dpred)/uncertainties]

fig = plt.figure(figsize=(17, 4))
plot_title=['Observed', 'Predicted', 'Normalized Misfit']
plot_units=['nT', 'nT', '']

ax1 = 3*[None]
ax2 = 3*[None]
norm = 3*[None]
cbar = 3*[None]
cplot = 3*[None]
v_lim = [
    np.max(np.abs(dobs)), np.max(np.abs(dobs)), np.max(np.abs(data_array[:, 2]))
]

for ii in range(0, 3):

    ax1[ii] = fig.add_axes([0.33*ii+0.03, 0.11, 0.25, 0.84])
    cplot[ii] = plot2Ddata(
        receiver_list[0].locations, data_array[:, ii], ax=ax1[ii], ncontour=30,
        clim=(-v_lim[ii], v_lim[ii]), contourOpts={"cmap": "bwr"}
    )
    ax1[ii].set_title(plot_title[ii])
    ax1[ii].set_xlabel('x (m)')
    ax1[ii].set_ylabel('y (m)')

    ax2[ii] = fig.add_axes([0.33*ii+0.27, 0.11, 0.01, 0.84])
    norm[ii] = mpl.colors.Normalize(vmin=-v_lim[ii], vmax=v_lim[ii])
    cbar[ii] = mpl.colorbar.ColorbarBase(
        ax2[ii], norm=norm[ii], orientation='vertical', cmap=mpl.cm.bwr
    )
    cbar[ii].set_label(plot_units[ii], rotation=270, labelpad=15, size=12)

plt.show()
