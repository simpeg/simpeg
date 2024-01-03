"""
Compare weighting strategy with Inversion of surface Gravity Anomaly Data
=========================================================================

Here we invert gravity anomaly data to recover a density contrast model. We formulate the inverse problem as an iteratively
re-weighted least-squares (IRLS) optimization problem. For this tutorial, we
focus on the following:

    - Setting regularization weights
    - Defining the survey from xyz formatted data
    - Generating a mesh based on survey geometry
    - Including surface topography
    - Defining the inverse problem (data misfit, regularization, optimization)
    - Specifying directives for the inversion
    - Setting sparse and blocky norms
    - Plotting the recovered model and data misfit

Although we consider gravity anomaly data in this tutorial, the same approach
can be used to invert gradiometry and other types of geophysical data.
"""

#########################################################################
# Import modules
# --------------
#

import os
import tarfile

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from discretize import TensorMesh
from discretize.utils import active_from_xyz

from SimPEG import (
    data,
    data_misfit,
    directives,
    inverse_problem,
    inversion,
    maps,
    optimization,
    regularization,
    utils,
)
from SimPEG.potential_fields import gravity
from SimPEG.utils import model_builder, plot2Ddata

# sphinx_gallery_thumbnail_number = 3

#############################################
# Define File Names
# -----------------
#
# File paths for assets we are loading. To set up the inversion, we require
# topography and field observations. The true model defined on the whole mesh
# is loaded to compare with the inversion result. These files are stored as a
# tar-file on our google cloud bucket:
# "https://storage.googleapis.com/simpeg/doc-assets/gravity.tar.gz"
#

# storage bucket where we have the data
data_source = "https://storage.googleapis.com/simpeg/doc-assets/gravity.tar.gz"

# download the data
downloaded_data = utils.download(data_source, overwrite=True)

# unzip the tarfile
tar = tarfile.open(downloaded_data, "r")
tar.extractall()
tar.close()

# path to the directory containing our data
dir_path = downloaded_data.split(".")[0] + os.path.sep

# files to work with
topo_filename = dir_path + "gravity_topo.txt"
data_filename = dir_path + "gravity_data.obs"


#############################################
# Load Data and Plot
# ------------------
#
# Here we load and plot synthetic gravity anomaly data. Topography is generally
# defined as an (N, 3) array. Gravity data is generally defined with 4 columns:
# x, y, z and data.
#

# Load topography
xyz_topo = np.loadtxt(str(topo_filename))

# Load field data
dobs = np.loadtxt(str(data_filename))

# Define receiver locations and observed data
receiver_locations = dobs[:, 0:3]
dobs = dobs[:, -1]

# Plot
mpl.rcParams.update({"font.size": 12})
fig = plt.figure(figsize=(7, 5))

ax1 = fig.add_axes([0.1, 0.1, 0.73, 0.85])
plot2Ddata(
    receiver_locations,
    dobs,
    ax=ax1,
    contourOpts={"cmap": "bwr"},
    shade=True,
    nx=20,
    ny=20,
    dataloc=True,
)
ax1.set_title("Gravity Anomaly")
ax1.set_xlabel("x (m)")
ax1.set_ylabel("y (m)")

ax2 = fig.add_axes([0.8, 0.1, 0.03, 0.85])
norm = mpl.colors.Normalize(vmin=-np.max(np.abs(dobs)), vmax=np.max(np.abs(dobs)))
cbar = mpl.colorbar.ColorbarBase(
    ax2, norm=norm, orientation="vertical", cmap=mpl.cm.bwr, format="%.1e"
)
cbar.set_label("$mGal$", rotation=270, labelpad=15, size=12)

plt.show()

#############################################
# Assign Uncertainties
# --------------------
#
# Inversion with SimPEG requires that we define the standard deviation of our data.
# This represents our estimate of the noise in our data. For a gravity inversion,
# a constant floor value is generally applied to all data. For this tutorial,
# the standard deviation on each datum will be 1% of the maximum observed
# gravity anomaly value.
#

maximum_anomaly = np.max(np.abs(dobs))

uncertainties = 0.01 * maximum_anomaly * np.ones(np.shape(dobs))

#############################################
# Defining the Survey
# -------------------
#
# Here, we define the survey that will be used for this tutorial. Gravity
# surveys are simple to create. The user only needs an (N, 3) array to define
# the xyz locations of the observation locations. From this, the user can
# define the receivers and the source field.
#

# Define the receivers. The data consists of vertical gravity anomaly measurements.
# The set of receivers must be defined as a list.
receiver_list = gravity.receivers.Point(receiver_locations, components="gz")

receiver_list = [receiver_list]

# Define the source field
source_field = gravity.sources.SourceField(receiver_list=receiver_list)

# Define the survey
survey = gravity.survey.Survey(source_field)

#############################################
# Defining the Data
# -----------------
#
# Here is where we define the data that is inverted. The data is defined by
# the survey, the observation values and the standard deviation.
#

data_object = data.Data(survey, dobs=dobs, standard_deviation=uncertainties)


#############################################
# Defining a Tensor Mesh
# ----------------------
#
# Here, we create the tensor mesh that will be used to invert gravity anomaly
# data. If desired, we could define an OcTree mesh.
#

dh = 5.0
hx = [(dh, 5, -1.3), (dh, 40), (dh, 5, 1.3)]
hy = [(dh, 5, -1.3), (dh, 40), (dh, 5, 1.3)]
hz = [(dh, 5, -1.3), (dh, 15)]
mesh = TensorMesh([hx, hy, hz], "CCN")

########################################################
# Starting/Reference Model and Mapping on Tensor Mesh
# ---------------------------------------------------
#
# Here, we create starting and/or reference models for the inversion as
# well as the mapping from the model space to the active cells. Starting and
# reference models can be a constant background value or contain a-priori
# structures.
#

# Find the indices of the active cells in forward model (ones below surface)
ind_active = active_from_xyz(mesh, xyz_topo)

# Define mapping from model to active cells
nC = int(ind_active.sum())
model_map = maps.IdentityMap(nP=nC)  # model consists of a value for each active cell

# Define and plot starting model
starting_model = np.zeros(nC)


##############################################
# Define the Physics and data misfit
# ----------------------------------
#
# Here, we define the physics of the gravity problem by using the simulation
# class.
#

simulation = gravity.simulation.Simulation3DIntegral(
    survey=survey, mesh=mesh, rhoMap=model_map, ind_active=ind_active
)

# Define the data misfit. Here the data misfit is the L2 norm of the weighted
# residual between the observed data and the data predicted for a given model.
# Within the data misfit, the residual between predicted and observed data are
# normalized by the data's standard deviation.
dmis = data_misfit.L2DataMisfit(data=data_object, simulation=simulation)


#######################################################################
# Running the Depth Weighted inversion
# ------------------------------------
#
# Here we define the directives, weights, regularization, and optimization
# for a depth-weighted inversion
#

# inversion directives
# Defining a starting value for the trade-off parameter (beta) between the data
# misfit and the regularization.
starting_beta = directives.BetaEstimate_ByEig(beta0_ratio=1e0)

# Defines the directives for the IRLS regularization. This includes setting
# the cooling schedule for the trade-off parameter.
update_IRLS = directives.Update_IRLS(
    f_min_change=1e-4,
    max_irls_iterations=30,
    coolEpsFact=1.5,
    beta_tol=1e-2,
)

# Options for outputting recovered models and predicted data for each beta.
save_iteration = directives.SaveOutputEveryIteration(save_txt=False)

# Updating the preconditionner if it is model dependent.
update_jacobi = directives.UpdatePreconditioner()

# The directives are defined as a list
directives_list = [
    update_IRLS,
    starting_beta,
    save_iteration,
    update_jacobi,
]

# Define the regularization (model objective function) with depth weighting.
reg_dpth = regularization.Sparse(mesh, active_cells=ind_active, mapping=model_map)
reg_dpth.norms = [0, 2, 2, 2]
depth_weights = utils.depth_weighting(
    mesh, receiver_locations, active_cells=ind_active, exponent=2
)
reg_dpth.set_weights(depth_weights=depth_weights)

# Define how the optimization problem is solved. Here we will use a projected
# Gauss-Newton approach that employs the conjugate gradient solver.
opt = optimization.ProjectedGNCG(
    maxIter=100, lower=-1.0, upper=1.0, maxIterLS=20, maxIterCG=10, tolCG=1e-3
)

# Here we define the inverse problem that is to be solved
inv_prob = inverse_problem.BaseInvProblem(dmis, reg_dpth, opt)

# Here we combine the inverse problem and the set of directives
inv = inversion.BaseInversion(inv_prob, directives_list)

# Run inversion
recovered_model_dpth = inv.run(starting_model)

#######################################################################
# Running the Distance Weighted inversion
# ---------------------------------------
#
# Here we define the directives, weights, regularization, and optimization
# for a distance-weighted inversion
#

# inversion directives
# Defining a starting value for the trade-off parameter (beta) between the data
# misfit and the regularization.
starting_beta = directives.BetaEstimate_ByEig(beta0_ratio=1e0)

# Defines the directives for the IRLS regularization. This includes setting
# the cooling schedule for the trade-off parameter.
update_IRLS = directives.Update_IRLS(
    f_min_change=1e-4,
    max_irls_iterations=30,
    coolEpsFact=1.5,
    beta_tol=1e-2,
)

# Options for outputting recovered models and predicted data for each beta.
save_iteration = directives.SaveOutputEveryIteration(save_txt=False)

# Updating the preconditionner if it is model dependent.
update_jacobi = directives.UpdatePreconditioner()

# The directives are defined as a list
directives_list = [
    update_IRLS,
    starting_beta,
    save_iteration,
    update_jacobi,
]

# Define the regularization (model objective function) with distance weighting.
reg_dist = regularization.Sparse(mesh, active_cells=ind_active, mapping=model_map)
reg_dist.norms = [0, 2, 2, 2]
distance_weights = utils.distance_weighting(
    mesh, receiver_locations, active_cells=ind_active, exponent=2
)
reg_dist.set_weights(distance_weights=distance_weights)

# Define how the optimization problem is solved. Here we will use a projected
# Gauss-Newton approach that employs the conjugate gradient solver.
opt = optimization.ProjectedGNCG(
    maxIter=100, lower=-1.0, upper=1.0, maxIterLS=20, maxIterCG=10, tolCG=1e-3
)

# Here we define the inverse problem that is to be solved
inv_prob = inverse_problem.BaseInvProblem(dmis, reg_dist, opt)

# Here we combine the inverse problem and the set of directives
inv = inversion.BaseInversion(inv_prob, directives_list)

# Run inversion
recovered_model_dist = inv.run(starting_model)

#######################################################################
# Running the Distance Weighted inversion
# ---------------------------------------
#
# Here we define the directives, weights, regularization, and optimization
# for a sensitivity weighted inversion
#

# inversion directives
# Defining a starting value for the trade-off parameter (beta) between the data
# misfit and the regularization.
starting_beta = directives.BetaEstimate_ByEig(beta0_ratio=1e0)

# Defines the directives for the IRLS regularization. This includes setting
# the cooling schedule for the trade-off parameter.
update_IRLS = directives.Update_IRLS(
    f_min_change=1e-4,
    max_irls_iterations=30,
    coolEpsFact=1.5,
    beta_tol=1e-2,
)

# Options for outputting recovered models and predicted data for each beta.
save_iteration = directives.SaveOutputEveryIteration(save_txt=False)

# Updating the preconditionner if it is model dependent.
update_jacobi = directives.UpdatePreconditioner()

# Add sensitivity weights
sensitivity_weights = directives.UpdateSensitivityWeights(every_iteration=False)

# The directives are defined as a list
directives_list = [
    update_IRLS,
    sensitivity_weights,
    starting_beta,
    save_iteration,
    update_jacobi,
]

# Define the regularization (model objective function) for sensitivity weighting.
reg_sensw = regularization.Sparse(mesh, active_cells=ind_active, mapping=model_map)
reg_sensw.norms = [0, 2, 2, 2]

# Define how the optimization problem is solved. Here we will use a projected
# Gauss-Newton approach that employs the conjugate gradient solver.
opt = optimization.ProjectedGNCG(
    maxIter=100, lower=-1.0, upper=1.0, maxIterLS=20, maxIterCG=10, tolCG=1e-3
)

# Here we define the inverse problem that is to be solved
inv_prob = inverse_problem.BaseInvProblem(dmis, reg_sensw, opt)

# Here we combine the inverse problem and the set of directives
inv = inversion.BaseInversion(inv_prob, directives_list)

# Run inversion
recovered_model_sensw = inv.run(starting_model)

############################################################
# Recreate True Model
# -------------------
#

# Define density contrast values for each unit in g/cc
background_density = 0.0
block_density = -0.2
sphere_density = 0.2

# Define model. Models in SimPEG are vector arrays.
true_model = background_density * np.ones(nC)

# You could find the indicies of specific cells within the model and change their
# value to add structures.
ind_block = (
    (mesh.gridCC[ind_active, 0] > -50.0)
    & (mesh.gridCC[ind_active, 0] < -20.0)
    & (mesh.gridCC[ind_active, 1] > -15.0)
    & (mesh.gridCC[ind_active, 1] < 15.0)
    & (mesh.gridCC[ind_active, 2] > -50.0)
    & (mesh.gridCC[ind_active, 2] < -30.0)
)
true_model[ind_block] = block_density

# You can also use SimPEG utilities to add structures to the model more concisely
ind_sphere = model_builder.get_indices_sphere(
    np.r_[35.0, 0.0, -40.0], 15.0, mesh.gridCC
)
ind_sphere = ind_sphere[ind_active]
true_model[ind_sphere] = sphere_density


############################################################
# Plotting True Model and Recovered Models
# ----------------------------------------
#

# Plot Models
fig, ax = plt.subplots(2, 2, figsize=(20, 10), sharex=True, sharey=True)
ax = ax.flatten()
plotting_map = maps.InjectActiveCells(mesh, ind_active, np.nan)
cmap = "coolwarm"
slice_y_loc = 0.0

mm = mesh.plot_slice(
    plotting_map * true_model,
    normal="Y",
    ax=ax[0],
    grid=False,
    slice_loc=slice_y_loc,
    pcolor_opts={"cmap": cmap, "norm": norm},
)
ax[0].set_title(f"True model slice at y = {slice_y_loc} m")
plt.colorbar(mm[0], label="$g/cm^3$", ax=ax[0])

# plot depth weighting result
vmax = np.abs(recovered_model_dpth).max()
norm = mpl.colors.TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax)
mm = mesh.plot_slice(
    plotting_map * recovered_model_dpth,
    normal="Y",
    ax=ax[1],
    grid=False,
    slice_loc=slice_y_loc,
    pcolor_opts={"cmap": cmap, "norm": norm},
)
ax[1].set_title(f"Depth weighting Model slice at y = {slice_y_loc} m")
plt.colorbar(mm[0], label="$g/cm^3$", ax=ax[1])

# plot distance weighting result
vmax = np.abs(recovered_model_dist).max()
norm = mpl.colors.TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax)
mm = mesh.plot_slice(
    plotting_map * recovered_model_dist,
    normal="Y",
    ax=ax[2],
    grid=False,
    slice_loc=slice_y_loc,
    pcolor_opts={"cmap": cmap, "norm": norm},
)
ax[2].set_title(f"Distance weighting Model slice at y = {slice_y_loc} m")
plt.colorbar(mm[0], label="$g/cm^3$", ax=ax[2])

# plot sensitivity weighting result
vmax = np.abs(recovered_model_sensw).max()
norm = mpl.colors.TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax)
mm = mesh.plot_slice(
    plotting_map * recovered_model_sensw,
    normal="Y",
    ax=ax[3],
    grid=False,
    slice_loc=slice_y_loc,
    pcolor_opts={"cmap": cmap, "norm": norm},
)
ax[3].set_title(f"Sensitivity weighting Model slice at y = {slice_y_loc} m")
plt.colorbar(mm[0], label="$g/cm^3$", ax=ax[3])

# shared plotting
plotting_map = maps.InjectActiveCells(mesh, ind_active, 0.0)
slice_y_ind = (
    mesh.cell_centers[:, 1] == np.abs(mesh.cell_centers[:, 1] - slice_y_loc).min()
)
for axx in ax:
    utils.plot2Ddata(
        mesh.cell_centers[slice_y_ind][:, [0, 2]],
        (plotting_map * true_model)[slice_y_ind],
        contourOpts={"alpha": 0},
        level=True,
        ncontour=2,
        levelOpts={"colors": "grey", "linewidths": 2, "linestyles": "--"},
        method="nearest",
        ax=axx,
    )
    axx.set_aspect(1)

plt.tight_layout()

############################################################
# Visualize weights
# -----------------
#
# Plot Weights
fig, ax = plt.subplots(1, 3, figsize=(20, 4), sharex=True, sharey=True)
plotting_map = maps.InjectActiveCells(mesh, ind_active, np.nan)
cmap = "magma"
slice_y_loc = 0.0

# plot depth weights
mm = mesh.plot_slice(
    plotting_map * np.log10(depth_weights),
    normal="Y",
    ax=ax[0],
    grid=False,
    slice_loc=slice_y_loc,
    pcolor_opts={"cmap": cmap},
)
ax[0].set_title(f"log10(depth weights) slice at y = {slice_y_loc} m")
plt.colorbar(mm[0], label="log10(depth weights)", ax=ax[0])

# plot distance weights
mm = mesh.plot_slice(
    plotting_map * np.log10(distance_weights),
    normal="Y",
    ax=ax[1],
    grid=False,
    slice_loc=slice_y_loc,
    pcolor_opts={"cmap": cmap},
)
ax[1].set_title(f"log10(distance weights) slice at y = {slice_y_loc} m")
plt.colorbar(mm[0], label="log10(distance weights)", ax=ax[1])

# plot sensitivity weights
mm = mesh.plot_slice(
    plotting_map * np.log10(reg_sensw.objfcts[0].get_weights(key="sensitivity")),
    normal="Y",
    ax=ax[2],
    grid=False,
    slice_loc=slice_y_loc,
    pcolor_opts={"cmap": cmap},
)
ax[2].set_title(f"log10(sensitivity weights) slice at y = {slice_y_loc} m")
plt.colorbar(mm[0], label="log10(sensitivity weights)", ax=ax[2])

# shared plotting
for axx in ax:
    axx.set_aspect(1)

plt.tight_layout()
