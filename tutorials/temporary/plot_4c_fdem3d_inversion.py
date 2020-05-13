
"""
Basic 3D Frequency-Domain Inversion
===================================

Here we invert gravity anomaly data to recover a density contrast model. We
formulate the inverse problem as a least-squares optimization problem. For
this tutorial, we focus on the following:

    - Defining the survey from xyz formatted data
    - Generating a mesh based on survey geometry
    - Including surface topography
    - Defining the inverse problem (data misfit, regularization, optimization)
    - Specifying directives for the inversion
    - Plotting the recovered model and data misfit

Although we consider gravity anomaly data in this tutorial, the same approach
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

from discretize import TreeMesh
from discretize.utils import refine_tree_xyz

from SimPEG.utils import plot2Ddata, surface2ind_topo, mkvc
from SimPEG.electromagnetics import frequency_domain as fdem
from SimPEG import (
    maps, data, data_misfit, inverse_problem, regularization, optimization,
    directives, inversion, utils
    )

try:
    from pymatsolver import Pardiso as Solver
except ImportError:
    from SimPEG import SolverLU as Solver

# sphinx_gallery_thumbnail_number = 3

#############################################
# Define File Names
# -----------------
#
# File paths for assets we are loading. To set up the inversion, we require
# topography and field observations. The true model defined on the whole mesh
# is loaded to compare with the inversion result.
#

topo_filename = os.path.dirname(fdem.__file__) + '\\..\\..\\..\\tutorials\\assets\\fdem\\fdem_topo.txt'
data_filename = os.path.dirname(fdem.__file__) + '\\..\\..\\..\\tutorials\\assets\\fdem\\fdem_data.obs'
model_filename = os.path.dirname(fdem.__file__) + '\\..\\..\\..\\tutorials\\assets\\fdem\\true_model.txt'


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
frequencies = dobs[:, 0]
receiver_locations = dobs[:, 1:4]
dobs_real = dobs[:, 4]
dobs_imag = dobs[:, 5]

# Plot the data
unique_frequencies = np.unique(frequencies)
frequency_index = 0
k = frequencies==unique_frequencies[frequency_index]

fig = plt.figure(figsize=(10, 4))

# Real Component
v_max = np.max(np.abs(dobs_real[k]))
ax1 = fig.add_axes([0.05, 0.05, 0.35, 0.9])
plot2Ddata(
    receiver_locations[k, 0:2], dobs_real[k], ax=ax1,
    ncontour=30, clim=(-v_max, v_max), contourOpts={"cmap": "RdBu_r"}
    )
ax1.set_title('Re[$B_z$] at 200 Hz')

ax2 = fig.add_axes([0.41, 0.05, 0.02, 0.9])
norm = mpl.colors.Normalize(vmin=-v_max, vmax=v_max)
cbar = mpl.colorbar.ColorbarBase(
    ax2, norm=norm, orientation='vertical', cmap=mpl.cm.RdBu_r
)
cbar.set_label('$T$', rotation=270, labelpad=15, size=12)

# Imaginary Component
v_max = np.max(np.abs(dobs_imag[k]))
ax1 = fig.add_axes([0.55, 0.05, 0.35, 0.9])
plot2Ddata(
    receiver_locations[k, 0:2], dobs_imag[k], ax=ax1,
    ncontour=30, clim=(-v_max, v_max), contourOpts={"cmap": "RdBu_r"}
)
ax1.set_title('Im[$B_z$] at 200 Hz')

ax2 = fig.add_axes([0.91, 0.05, 0.02, 0.9])
norm = mpl.colors.Normalize(vmin=-v_max, vmax=v_max)
cbar = mpl.colorbar.ColorbarBase(
    ax2, norm=norm, orientation='vertical', cmap=mpl.cm.RdBu_r
)
cbar.set_label('$T$', rotation=270, labelpad=15, size=12)

plt.show()

#############################################
# Assign Uncertainties
# --------------------
#
# Inversion with SimPEG requires that we define uncertainties on our data. The
# uncertainty represents our estimate of the standard deviation of the noise on
# our data. For gravity inversion, a constant floor value is generall applied to
# all data. For this tutorial, the uncertainty on each datum will be 1% of the
# maximum observed gravity anomaly value.
#

uncertainties_real = 1e-14*np.ones(len(dobs_real))
uncertainties_imag = 1e-14*np.ones(len(dobs_imag))


#uncertainties_real = 1e-20 + 0.05*np.abs(dobs_real)
#uncertainties_imag = 1e-20 + 0.05*np.abs(dobs_imag)

#############################################
# Defining the Survey
# -------------------
#
# Here, we define survey that will be used for this tutorial. Gravity
# surveys are simple to create. The user only needs an (N, 3) array to define
# the xyz locations of the observation locations. From this, the user can
# define the receivers and the source field.
#

source_list = []  # Create empty list to store sources

# Each unique location and frequency defines a new transmitter
n_data = len(dobs_real)
for ii in range(n_data):

    # Define receivers of different type at each location
    bzr_receiver = fdem.receivers.PointMagneticFluxDensitySecondary(
            receiver_locations[ii, :], 'z', 'imag'
            )
    bzi_receiver = fdem.receivers.PointMagneticFluxDensitySecondary(
            receiver_locations[ii, :], 'z', 'imag'
            )
    receivers_list = [bzr_receiver, bzi_receiver]
    
    # Must define the transmitter properties and associated receivers
    source_location = receiver_locations[ii, :] + np.c_[0, 0, 20]
    
    source_list.append(
        fdem.sources.MagDipole(
            receivers_list, frequencies[ii], source_location, orientation='z'
        )
    )
    
survey = fdem.Survey(source_list)

#############################################
# Defining the Data
# -----------------
#
# Here is where we define the data that are inverted. The data are defined by
# the survey, the observation values and the uncertainties.
#

mu0 = 4*np.pi*1e-7
dobs = mkvc(np.c_[dobs_real, dobs_imag].T)
uncertainties = mkvc(np.c_[uncertainties_real, uncertainties_imag].T)


data_object = data.Data(survey, dobs=dobs, noise_floor=uncertainties)


###############################################################
# Create OcTree Mesh
# ------------------
#
# Here we define the OcTree mesh that is used for this example.
# We chose to design a coarser mesh to decrease the run time.
# When designing a mesh to solve practical frequency domain problems:
# 
#     - Your smallest cell size should be 10%-20% the size of your smallest skin depth
#     - The thickness of your padding needs to be 2-3 times biggest than your largest skin depth
#     - The skin depth is ~500*np.sqrt(rho/f)
#
#

dh = 25.                                                     # base cell width
dom_width = 3000.                                            # domain width
nbc = 2**int(np.round(np.log(dom_width/dh)/np.log(2.)))      # num. base cells

# Define the base mesh
h = [(dh, nbc)]
mesh = TreeMesh([h, h, h], x0='CCC')

# Mesh refinement based on topography
mesh = refine_tree_xyz(
    mesh, xyz_topo, octree_levels=[0, 0, 0, 1], method='surface', finalize=False
)

# Mesh refinement near transmitters and receivers
mesh = refine_tree_xyz(
    mesh, receiver_locations, octree_levels=[2, 4], method='radial', finalize=False
)

# Refine core mesh region
xp, yp, zp = np.meshgrid([-250., 250.], [-250., 250.], [-250., 0.])
xyz = np.c_[mkvc(xp), mkvc(yp), mkvc(zp)]
mesh = refine_tree_xyz(
    mesh, xyz, octree_levels=[0, 2, 4], method='box', finalize=False
)

mesh.finalize()

########################################################
# Starting/Reference Model and Mapping on OcTree Mesh
# ---------------------------------------------------
#
# Here, we would create starting and/or reference models for the DC inversion as
# well as the mapping from the model space to the active cells. Starting and
# reference models can be a constant background value or contain a-priori
# structures. Here, the starting model is the natural log of 0.01 S/m.
#

# Define conductivity model in S/m
air_conductivity = np.log(1e-8)
background_conductivity = np.log(1e-2)

# Find the indecies of the active cells in forward model (ones below surface)
ind_active = surface2ind_topo(mesh, xyz_topo)

active_map = maps.InjectActiveCells(mesh, ind_active, np.exp(air_conductivity))
nC = int(ind_active.sum())

model_map = active_map*maps.ExpMap()

# Define model
starting_model = background_conductivity*np.ones(nC)


##############################################
# Define the Physics
# ------------------
#
# Here, we define the physics of the gravity problem by using the simulation
# class.
# 

simulation = fdem.simulation.Simulation3DMagneticFluxDensity(
        mesh, survey=survey, sigmaMap=model_map, Solver=Solver
        )


#######################################################################
# Define the Inverse Problem
# --------------------------
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
reg = regularization.Simple(
    mesh, indActive=ind_active, mref=starting_model,
    alpha_s=1e-2, alpha_x=1, alpha_y=1, alpha_z=1
    )

# Define how the optimization problem is solved. Here we will use a projected
# Gauss-Newton approach that employs the conjugate gradient solver.
#opt = optimization.ProjectedGNCG(
#    maxIterCG=5, tolCG=1e-2, lower=-10, upper=5
#    )
opt = optimization.InexactGaussNewton(
    maxIterCG=5, tolCG=1e-2
    )

# Here we define the inverse problem that is to be solved
inv_prob = inverse_problem.BaseInvProblem(
    dmis, reg, opt
    )

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
starting_beta = directives.BetaEstimate_ByEig(beta0_ratio=10)

# Defining the fractional decrease in beta and the number of Gauss-Newton solves
# for each beta value.
beta_schedule = directives.BetaSchedule(coolingFactor=10, coolingRate=3)

# Options for outputting recovered models and predicted data for each beta.
save_iteration = directives.SaveOutputEveryIteration(save_txt=False)

# Setting a stopping criteria for the inversion.
target_misfit = directives.TargetMisfit(chifact=1)

# The directives are defined as a list.
directives_list = [
        starting_beta, beta_schedule, save_iteration, target_misfit
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

# Run inversion
recovered_model = inv.run(starting_model)


############################################################
# Plotting True Model and Recovered Model
# ---------------------------------------
#

# Load the true model (was defined on the whole mesh) and extract only the
# values on active cells.
true_model = np.loadtxt(str(model_filename))
true_model = np.log(true_model[ind_active])

# Plot True Model
fig = plt.figure(figsize=(9, 4))
plotting_map = maps.InjectActiveCells(mesh, ind_active, np.nan)

ax1 = fig.add_axes([0.1, 0.1, 0.73, 0.8])
mesh.plotSlice(
    plotting_map*true_model, normal='Y', ax=ax1, ind=int(mesh.hy.size/2), grid=True,
    clim=(np.min(true_model), np.max(true_model)), pcolorOpts={'cmap': 'jet'}
    )
ax1.set_title('Model slice at y = 0 m')


ax2 = fig.add_axes([0.85, 0.1, 0.05, 0.8])
norm = mpl.colors.Normalize(vmin=np.min(true_model), vmax=np.max(true_model))
cbar = mpl.colorbar.ColorbarBase(
    ax2, norm=norm, orientation='vertical', cmap=mpl.cm.jet, format='%.1f'
)
cbar.set_label(
    '$S/m$',
    rotation=270, labelpad=15, size=12
)

plt.show()

# Plot Recovered Model
fig = plt.figure(figsize=(9, 4))
plotting_map = maps.InjectActiveCells(mesh, ind_active, np.nan)

ax1 = fig.add_axes([0.1, 0.1, 0.73, 0.8])
mesh.plotSlice(
    plotting_map*recovered_model, normal='Z', ax=ax1, ind=int(mesh.hz.size/2-1), grid=True,
    clim=(np.min(recovered_model), np.max(recovered_model)), pcolorOpts={'cmap': 'jet'}
)
ax1.set_title('Model slice at y = 0 m')

ax2 = fig.add_axes([0.85, 0.1, 0.05, 0.8])
norm = mpl.colors.Normalize(vmin=np.min(recovered_model), vmax=np.max(recovered_model))
cbar = mpl.colorbar.ColorbarBase(
        ax2, norm=norm, orientation='vertical', cmap=mpl.cm.jet, format='%.1f'
        )
cbar.set_label('$S/m$',rotation=270, labelpad=15, size=12)

plt.show()

####################################################################
# Plotting Predicted Data and Normalized Misfit
# ---------------------------------------------
#

# Predicted data with final recovered model
dpred = inv_prob.dpred
misfits = (dobs-dpred)/uncertainties


dpred_real = dpred[0:len(dpred):2]
dpred_imag = dpred[1:len(dpred):2]
misfits_real = misfits[0:len(misfits):2]
misfits_imag = misfits[1:len(misfits):2]

# Observed data | Predicted data | Normalized data misfit
data_array = np.c_[dobs_imag[k], dpred_imag[k], misfits_imag[k]]

fig = plt.figure(figsize=(17, 4))
plot_title=['Observed', 'Predicted', 'Normalized Misfit']
plot_units=['mgal', 'mgal', '']

ax1 = 3*[None]
ax2 = 3*[None]
norm = 3*[None]
cbar = 3*[None]
cplot = 3*[None]
v_lim = [
    np.max(np.abs(dobs_imag[k])), np.max(np.abs(dobs_imag[k])), np.max(np.abs(misfits_imag[k]))
]

for ii in range(0, 3):
    
    ax1[ii] = fig.add_axes([0.33*ii+0.03, 0.11, 0.23, 0.84])
    cplot[ii] = plot2Ddata(
        receiver_locations[k], data_array[:, ii], ax=ax1[ii], ncontour=30,
        clim=(-v_lim[ii], v_lim[ii]), contourOpts={"cmap": "RdBu_r"}
    )
    ax1[ii].set_title(plot_title[ii])
    ax1[ii].set_xlabel('x (m)')
    ax1[ii].set_ylabel('y (m)')
    
    ax2[ii] = fig.add_axes([0.33*ii+0.27, 0.11, 0.01, 0.85])
    norm[ii] = mpl.colors.Normalize(vmin=-v_lim[ii], vmax=v_lim[ii])
    cbar[ii] = mpl.colorbar.ColorbarBase(
        ax2[ii], norm=norm[ii], orientation='vertical', cmap=mpl.cm.RdBu_r
    )
    cbar[ii].set_label(plot_units[ii], rotation=270, labelpad=15, size=12)

plt.show()

