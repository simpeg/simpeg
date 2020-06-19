# -*- coding: utf-8 -*-
"""
Inverting for Conductivity and Layers
=====================================

Here we use the module *SimPEG.electromangetics.natural_source* to invert
MT sounding data and recover both the resistivities and layer thicknesses
for a 1D layered Earth. In this tutorial, we focus on the following:

    - How to define sources and receivers from a survey file
    - How to define the survey
    - Defining a model that consists of resistivities and layer thicknesses

For this tutorial, we will invert sounding data collected over a layered Earth using
a Wenner array. The end product is layered Earth model which explains the data.


"""


#########################################################################
# Import modules
# --------------
#

from SimPEG.electromagnetics import natural_source as nsem
from SimPEG.electromagnetics.static.utils.static_utils import plot_layer
from SimPEG import (maps, data, data_misfit, regularization,
    optimization, inverse_problem, inversion, directives
    )

from discretize import TensorMesh
from discretize.utils import mkvc
import numpy as np
import matplotlib.pyplot as plt
import os

# sphinx_gallery_thumbnail_number = 2

#############################################
# Define File Names
# -----------------
#
# Here we provide the file paths to assets we need to run the inversion. The
# Path to the true model is also provided for comparison with the inversion
# results.
#

data_filename = os.path.dirname(nsem.__file__) + '\\..\\..\\..\\tutorials\\assets\\mt1d\\app_res_1d_data.dobs'
model_filename = os.path.dirname(nsem.__file__) + '\\..\\..\\..\\tutorials\\assets\\mt1d\\true_model.txt'
mesh_filename = os.path.dirname(nsem.__file__) + '\\..\\..\\..\\tutorials\\assets\\mt1d\\layers.txt'


#############################################
# Load Data, Define Survey and Plot
# ---------------------------------
#
# Here we load the observed data (apparent resistivity), define the MT survey
# and plot the data values.
#

# Load data (only extract apparent resistivity data)
dobs = np.loadtxt(str(data_filename))
frequencies = dobs[:, 0]
dobs = dobs[:, 1]

# Define receivers for real and imaginary impedance measurements
receivers_list = [nsem.receivers.AnalyticReceiver1D(component='app_res')]

# Use a list to define the planewave source at each frequency and assign receivers
source_list = []
for ii in range(0, len(frequencies)):

    source_list.append(nsem.sources.AnalyticPlanewave1D(receivers_list, frequencies[ii]))

# Define the survey object
survey = nsem.survey.Survey1D(source_list)

# Define a data vector for impedances organized in same way as receivers
dobs = mkvc(dobs.T)

# Plot impedance data
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.loglog(frequencies, dobs, 'b')
ax1.set_xlabel("Frequency (Hz)")
ax1.set_ylabel("Apparent Resistivity ($\Omega m$)")
ax1.set_title("Observed Data")
plt.show()

###############################################
# Assign Uncertainties
# --------------------
#
# Inversion with SimPEG requires that we define uncertainties on our data. The
# uncertainty represents our estimate of the standard deviation of the noise on
# our data. For MT sounding data, a percent uncertainty is applied to each datum.
# For this tutorial, the uncertainty on each datum will be 5%.
#

uncertainties = 0.05*np.abs(dobs)


###############################################
# Define Data
# --------------------
#
# Here is where we define the data that are inverted. The data are defined by
# the survey, the observation values and the uncertainties.
#

data_object = data.Data(survey, dobs=dobs, noise_floor=uncertainties)


###################################################
# Defining the Starting Model and Mapping
# ---------------------------------------
#
# In this case, the model consists of parameters which define the respective
# conductivities and thickness for a set of horizontal layer. Here, we choose to
# define a model consisting of 3 layers.
#

# Define the resistivities and thicknesses for the starting model. The thickness
# of the bottom layer is assumed to extend downward to infinity so we don't
# need to define it.
starting_conductivities = np.r_[0.002, 0.005, 0.0005]
starting_thicknesses = np.r_[150., 100.]

# Define a mesh for plotting and regularization.
inv_mesh = TensorMesh([(np.r_[starting_thicknesses, starting_thicknesses[-1]])], '0')
print(inv_mesh)

# Define model. We are inverting for the layer conductivities and layer thicknesses.
# Since the bottom layer extends to infinity, it is not a model parameter for
# which we need to invert. For a 3 layer model, there is a total of 5 parameters.
# For stability, our model is the log-conductivity and log-thickness.
starting_model = np.r_[np.log(starting_conductivities), np.log(starting_thicknesses)]

# Since the model contains two different properties for each layer, we use
# wire maps to distinguish the properties.
wire_map = maps.Wires(('sigma', inv_mesh.nC), ('t', inv_mesh.nC-1))
conductivity_map = maps.ExpMap(nP=inv_mesh.nC) * wire_map.sigma
layer_map = maps.ExpMap(nP=inv_mesh.nC-1) * wire_map.t


#######################################################################
# Define the Physics
# ------------------
#
# Here we define the physics of the problem using the Simulation1DRecursive class.
#

simulation_inv = nsem.simulation_1d.Simulation1DRecursive(
    survey=survey, thicknessesMap=layer_map, sigmaMap=conductivity_map
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
#

# Define the data misfit. Here the data misfit is the L2 norm of the weighted
# residual between the observed data and the data predicted for a given model.
# The weighting is defined by the reciprocal of the uncertainties.
dmis = data_misfit.L2DataMisfit(simulation=simulation_inv, data=data_object)
dmis.W = 1./uncertainties

# Define the regularization on the parameters related to resistivity
mesh_sigma = TensorMesh([inv_mesh.hx.size])
reg_sigma = regularization.Simple(
    mesh_sigma, alpha_s=0.0001, alpha_x=1,
    mapping=wire_map.sigma
)

# Define the regularization on the parameters related to layer thickness
mesh_t = TensorMesh([inv_mesh.hx.size-1])
reg_t = regularization.Simple(
    mesh_t, alpha_s=0.0001, alpha_x=1,
    mapping=wire_map.t
)

# Combine to make regularization for the inversion problem
reg = reg_sigma + reg_t

# Define how the optimization problem is solved. Here we will use an inexact
# Gauss-Newton approach that employs the conjugate gradient solver.
opt = optimization.InexactGaussNewton(
    maxIter=25, maxIterCG=20
)

# Define the inverse problem
inv_prob = inverse_problem.BaseInvProblem(dmis, reg, opt)


#######################################################################
# Define Inversion Directives
# ---------------------------
#
# Here we define any directives that are carried out during the inversion. This
# includes the cooling schedule for the trade-off parameter (beta), stopping
# criteria for the inversion and saving inversion results at each iteration.
#

# Defining a starting value for the trade-off parameter (beta) between the data
# misfit and the regularization.
starting_beta = directives.BetaEstimate_ByEig(beta0_ratio=1e0)

# Set the rate of reduction in trade-off parameter (beta) each time the
# the inverse problem is solved. And set the number of Gauss-Newton iterations
# for each trade-off paramter value.
beta_schedule = directives.BetaSchedule(coolingFactor=5., coolingRate=3.)

# Options for outputting recovered models and predicted data for each beta.
save_iteration = directives.SaveOutputEveryIteration(save_txt=False)

# Setting a stopping criteria for the inversion.
target_misfit = directives.TargetMisfit(chifact=1)

# The directives are defined as a list.
directives_list = [
    starting_beta,
    beta_schedule,
    save_iteration, target_misfit
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

# Run the inversion
recovered_model = inv.run(starting_model)

############################################################
# Examining the Results
# ---------------------
#

# Load the true model and layer thicknesses
true_model = np.loadtxt(str(model_filename))
true_layers = np.loadtxt(str(mesh_filename))
true_layers = TensorMesh([true_layers], 'N')

# Plot true model and recovered model
fig = plt.figure(figsize=(6, 4))
plotting_mesh = TensorMesh([np.r_[layer_map*recovered_model, starting_thicknesses[-1]]], '0')
x_min = np.min([np.min(conductivity_map*recovered_model), np.min(true_model)])
x_max = np.max([np.max(conductivity_map*recovered_model), np.max(true_model)])

ax1 = fig.add_axes([0.2, 0.15, 0.7, 0.7])
plot_layer(true_model, true_layers, ax=ax1, depth_axis=False, color='b')
plot_layer(conductivity_map*recovered_model, plotting_mesh, ax=ax1, depth_axis=False, color='r')
ax1.set_xlim(0.9*x_min, 1.1*x_max)
ax1.set_xlabel("Conductivity [S/m]")
ax1.set_ylabel("Depth (m)")
ax1.legend(['True Model', 'Recovered Model'])

# Plot the true and apparent resistivities on a sounding curve
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.loglog(frequencies, inv_prob.dpred, 'b')
ax1.loglog(frequencies, dobs, 'b*')
ax1.set_xlabel("Frequencies (Hz)")
ax1.set_ylabel("Apparent Resistivity ($\Omega m$)")
ax1.legend(['Observed Sounding Curve', 'True Sounding Curve'])
plt.show()


