# -*- coding: utf-8 -*-
"""
Smooth 1D Inversion
===================

Here we use the module *SimPEG.electromangetics.natural_source* to invert
MT sounding data and recover a 1D electrical conductivity model.
In this tutorial, we focus on the following:

    - How to define sources and receivers from a survey file
    - How to define the survey
    - 1D inversion of impedance data

For this tutorial, we will invert sounding data collected over a layered Earth.
The end product is layered Earth model which explains the data.


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

data_filename = os.path.dirname(nsem.__file__) + '\\..\\..\\..\\tutorials\\assets\\mt1d\\impedance_1d_data.dobs'
model_filename = os.path.dirname(nsem.__file__) + '\\..\\..\\..\\tutorials\\assets\\mt1d\\true_model.txt'
mesh_filename = os.path.dirname(nsem.__file__) + '\\..\\..\\..\\tutorials\\assets\\mt1d\\layers.txt'

#############################################
# Load Data, Define Survey and Plot
# ---------------------------------
#
# Here we load the observed data (impedances), define the MT survey and plot the
# data values.
#

# Load data
dobs = np.loadtxt(str(data_filename))
frequencies = dobs[:, 0]
dobs = dobs[:, 1:]

# Define receivers for real and imaginary impedance measurements
receivers_list = [
        nsem.receivers.AnalyticReceiver1D(component='real'),
        nsem.receivers.AnalyticReceiver1D(component='imag')
        ]

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
ax1.loglog(frequencies, np.abs(dobs[0::2]), 'b')
ax1.loglog(frequencies, np.abs(dobs[1::2]), 'r')
ax1.set_xlabel("Frequency (Hz)")
ax1.set_ylabel("Impedance (V/A)")
ax1.set_title("Observed Data")
ax1.legend(['Re[Z]','Im[Z]'])
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

###############################################################
# Defining a 1D Layered Earth (1D Tensor Mesh)
# --------------------------------------------
#
# Here, we define the layer thicknesses for our 1D simulation. To do this, we use
# the TensorMesh class.
#

# Define layer thicknesses
inv_thicknesses = 10*np.logspace(0,1,25)

# Define a mesh for plotting and regularization.
mesh = TensorMesh([(np.r_[inv_thicknesses, inv_thicknesses[-1]])], '0')

print(mesh)

########################################################
# Define a Starting and Reference Model
# -------------------------------------
#
# Here, we create starting and/or reference models for the inversion as
# well as the mapping from the model space to the active cells. Starting and
# reference models can be a constant background value or contain a-priori
# structures. Here, the starting model is log(0.001) S/m.
#
# Define log-conductivity values for each layer since our model is the
# log-conductivity. Don't make the values 0!
# Otherwise the gradient for the 1st iteration is zero and the inversion will
# not converge.

# Define model. A resistivity (Ohm meters) or conductivity (S/m) for each layer.
starting_model = np.log(0.001*np.ones((len(inv_thicknesses)+1)))

# Define mapping from model to active cells.
model_mapping = maps.IdentityMap(nP=len(starting_model))*maps.ExpMap()

#######################################################################
# Define the Physics
# ------------------
#
# Here we define the physics of the problem using the Simulation1DRecursive class.
#

simulation_inv = nsem.simulation_1d.Simulation1DRecursive(
    survey=survey, thicknesses=inv_thicknesses, sigmaMap=model_mapping, sensitivity_method='2nd_order'
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

# Define the regularization (model objective function)
reg = regularization.Simple(
    mesh, alpha_s=0.0001, alpha_x=1.
)

# Define how the optimization problem is solved. Here we will use an inexact
# Gauss-Newton approach that employs the conjugate gradient solver.
opt = optimization.InexactGaussNewton(
    maxIter=30, maxIterCG=20
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
target_misfit = directives.TargetMisfit(chifact=1.1)

# The directives are defined as a list.
directives_list = [
    starting_beta,
    beta_schedule,
    save_iteration,
    target_misfit
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
x_min = np.min([np.min(model_mapping*recovered_model), np.min(true_model)])
x_max = np.max([np.max(model_mapping*recovered_model), np.max(true_model)])

ax1 = fig.add_axes([0.2, 0.15, 0.7, 0.7])
plot_layer(true_model, true_layers, ax=ax1, depth_axis=False, color='b')
plot_layer(model_mapping*recovered_model, mesh, ax=ax1, depth_axis=False, color='r')
ax1.set_xlim(0.9*x_min, 1.1*x_max)
ax1.set_xlabel("Conductivity [S/m]")
ax1.set_ylabel("Depth (m)")
ax1.legend(['True Model','Recovered Model'])


# Plot real and imaginary impedances
fig = plt.figure(figsize=(11, 4))
ax2 = fig.add_subplot(121)
ax2.loglog(frequencies, np.abs(dobs[0::2]), 'b*')
ax2.loglog(frequencies, np.abs(inv_prob.dpred[0::2]), 'b')
ax2.legend(['Observed','Predicted'])
ax2.set_xlabel('Frequency (Hz)')
ax2.set_ylabel('Re[Z]')

ax3 = fig.add_subplot(122)
ax3.loglog(frequencies, np.abs(dobs[1::2]), 'r*')
ax3.loglog(frequencies, np.abs(inv_prob.dpred[1::2]), 'r')
ax3.legend(['Observed','Predicted'])
ax3.set_xlabel('Frequency (Hz)')
ax3.set_ylabel('Im[Z]')


