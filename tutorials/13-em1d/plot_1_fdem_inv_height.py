"""
Inversion of 1D Frequency-Domain Data
==============================================


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

import simpegEM1D as em1d
from simpegEM1D.utils import get_vertical_discretization_frequency, plotLayer
from SimPEG.utils import mkvc
from SimPEG import (
    maps, data, data_misfit, inverse_problem, regularization, optimization,
    directives, inversion, utils
    )

plt.rcParams.update({'font.size': 16, 'lines.linewidth': 2, 'lines.markersize':8})

# sphinx_gallery_thumbnail_number = 3

#############################################
# Define File Names
# -----------------
#
# File paths for assets we are loading. To set up the inversion, we require
# topography and field observations. The true model defined on the whole mesh
# is loaded to compare with the inversion result.
#

data_filename = os.path.dirname(em1d.__file__) + '\\..\\tutorials\\assets\\em1dfm_data.obs'



#############################################
# Load Data and Plot
# ------------------
#
# Here we load and plot synthetic gravity anomaly data. Topography is generally
# defined as an (N, 3) array. Gravity data is generally defined with 4 columns:
# x, y, z and data.
#

# Load field data
dobs = np.loadtxt(str(data_filename))

# Define receiver locations and observed data
frequencies = dobs[:, 0]
dobs = mkvc(dobs[:, 1:])

fig, ax = plt.subplots(1,1, figsize = (7, 7))
ax.loglog(frequencies, np.abs(dobs[0:len(frequencies)]), 'k-o', lw=3)
ax.loglog(frequencies, np.abs(dobs[len(frequencies):]), 'k:o', lw=3)
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("|Hs/Hp| (ppm)")
ax.set_title("Magnetic Field as a Function of Frequency")
ax.legend(["Real", "Imaginary"])



#############################################
# Defining the Survey
# -------------------

source_location = np.array([0., 0., 30.]) 
source_current = 1.
source_radius = 5.
moment_amplitude=1.

source_receiver_offset = np.array([10., 0., 0.])
receiver_orientation = "z"
field_type = "ppm"

# Receiver list
receiver_list = []
receiver_list.append(
    em1d.receivers.HarmonicPointReceiver(
        source_receiver_offset, frequencies, orientation=receiver_orientation,
        field_type=field_type, component="real", use_source_receiver_offset=True
    )
)
receiver_list.append(
    em1d.receivers.HarmonicPointReceiver(
        source_receiver_offset, frequencies, orientation=receiver_orientation,
        field_type=field_type, component="imag", use_source_receiver_offset=True
    )
)
    
source_list = [
    em1d.sources.HarmonicMagneticDipoleSource(
        receiver_list=receiver_list, location=source_location, orientation="z",
        moment_amplitude=moment_amplitude
    )
]

# Survey
survey = em1d.survey.EM1DSurveyFD(source_list)


#############################################
# Assign Uncertainties
# --------------------
#
#

uncertainties = 0.05*np.abs(dobs)*np.ones(np.shape(dobs))


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

# Based on estimate of background conductivity, make layers

#inv_thicknesses = get_vertical_discretization_frequency(
#    frequencies, sigma_background=0.1,
#    factor_fmax=20, factor_fmin=1., n_layer=50,
#)

inv_thicknesses = np.logspace(0,1.5,25)

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
starting_conductivity = np.log(0.1*np.ones(mesh.nC))
starting_height = 25.
starting_model = np.r_[starting_conductivity, starting_height]

# Define mapping from model to active cells.
wires = maps.Wires(('sigma', mesh.nC),('h', 1))
sigma_map = maps.ExpMap() * wires.sigma
h_map = wires.h



#######################################################################
# Define the Physics
# ------------------
#

simulation = em1d.simulation.EM1DFMSimulation(
    survey=survey, thicknesses=inv_thicknesses, sigmaMap=sigma_map, hMap=h_map
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
dmis = data_misfit.L2DataMisfit(simulation=simulation, data=data_object)
dmis.W = 1./uncertainties




# Define the regularization (model objective function)
reg_sigma = regularization.Sparse(
    mesh, mapping=wires.sigma,
#    alpha_s=1,
)

# Define sparse and blocky norms p, q
p = 0.
q = 0.
reg_sigma.norms = np.c_[p, q]

#reg.eps_p = 1e-3
#reg.eps_q = 1e-3

reg_height = regularization.Sparse(
    TensorMesh([1]), mapping=wires.h,
)

p = 0.
reg_sigma.p = p

reg = reg_sigma + reg_height
reg.mref = starting_model

# Define how the optimization problem is solved. Here we will use an inexact
# Gauss-Newton approach that employs the conjugate gradient solver.
opt = optimization.ProjectedGNCG(maxIter=100, maxIterLS=20, maxIterCG=20, tolCG=1e-3)

# Define the inverse problem
inv_prob = inverse_problem.BaseInvProblem(dmis, reg, opt)


#######################################################################
# Define Inversion Directives
# ---------------------------
#
# Here we define any directiveas that are carried out during the inversion. This
# includes the cooling schedule for the trade-off parameter (beta), stopping
# criteria for the inversion and saving inversion results at each iteration.
#

# Apply and update sensitivity weighting as the model updates
#sensitivity_weights = directives.UpdateSensitivityWeights()

# Reach target misfit for L2 solution, then use IRLS until model stops changing.
#IRLS = directives.Update_IRLS(max_irls_iterations=40, minGNiter=1, f_min_change=1e-5, chifact_start=2)
#IRLS = directives.Update_IRLS(
#    max_irls_iterations=20, minGNiter=1, fix_Jmatrix=True, coolingRate=2, 
#    beta_tol=1e-2, f_min_change=1e-5,
#    chifact_start = 1.
#)

# Defining a starting value for the trade-off parameter (beta) between the data
# misfit and the regularization.
starting_beta = directives.BetaEstimate_ByEig(beta0_ratio=1e1)

# Update the preconditionner
update_Jacobi = directives.UpdatePreconditioner()

# Options for outputting recovered models and predicted data for each beta.
save_iteration = directives.SaveOutputEveryIteration(save_txt=False)

# The directives are defined as a list.
#directives_list = [
#    IRLS,
#    starting_beta,
#    save_iteration,
#]





update_IRLS = directives.Update_IRLS(
    max_irls_iterations=30, minGNiter=1,
    coolEpsFact=1.5, update_beta=True
)

# Updating the preconditionner if it is model dependent.
update_jacobi = directives.UpdatePreconditioner()

# Setting a stopping criteria for the inversion.
#target_misfit = directives.TargetMisfit(chifact=1)

# Add sensitivity weights
sensitivity_weights = directives.UpdateSensitivityWeights()

# The directives are defined as a list.
directives_list = [
    sensitivity_weights,
    starting_beta,
    save_iteration,
    update_IRLS,
    update_jacobi,
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


#####################################################################
# Plotting Results
# ---------------------


# Load the true model and layer thicknesses
true_model = np.array([0.1, 1., 0.1])
hz = np.r_[20., 40., 160.]
true_layers = TensorMesh([hz])

# Extract Least-Squares model
l2_model = inv_prob.l2model

# Plot true model and recovered model
fig = plt.figure(figsize=(8, 9))
x_min = np.min(np.r_[sigma_map * recovered_model, sigma_map * l2_model, true_model])
x_max = np.max(np.r_[sigma_map * recovered_model, sigma_map * l2_model, true_model])

ax1 = fig.add_axes([0.2, 0.15, 0.7, 0.7])
plotLayer(true_model, true_layers, ax=ax1, showlayers=False, color="k")
plotLayer(sigma_map * l2_model, mesh, ax=ax1, showlayers=False, color="b")
plotLayer(sigma_map * recovered_model, mesh, ax=ax1, showlayers=False, color="r")
ax1.set_xlim(0.01, 10)
ax1.legend(["True Model", "L2-Model", "Sparse Model"])

# Plot the true and apparent resistivities on a sounding curve
dpred_l2 = simulation.dpred(l2_model)
dpred_final = simulation.dpred(recovered_model)

fig = plt.figure(figsize=(11, 6))
ax1 = fig.add_axes([0.2, 0.1, 0.6, 0.8])
ax1.loglog(frequencies, np.abs(dobs[0:len(frequencies)]), "k-o")
ax1.loglog(frequencies, np.abs(dobs[len(frequencies):]), "k:o")
ax1.loglog(frequencies, np.abs(dpred_l2[0:len(frequencies)]), "b-o")
ax1.loglog(frequencies, np.abs(dpred_l2[len(frequencies):]), "b:o")
ax1.loglog(frequencies, np.abs(dpred_final[0:len(frequencies)]), "r-o")
ax1.loglog(frequencies, np.abs(dpred_final[len(frequencies):]), "r:o")
ax1.set_xlabel("Frequencies (Hz)")
ax1.set_ylabel("|Hs/Hp| (ppm)")
ax1.legend([
    "Observed (real)", "Observed (imag)",
    "L2-Model (real)", "L2-Model (imag)",
    "Sparse (real)", "Sparse (imag)"],
    loc="upper left"
)
plt.show()
























