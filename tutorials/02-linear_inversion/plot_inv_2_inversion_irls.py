"""
Sparse Inversion with Iteratively Re-Weighted Least-Squares
===========================================================

Least-squares inversion produces smooth models which may not be an accurate
representation of the true model. Here we demonstrate the basics of inverting 
for sparse and/or blocky models. Here, we used the iteratively reweighted
least-squares approach. For this tutorial, we focus on the following:

    - Defining the forward problem
    - Defining the inverse problem (data misfit, regularization, optimization)
    - Defining the paramters for the IRLS algorithm
    - Specifying directives for the inversion
    - Recovering a set of model parameters which explains the observations


"""

from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

from discretize import TensorMesh

from SimPEG.simulation import LinearSimulation
from SimPEG.data import Data
from SimPEG import (
    simulation,
    maps,
    data_misfit,
    directives,
    optimization,
    regularization,
    inverse_problem,
    inversion,
)

# sphinx_gallery_thumbnail_number = 3

#############################################
# Defining the Model and Mapping
# ------------------------------
#
# Here we generate a synthetic model and a mappig which goes from the model
# space to the row space of our linear operator.
#

nParam = 100  # Number of model paramters

# A 1D mesh is used to define the row-space of the linear operator.
mesh = TensorMesh([nParam])

# Creating the true model
true_model = np.zeros(mesh.nC)
true_model[mesh.vectorCCx > 0.3] = 1.0
true_model[mesh.vectorCCx > 0.45] = -0.5
true_model[mesh.vectorCCx > 0.6] = 0

# Mapping from the model space to the row space of the linear operator
model_map = maps.IdentityMap(mesh)

# Plotting the true model
fig = plt.figure(figsize=(8, 5))
ax = fig.add_subplot(111)
ax.plot(mesh.vectorCCx, true_model, "b-")
ax.set_ylim([-2, 2])

#############################################
# Defining the Linear Operator
# ----------------------------
#
# Here we define the linear operator with dimensions (nData, nParam). In practive,
# you may have a problem-specific linear operator which you would like to construct
# or load here.
#

# Number of data observations (rows)
nData = 20

# Create the linear operator for the tutorial. The columns of the linear operator
# represents a set of decaying and oscillating functions.
jk = np.linspace(1.0, 60.0, nData)
p = -0.25
q = 0.25


def g(k):
    return np.exp(p * jk[k] * mesh.vectorCCx) * np.cos(
        np.pi * q * jk[k] * mesh.vectorCCx
    )


G = np.empty((nData, nParam))

for i in range(nData):
    G[i, :] = g(i)

# Plot the columns of G
fig = plt.figure(figsize=(8, 5))
ax = fig.add_subplot(111)
for i in range(G.shape[0]):
    ax.plot(G[i, :])

ax.set_title("Columns of matrix G")


#############################################
# Defining the Simulation
# -----------------------
#
# The simulation defines the relationship between the model parameters and
# predicted data.
#

sim = simulation.LinearSimulation(mesh, G=G, model_map=model_map)


#############################################
# Predict Synthetic Data
# ----------------------
#
# Here, we use the true model to create synthetic data which we will subsequently
# invert.
#

# Standard deviation of Gaussian noise being added
std = 0.02
np.random.seed(1)

# Create a SimPEG data object
data_obj = sim.make_synthetic_data(true_model, noise_floor=std, add_noise=True)

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
# Within the data misfit, the residual between predicted and observed data are
# normalized by the data's standard deviation.
dmis = data_misfit.L2DataMisfit(simulation=sim, data=data_obj)

# Define the regularization (model objective function). Here, 'p' defines the
# the norm of the smallness term and 'q' defines the norm of the smoothness
# term.
reg = regularization.Sparse(mesh, mapping=model_map)
reg.reference_model = np.zeros(nParam)
p = 0.0
q = 0.0
reg.norms = [p, q]

# Define how the optimization problem is solved.
opt = optimization.ProjectedGNCG(
    maxIter=100, lower=-2.0, upper=2.0, maxIterLS=20, maxIterCG=30, tolCG=1e-4
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

# Add sensitivity weights but don't update at each beta
sensitivity_weights = directives.UpdateSensitivityWeights(everyIter=False)

# Reach target misfit for L2 solution, then use IRLS until model stops changing.
IRLS = directives.Update_IRLS(max_irls_iterations=40, minGNiter=1, f_min_change=1e-4)

# Defining a starting value for the trade-off parameter (beta) between the data
# misfit and the regularization.
starting_beta = directives.BetaEstimate_ByEig(beta0_ratio=1e0)

# Update the preconditionner
update_Jacobi = directives.UpdatePreconditioner()

# Save output at each iteration
saveDict = directives.SaveOutputEveryIteration(save_txt=False)

# Define the directives as a list
directives_list = [sensitivity_weights, IRLS, starting_beta, update_Jacobi, saveDict]


#####################################################################
# Setting a Starting Model and Running the Inversion
# --------------------------------------------------
#
# To define the inversion object, we need to define the inversion problem and
# the set of directives. We can then run the inversion.
#

# Here we combine the inverse problem and the set of directives
inv = inversion.BaseInversion(inv_prob, directives_list)

# Starting model
starting_model = 1e-4 * np.ones(nParam)

# Run inversion
recovered_model = inv.run(starting_model)

#####################################################################
# Plotting Results
# ----------------
#

fig, ax = plt.subplots(1, 2, figsize=(12 * 1.2, 4 * 1.2))

# True versus recovered model
ax[0].plot(mesh.vectorCCx, true_model, "k-")
ax[0].plot(mesh.vectorCCx, inv_prob.l2model, "b-")
ax[0].plot(mesh.vectorCCx, recovered_model, "r-")
ax[0].legend(("True Model", "Recovered L2 Model", "Recovered Sparse Model"))
ax[0].set_ylim([-2, 2])

# Observed versus predicted data
ax[1].plot(data_obj.dobs, "k-")
ax[1].plot(inv_prob.dpred, "ko")
ax[1].legend(("Observed Data", "Predicted Data"))

# Plot convergence
fig = plt.figure(figsize=(9, 5))
ax = fig.add_axes([0.2, 0.1, 0.7, 0.85])
ax.plot(saveDict.phi_d, "k", lw=2)

twin = ax.twinx()
twin.plot(saveDict.phi_m, "k--", lw=2)
ax.plot(np.r_[IRLS.iterStart, IRLS.iterStart], np.r_[0, np.max(saveDict.phi_d)], "k:")
ax.text(
    IRLS.iterStart,
    0.0,
    "IRLS Start",
    va="bottom",
    ha="center",
    rotation="vertical",
    size=12,
    bbox={"facecolor": "white"},
)

ax.set_ylabel("$\phi_d$", size=16, rotation=0)
ax.set_xlabel("Iterations", size=14)
twin.set_ylabel("$\phi_m$", size=16, rotation=0)
