# -*- coding: utf-8 -*-
"""
DC Resistivity Sounding
=======================

Here we use the module *SimPEG.electromangetics.static.resistivity* to predict
DC resistivity data. In this tutorial, we focus on the following:

    - How to define sources and receivers
    - How to define the survey
    - How to predict voltage or apparent resistivity data
    - The units of the model and resulting data

For this tutorial, we will simulation sounding data over a layered Earth using
a Wenner array. The end product is a sounding curve which tells us how the
electrical resistivity changes with depth.
    

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

from SimPEG import (maps, data, data_misfit, regularization,
    optimization, inverse_problem, inversion, directives
    )
from SimPEG.electromagnetics.static import resistivity as dc
from SimPEG.electromagnetics.static.utils.StaticUtils import plot_layer

# sphinx_gallery_thumbnail_number = 2


#############################################
# Load Data, Define Survey and Plot
# ---------------------------------
#
# Here we load and plot synthetic DCIP data. We a
#

# File names
data_filename = os.path.dirname(dc.__file__) + '\\..\\..\\..\\..\\tutorials\\assets\\dcip1d\\app_res_1d_data.dobs'
model_filename = os.path.dirname(dc.__file__) + '\\..\\..\\..\\..\\tutorials\\assets\\dcip1d\\true_model.txt'
mesh_filename = os.path.dirname(dc.__file__) + '\\..\\..\\..\\..\\tutorials\\assets\\dcip1d\\layers.txt'


# Load data
dobs = np.loadtxt(str(data_filename))

a_electrodes = dobs[:, 0:3]
b_electrodes = dobs[:, 3:6]
m_electrodes = dobs[:, 6:9]
n_electrodes = dobs[:, 9:12]
dobs = dobs[:, -1]

# Define survey
unique_tx, k = np.unique(np.c_[a_electrodes, b_electrodes], axis=0, return_index=True)
n_tx = len(k)
k=np.sort(k)
k = np.r_[k, len(k)+1]

source_list = []
for ii in range(0, n_tx):
    
    m_locations = m_electrodes[k[ii]:k[ii+1], :]
    n_locations = n_electrodes[k[ii]:k[ii+1], :]
    receiver_list = [dc.receivers.Dipole(m_locations, n_locations)]
    
    a_location = a_electrodes[k[ii], :]
    b_location = b_electrodes[k[ii], :]
    source_list.append(dc.sources.Dipole(receiver_list, a_location, b_location))

# Define survey
survey = dc.Survey(source_list)

# Plot the data
survey.getABMN_locations()
electrode_separations = np.sqrt(
        np.sum((survey.m_locations - survey.n_locations)**2, axis=1)
        )

# Plot apparent resistivities on sounding curve
fig = plt.figure(figsize=(11, 5))

ax1 = fig.add_axes([0.05, 0.05, 0.8, 0.9])
ax1.semilogy(electrode_separations, dobs)

plt.show()

###############################################
# Assign Uncertainties
# --------------------

uncertainties = 0.025*dobs


###############################################
# Define Data
# --------------------

data_object = data.Data(survey, dobs=dobs, noise_floor=uncertainties)


###############################################
# Defining a 1D Layered Earth (1D Tensor Mesh)
# --------------------------------------------
#
# Here, we define the layer thicknesses for our 1D simulation. To do this, we use
# the TensorMesh class.

layer_thicknesses = 5.*np.ones((80))
mesh = TensorMesh([layer_thicknesses], '0')

print(mesh)

###############################################################
# Create Conductivity Model and Mapping for OcTree Mesh
# -----------------------------------------------------
#
# Here we define the resistivity model that will be used to predict DC data.
# For each layer in our 1D Earth, we must provide a resistivity value. For a
# 1D simulation, we assume the bottom layer extends to infinity.
#

# Define model. A resistivity (Ohm meters) or conductivity (S/m) for each layer.
starting_model = np.log(1e3*np.ones((len(layer_thicknesses))))

# Define mapping from model to active cells.
model_map = maps.IdentityMap(mesh)*maps.ExpMap()
# inversion_map = maps.IdentityMap(mesh)

#plot_layer(model_map*model, mesh)


#######################################################################
# Predict DC Resistivity Data
# ---------------------------
#
# Here we predict DC resistivity data. If the keyword argument *sigmaMap* is
# defined, the simulation will expect a conductivity model. If the keyword
# argument *rhoMap* is defined, the simulation will expect a resistivity model.
#

simulation = dc.simulation_1d.DCSimulation_1D(
        mesh, survey=survey, rhoMap=model_map, t=layer_thicknesses,
        data_type="apparent_resistivity"
        )









dmis = data_misfit.L2DataMisfit(simulation=simulation, data=data_object)
dmis.W = 1./uncertainties




reg_rho = regularization.Simple(
    mesh, alpha_s=1., alpha_x=1., mref=starting_model,
    # mapping=inversion_map
)

#reg_rho = regularization.Tikhonov(
#    mesh, alpha_s=0.2, alpha_x=1., mref=starting_model,
#    # mapping=inversion_map
#)


# Create model weights based on sensitivity matrix (sensitivity weighting)
wr = np.sum(simulation.getJ(starting_model)**2, axis=0)**0.5
wr = (wr/np.max(np.abs(wr)))
reg_rho.cell_weights = wr  # include in regularization
#directives.UpdateSensitivityWeights(JtJdiag=wr)  # Don't think this is applied to this type of problem

#reg = reg_rho + reg_t
opt = optimization.InexactGaussNewton(
    maxIter=30, maxIterCG=20
)
invProb = inverse_problem.BaseInvProblem(dmis, reg_rho, opt)



# Create an inversion object
starting_beta = directives.BetaEstimate_ByEig(beta0_ratio=1e0)
beta_schedule = directives.BetaSchedule(coolingFactor=5., coolingRate=3.)
target_misfit = directives.TargetMisfit(chifact=1.)

#save =  directives.SaveOutputDictEveryIteration()

directives_list = [starting_beta, beta_schedule, target_misfit]

inv = inversion.BaseInversion(invProb, directiveList=directives_list)

MOPT = []

recovered_model = inv.run(starting_model)

#from SimPEG.electromagnetics.static.utils.StaticUtils import plot_layer
fig = plt.figure(figsize=(5, 5))

true_model = np.loadtxt(str(model_filename))
true_layers = np.loadtxt(str(mesh_filename))
true_layers = TensorMesh([true_layers], 'N')

ax1 = fig.add_axes([0.05, 0.05, 0.8, 0.9])
plot_layer(model_map*recovered_model, mesh, ax=ax1, depth_axis=False)
plot_layer(true_model, true_layers, ax=ax1, depth_axis=False, color='r')


# Plot apparent resistivities on sounding curve
fig = plt.figure(figsize=(11, 5))

ax1 = fig.add_axes([0.05, 0.05, 0.8, 0.9])
ax1.semilogy(electrode_separations, dobs)
ax1.semilogy(electrode_separations, invProb.dpred)

plt.show()





