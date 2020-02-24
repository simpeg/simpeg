# -*- coding: utf-8 -*-
"""
DC Resistivity: 1D Sounding
===========================

Here we use the module *SimPEG.electromangetics.static.resistivity* to predict
sounding data over a 1D layered Earth. In this tutorial, we focus on the following:

    - General definition of sources and receivers
    - How to define the survey
    - How to predict voltage or apparent resistivity data
    - The units of the model and resulting data
    - 1D simulation for DC resistivity

For this tutorial, we will simulate sounding data over a layered Earth using
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

from SimPEG import maps
from SimPEG.electromagnetics.static import resistivity as dc
from SimPEG.electromagnetics.static.utils.static_utils import plot_layer

# sphinx_gallery_thumbnail_number = 2


#####################################################################
# Create Survey
# -------------
#
# Here we demonstrate a general way to define sources and receivers.
# For pole and dipole sources, we must define the A or AB electrode locations,
# respectively. For the pole and dipole receivers, we must define the M or
# MN electrode locations, respectively.
#

a_min = 20.
a_max = 500.
n_stations = 25

# Define the 'a' spacing for Wenner array measurements for each reading
electrode_separations = np.linspace(a_min, a_max, n_stations)

source_list = []  # create empty array for sources to live

for ii in range(0, len(electrode_separations)):
    
    # Extract separation parameter for sources and receivers
    a = electrode_separations[ii]
    
    # AB electrode locations for source. Each is a (1, 3) numpy array
    A_location = np.r_[-1.5*a, 0., 0.]
    B_location = np.r_[1.5*a, 0., 0.]

    # MN electrode locations for receivers. Each is an (N, 3) numpy array
    M_location = np.r_[-0.5*a, 0., 0.]
    N_location = np.r_[0.5*a, 0., 0.]

    # Create receivers list. Define as pole or dipole.
    receiver_list = dc.receivers.Dipole(
            M_location, N_location
            )
    receiver_list = [receiver_list]

    # Define the source properties and associated receivers
    source_list.append(
            dc.sources.Dipole(receiver_list, A_location, B_location)
            )

# Define survey
survey = dc.Survey(source_list)


###############################################
# Defining a 1D Layered Earth (1D Tensor Mesh)
# --------------------------------------------
#
# Here, we define the layer thicknesses for our 1D simulation. To do this, we use
# the TensorMesh class. The thickness of the n-th layer is just for show. 
# We assume the bottom layer extends to infinity.
#

layer_thicknesses = np.r_[100., 100., 300.]
mesh = TensorMesh([layer_thicknesses])

print(mesh)

###############################################################
# Create Resistivity Model and Mapping
# ------------------------------------
#
# Here we define the resistivity model that will be used to predict DC sounding data.
# For each layer in our 1D Earth, we must provide a resistivity value. For a
# 1D simulation, we assume the bottom layer extends to infinity.
#

# Define model. A resistivity (Ohm meters) for each layer.
model = np.r_[1e3, 4e3, 2e2]

# Define mapping from model to active cells. Here, all layers are use in the
# forward simulation.
model_map = maps.IdentityMap(mesh)

# Plot the 1D model
plot_layer(model_map*model, mesh)

#######################################################################
# Define the Forward Simulation and Predict DC Resistivity Data
# -------------------------------------------------------------
#
# Here we predict DC resistivity data. If the keyword argument *rhoMap* is
# defined, the simulation will expect a resistivity model. If the keyword
# argument *sigmaMap* is defined, the simulation will expect a conductivity model. 
#

simulation = dc.simulation_1d.DCSimulation_1D(
        mesh, survey=survey, rhoMap=model_map, t=layer_thicknesses,
        data_type="apparent_resistivity"
        )

# Predict data for a given model
dpred = simulation.dpred(model)

# Plot apparent resistivities on sounding curve
fig = plt.figure(figsize=(11, 5))
ax1 = fig.add_axes([0.05, 0.05, 0.8, 0.9])
ax1.semilogy(electrode_separations, dpred, 'b')
ax1.set_xlabel("Wenner Array Separation Parameter (m)")
ax1.set_ylabel("Apparent Resistivity ($\Omega m$)")
plt.show()


#########################################################################
# Optional: Export Data
# ---------------------
#

# MAKE A WRITE TO UBC FORMAT HERE

survey.getABMN_locations()

noise = 0.025*dpred*np.random.rand(len(dpred))

data_array = np.c_[
    survey.a_locations,
    survey.b_locations,
    survey.m_locations,
    survey.n_locations,
    dpred + noise
    ]

fname = os.path.dirname(dc.__file__) + '\\..\\..\\..\\..\\tutorials\\assets\\dcip1d\\app_res_1d_data.dobs'
np.savetxt(fname, data_array, fmt='%.4e')


fname = os.path.dirname(dc.__file__) + '\\..\\..\\..\\..\\tutorials\\assets\\dcip1d\\true_model.txt'
np.savetxt(fname, model, fmt='%.4e')


fname = os.path.dirname(dc.__file__) + '\\..\\..\\..\\..\\tutorials\\assets\\dcip1d\\layers.txt'
np.savetxt(fname, mesh.hx, fmt='%d')




