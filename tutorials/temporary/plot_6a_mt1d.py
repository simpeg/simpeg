# -*- coding: utf-8 -*-
"""
1D Magnetotelluric Sounding
===========================

Here we use the module *SimPEG.electromangetics.natural_source* to predict
magnetotelluric data over a layered Earth. In this tutorial, we focus on the
following:

    - General definition of sources and receivers
    - How to define the survey
    - How to predict impedances, apparent resistivity or phase data
    - The units of the model and resulting data
    - 1D simulation for magnetotellurics

For this tutorial, we will simulate sounding data over a layered Earth.

"""

#########################################################################
# Import modules
# --------------
#

from SimPEG.electromagnetics import natural_source as nsem
from SimPEG.electromagnetics.static.utils.static_utils import plot_layer
from SimPEG import maps

from discretize import TensorMesh

import numpy as np
import matplotlib.pyplot as plt
import os

save_file = False

# sphinx_gallery_thumbnail_number = 2


#####################################################################
# Create Survey
# -------------
#
# Here we demonstrate a general way to define sources and receivers.
# For the receivers, you choose one of 4 data type options: 'real', 'imag',
# 'app_res' or 'phase'. The source is a planewave whose frequency must be
# defined in Hz.
#

# Frequencies being measured
frequencies = np.logspace(0, 4, 21)

# Define a receiver for each data type as a list
receivers_list = [
        nsem.receivers.AnalyticReceiver1D(component='real'),
        nsem.receivers.AnalyticReceiver1D(component='imag'),
        nsem.receivers.AnalyticReceiver1D(component='app_res'),
        nsem.receivers.AnalyticReceiver1D(component='phase')
        ]

# Use a list to define the planewave source at each frequency and assign receivers
source_list = []
for ii in range(0, len(frequencies)):

    source_list.append(nsem.sources.AnalyticPlanewave1D(receivers_list, frequencies[ii]))

# Define the survey object
survey = nsem.survey.Survey1D(source_list)

###############################################
# Defining a 1D Layered Earth Model
# ---------------------------------
#
# Here, we define the layer thicknesses and electrical conductivities for our
# 1D simulation. If we have N layers, we define N electrical conductivity
# values and N-1 layer thicknesses. The lowest layer is assumed to extend to
# infinity.
#

# Layer thicknesses
layer_thicknesses = np.array([200, 200])

# Layer conductivities
model = np.array([0.001, 0.01, 0.001])

# Define a mapping for conductivities
model_mapping = maps.IdentityMap()

###############################################################
# Plot Resistivity Model
# ----------------------
#
# Here we plot the 1D conductivity model.
#

# Define a 1D mesh for plotting. Provide a maximum depth for the plot.
max_depth = 600
plotting_mesh = TensorMesh(
    [np.r_[layer_thicknesses, max_depth-layer_thicknesses.sum()]]
    )

# Plot the 1D model
fig = plt.figure()
ax1 = fig.add_subplot(111)
plot_layer(model_mapping*model, plotting_mesh, ax=ax1, depth_axis=False, color='b')
ax1.set_xlabel("Conductivity [S/m]")
ax1.set_ylabel("Depth (m)")

#######################################################################
# Define the Forward Simulation and Predict MT Data
# -------------------------------------------------
#
# Here we predict MT data. If the keyword argument *rhoMap* is
# defined, the simulation will expect a resistivity model. If the keyword
# argument *sigmaMap* is defined, the simulation will expect a conductivity model.
#

simulation = nsem.simulation_1d.Simulation1DRecursive(
    survey=survey, thicknesses=layer_thicknesses, sigmaMap=model_mapping
)

# Predict data for a given model
dpred = simulation.dpred(model)


#######################################################################
# Plotting the data
# -------------------------------------------------
#
# We simulated impedances, apparent resistivities and phases. Here, we plot
# those quantities and display the units.
#

# Number of data types we simulated
ntype = 4

# impedance data
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.loglog(frequencies, np.abs(dpred[0::ntype]), 'b')
ax1.loglog(frequencies, np.abs(dpred[1::ntype]), 'r')
ax1.set_xlabel("Frequency (Hz)")
ax1.set_ylabel("Impedance (V/A)")
ax1.set_title("Real and Imaginary Impedances")
ax1.legend(['Re[Z]','Im[Z]'])
plt.show()

# apparent resistivity and phase
fig = plt.figure(figsize=(11, 4))
ax1 = fig.add_subplot(121)
ax1.loglog(frequencies, np.abs(dpred[2::ntype]), 'b')
ax1.set_xlabel("Frequency (Hz)")
ax1.set_ylabel("Apparent Resistivity ($\Omega m$)")
ax1.set_title("Apparent Resistivity")
ax2 = fig.add_subplot(122)
ax2.semilogx(frequencies, np.abs(dpred[3::ntype]), 'r')
ax2.set_xlabel("Frequency (Hz)")
ax2.set_ylabel("Phase (deg)")
ax2.set_title("Phase")
plt.show()



#########################################################################
# Optional: Export Data
# ---------------------
#

if save_file == True:

    # Extract and at 5% noise to impedance data
    d_imp = np.c_[dpred[0::ntype], dpred[1::ntype]]
    d_imp = d_imp + 0.05*np.abs(d_imp)*np.random.rand(np.shape(d_imp)[0],2)
    
    # Extract and at 5% noise to apparent resistivity data
    d_res = dpred[2::ntype]
    d_res = d_res + 0.05*np.abs(d_res)*np.random.rand(len(d_imp))
    d_phs = dpred[3::ntype]
    d_phs = d_phs + 2.*np.random.rand(len(d_phs))
    
    # Write files
    fname = os.path.dirname(nsem.__file__) + '\\..\\..\\..\\tutorials\\assets\\mt1d\\impedance_1d_data.dobs'
    np.savetxt(fname, np.c_[survey.frequencies, d_imp], fmt='%.4e')
    
    fname = os.path.dirname(nsem.__file__) + '\\..\\..\\..\\tutorials\\assets\\mt1d\\app_res_1d_data.dobs'
    np.savetxt(fname, np.c_[survey.frequencies, d_res, d_phs], fmt='%.4e')
    
    fname = os.path.dirname(nsem.__file__) + '\\..\\..\\..\\tutorials\\assets\\mt1d\\true_model.txt'
    np.savetxt(fname, model, fmt='%.4e')

    fname = os.path.dirname(nsem.__file__) + '\\..\\..\\..\\tutorials\\assets\\mt1d\\layers.txt'
    np.savetxt(fname, plotting_mesh.hx, fmt='%d')



