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

from SimPEG import maps
from SimPEG.electromagnetics.static import resistivity as dc
from SimPEG.electromagnetics.static.utils.StaticUtils import plot_layer

# sphinx_gallery_thumbnail_number = 2


#####################################################################
# Create Survey
# -------------
#
# Here we define the sources and receivers.
# For the source, we must define the AB electrode locations. For the receivers
# we must define the MN electrode locations. When creating DCIP surveys, it
# is important for the electrode locations NOT to lie within air cells. Here
# we shift the vertical locations of the electrodes down by a constant. The
# user may choose to do something more complex.
#

# Define all electrode locations (Src and Rx) as an (N, 2) numpy array
electrode_separations = np.linspace(10., 400., 20)  # Number of electrode locations along EW profile

source_list = []  # create empty array for sources to live

for ii in range(0, len(electrode_separations)):
    
    a = electrode_separations[ii]
    
    # AB electrode locations for source
    a_location = np.r_[-1.5*a, 0., 0.]
    b_location = np.r_[1.5*a, 0., 0.]

    # MN electrode locations for receivers
    m_location = np.c_[-0.5*a, 0., 0.]
    n_location = np.c_[0.5*a, 0., 0.]

    # Create receivers list. Define as pole or dipole. Can choose to
    # measured potential or components of electric field.
    receiver_list = dc.receivers.Dipole(
            m_location, n_location
            )
    receiver_list = [receiver_list]

    # Define the source properties and associated receivers
    source_list.append(
            dc.sources.Dipole(receiver_list, a_location, b_location)
            )

# Define survey
survey = dc.Survey(source_list)


###############################################
# Defining a 1D Layered Earth (1D Tensor Mesh)
# --------------------------------------------
#
# Here, we define the layer thicknesses for our 1D simulation. To do this, we use
# the TensorMesh class.

layer_thicknesses = np.r_[50., 100., 100.]
mesh = TensorMesh([layer_thicknesses], 'N')

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
model = np.r_[1e3, 1e4, 1e2]

# Define mapping from model to active cells.
model_map = maps.IdentityMap(mesh)

plot_layer(model_map*model, mesh)


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

dpred = simulation.dpred(model)

# Plot apparent resistivities on sounding curve
fig = plt.figure(figsize=(11, 5))

ax1 = fig.add_axes([0.05, 0.05, 0.8, 0.9])
ax1.semilogy(electrode_separations, dpred)

plt.show()


#########################################################################
# Optional: Export Data
# ---------------------
#

# MAKE A WRITE TO UBC FORMAT HERE

survey.getABMN_locations()

data_array = np.c_[
    survey.a_locations,
    survey.b_locations,
    survey.m_locations,
    survey.n_locations,
    dpred*(1 + 0.05*np.random.rand(len(dpred)))
    ]

fname = os.path.dirname(dc.__file__) + '\\..\\..\\..\\..\\tutorials\\assets\\app_res_1d_data.dobs'
np.savetxt(fname, data_array, fmt='%.4e')
