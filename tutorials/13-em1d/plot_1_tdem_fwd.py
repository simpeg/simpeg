"""
Forward Simulation of 1D Time-Domain Data
==============================================





"""

#####################################################
# Import Modules
# --------------
#

import numpy as np
import os
from matplotlib import pyplot as plt
from discretize import TensorMesh

from SimPEG import maps
import SimPEG.electromagnetics.time_domain_1d as em1d
from SimPEG.electromagnetics.utils.em1d_utils import plot_layer

save_file = False
plt.rcParams.update({'font.size': 16})

#####################################################################
# Create Survey
# -------------
#
#

source_location = np.array([0., 0., 20.])  
source_orientation = "z"  # "x", "y" or "z"
source_current = 1.
source_radius = 6.

receiver_location = np.array([0., 0., 20.])
receiver_orientation = "z"  # "x", "y" or "z"
field_type = "secondary"  # "secondary", "total" or "ppm"

times = np.logspace(-5, -2, 31)

# Receiver list
receiver_list = []
receiver_list.append(
    em1d.receivers.PointReceiver(
        receiver_location, times, orientation=receiver_orientation,
        component="b"
    )
)

# Sources
source_list = [
    em1d.sources.HorizontalLoopSource(
        receiver_list=receiver_list, location=source_location,
        I=source_current, a=source_radius
    )
]

#source_list = [
#    em1d.sources.MagneticDipoleSource(
#        receiver_list=receiver_list, location=source_location, orientation="z",
#        I=source_current
#    )
#]

# Survey
survey = em1d.survey.EM1DSurveyTD(source_list)


###############################################
# Defining a 1D Layered Earth Model
# ---------------------------------
#
# Here, we define the layer thicknesses and electrical conductivities for our
# 1D simulation. If we have N layers, we define N electrical conductivity
# values and N-1 layer thicknesses. The lowest layer is assumed to extend to
# infinity.
#

# Physical properties
background_conductivity = 1e-1
layer_conductivity = 1e0

# Layer thicknesses
thicknesses = np.array([40., 40.])
n_layer = len(thicknesses) + 1

# physical property models
model = background_conductivity*np.ones(n_layer)
model[1] = layer_conductivity

# Define a mapping for conductivities
model_mapping = maps.IdentityMap(nP=n_layer)

# Plot conductivity model
plotting_mesh = TensorMesh([np.r_[thicknesses, 40.]])
plotLayer(model, plotting_mesh, showlayers=False)

#######################################################################
# Define the Forward Simulation and Predict Data
# ----------------------------------------------
#


# Simulate response for static conductivity
simulation = em1d.simulation.EM1DTMSimulation(
    survey=survey, thicknesses=thicknesses, sigmaMap=model_mapping,
)

dpred = simulation.dpred(model)

# Simulate response
fig = plt.figure(figsize = (8, 7))
ax = fig.add_axes([0.1, 0.1, 0.8, 0.85])
ax.loglog(times, dpred, 'k-o')
ax.set_xlabel("Times (s)")
ax.set_ylabel("|B| (T)")
ax.set_title("Magnetic Flux")


##################################################


if save_file == True:

    noise = 0.05*np.abs(dpred)*np.random.rand(len(dpred))
    dpred += noise
    fname = os.path.dirname(em1d.__file__) + '\\..\\tutorials\\assets\\em1dtm_data.obs'
    np.savetxt(
        fname,
        np.c_[times, dpred],
        fmt='%.4e'
    )






















