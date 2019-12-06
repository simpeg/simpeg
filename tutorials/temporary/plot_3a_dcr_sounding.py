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

from SimPEG.utils import ModelBuilder
from SimPEG import maps
from SimPEG.electromagnetics.static import resistivity as dc
from SimPEG.electromagnetics.static.utils.StaticUtils import geometric_factor

try:
    from pymatsolver import Pardiso as Solver
except ImportError:
    from SimPEG import SolverLU as Solver

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
electrode_separations = np.linspace(40., 320., 15)  # Number of electrode locations along EW profile

source_list = []  # create empty array for sources to live

for ii in range(0, len(electrode_separations)):
    
    a = electrode_separations[ii]
    
    # AB electrode locations for source
    a_location = np.r_[-1.5*a, 0.]
    b_location = np.r_[1.5*a, 0.]

    # MN electrode locations for receivers
    m_location = np.c_[-0.5*a, 0.]
    n_location = np.c_[0.5*a, 0.]

    # Create receivers list. Define as pole or dipole. Can choose to
    # measured potential or components of electric field.
    receiver_list = dc.receivers.Dipole_ky(
            m_location, n_location#, data_type="apparent_resistivity"
            )
    receiver_list = [receiver_list]

    # Define the source properties and associated receivers
    source_list.append(
            dc.sources.Dipole(receiver_list, a_location, b_location)
            )

# Define survey
survey = dc.Survey(source_list)


#############################################
# Defining a Tensor Mesh
# ----------------------
#
# Here, we create the tensor mesh
#

dh = 5.
hx = [(dh, 10, -1.3), (dh, 300), (dh, 10, 1.3)]
hz = [(dh, 10, -1.3), (dh, 150)]
mesh = TensorMesh([hx, hz], 'CN')



###############################################################
# Create Conductivity Model and Mapping for OcTree Mesh
# -----------------------------------------------------
#
# Here we define the conductivity model that will be used to predict DC
# resistivity data. The model consists of a conductive sphere and a
# resistive sphere within a moderately conductive background. Note that
# you can carry through this work flow with a resistivity model if desired.
#

# Define conductivity model in S/m (or resistivity model in Ohm m)
layer_tops = np.r_[0., -100., -200.]
layer_resistivities = np.r_[1e3, 4e3, 1e2]

# Define mapping from model to active cells
model_map = maps.IdentityMap(mesh)

# Define model
model = ModelBuilder.layeredModel(mesh.gridCC, layer_tops, layer_resistivities)


# Plot Conductivity Model
fig = plt.figure(figsize=(8.5, 4))

#plotting_map = maps.InjectActiveCells(mesh, ind_active, np.nan)
log_model = np.log10(model)

ax1 = fig.add_axes([0.05, 0.05, 0.8, 0.9])
mesh.plotImage(
    log_model, ax=ax1, grid=False,
    clim=(2, 4)
)
ax1.set_title('Resistivity Model')

ax2 = fig.add_axes([0.87, 0.05, 0.05, 0.9])
norm = mpl.colors.Normalize(vmin=2, vmax=4)
cbar = mpl.colorbar.ColorbarBase(
    ax2, norm=norm, orientation='vertical', format="$10^{%.1f}$"
)
cbar.set_label(
    'Conductivity [S/m]', rotation=270, labelpad=15, size=12
)


#######################################################################
# Predict DC Resistivity Data
# ---------------------------
#
# Here we predict DC resistivity data. If the keyword argument *sigmaMap* is
# defined, the simulation will expect a conductivity model. If the keyword
# argument *rhoMap* is defined, the simulation will expect a resistivity model.
#

simulation = dc.simulation_2d.Problem2D_N(
        mesh, survey=survey, rhoMap=model_map, Solver=Solver
        )

dpred = simulation.dpred(model)

G = geometric_factor(survey, survey_type='dipole-dipole', space_type='half-space')

# Plot apparent conductivity pseudo-section
fig = plt.figure(figsize=(11, 5))

ax1 = fig.add_axes([0.05, 0.05, 0.8, 0.9])
ax1.semilogy(electrode_separations, dpred/G)

plt.show()

