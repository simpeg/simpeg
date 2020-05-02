"""
Simulation on a Cylindrical Mesh
================================

Here we use the module *SimPEG.electromagnetics.frequency_domain* to simulate the
FDEM response for an airborne survey using a cylindrical mesh and a conductivity
model. We simulate a single line of airborne data at many frequencies for a vertical
coplanar survey geometry. For this tutorial, we focus on the following:

    - How to define the transmitters and receivers
    - How to define the survey
    - How to solve the FDEM problem on cylindrical meshes
    - The units of the conductivity/resistivity model and resulting data
    

Please note that we have used a coarse mesh to shorten the time of the simulation.
Proper discretization is required to simulate the fields at each frequency with
sufficient accuracy.


"""

#########################################################################
# Import modules
# --------------
#

from discretize import CylMesh
from discretize.utils import mkvc

from SimPEG import maps
import SimPEG.electromagnetics.frequency_domain as fdem

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

try:
    from pymatsolver import Pardiso as Solver
except ImportError:
    from SimPEG import SolverLU as Solver
    
write_file = False

# sphinx_gallery_thumbnail_number = 2


#####################################################################
# Create Airborne Survey
# ----------------------
#
# Here we define an airborne survey that consists of a single line of EM data
# measurements over a range of frequencies. The loop geometry is horizontal
# coplanar.
#

# Frequencies being predicted (10 Hz to 10000 Hz)
frequencies = np.logspace(1, 4, 16)

# Defining transmitter locations
xtx, ytx, ztx = np.meshgrid(np.linspace(0, 200, 41), [0], [55])
source_locations = np.c_[mkvc(xtx), mkvc(ytx), mkvc(ztx)]
ntx = np.size(xtx)

# Define receiver locations
xrx, yrx, zrx = np.meshgrid(np.linspace(0, 200, 41), [0], [50])
receiver_locations = np.c_[mkvc(xrx), mkvc(yrx), mkvc(zrx)]

source_list = []  # Create empty list to store sources

# Each unique location and frequency defines a new transmitter
for ii in range(ntx):

    # Define receivers of different types at each location. Real and imaginary
    # measurements require separate receivers. You can define the orientation of
    # the transmitters and receivers for different survey geometries.
    bzr_receiver = fdem.receivers.PointMagneticFluxDensitySecondary(
        receiver_locations[ii, :], 'z', 'real'
    )
    bzi_receiver = fdem.receivers.PointMagneticFluxDensitySecondary(
        receiver_locations[ii, :], 'z', 'imag'
    )
    receivers_list = [bzr_receiver, bzi_receiver]  # must be a list

    for jj in range(len(frequencies)):

        # Must define the transmitter properties and associated receivers
        source_list.append(
            fdem.sources.MagDipole(
                receivers_list, frequencies[jj], source_locations[ii], orientation='z'
            )
        )

# Define the survey
survey = fdem.Survey(source_list)

###############################################################
# Create Cylindrical Mesh
# -----------------------
#
# Here we create the cylindrical mesh that will be used for this tutorial
# example. We chose to design a coarser mesh to decrease the run time.
# When designing a mesh to solve practical frequency domain problems:
# 
#     - Your smallest cell size should be 10%-20% the size of your smallest skin depth
#     - The thickness of your padding needs to be 2-3 times biggest than your largest skin depth
#     - The skin depth is ~500*np.sqrt(rho/f)
#     
#

hr = [(10., 50), (10., 10, 1.5)]  # discretization in the radial direction
hz = [(10., 10, -1.5), (10., 100), (10., 10, 1.5)]  # discretization in vertical direction

mesh = CylMesh([hr, 1, hz], x0='00C')

###############################################################
# Create Conductivity/Resistivity Model and Mapping
# -------------------------------------------------
#
# Here, we create the model that will be used to predict frequency domain
# data and the mapping from the model to the mesh. The model
# consists of a long vertical conductive pipe and a resistive
# surface layer. For this example, we will have only flat topography.
#

# Conductivity in S/m (or resistivity in Ohm m)
air_conductivity = 1e-8
background_conductivity = 1e-1
layer_conductivity = 1e-2
pipe_conductivity = 1e1

# Find cells that are active in the forward modeling (cells below surface)
ind_active = mesh.gridCC[:, 2] < 0

# Define mapping from model to active cells
model_map = maps.InjectActiveCells(mesh, ind_active, air_conductivity)

# Define the model
model = background_conductivity*np.ones(ind_active.sum())
ind_layer = (
    (mesh.gridCC[ind_active, 2] > -100.) & (mesh.gridCC[ind_active, 2] < -0)
)
model[ind_layer] = layer_conductivity
ind_pipe = (
    (mesh.gridCC[ind_active, 0] < 60.) &
    (mesh.gridCC[ind_active, 2] > -10000.) & (mesh.gridCC[ind_active, 2] < 0.)
)
model[ind_pipe] = pipe_conductivity

# Plot Conductivity Model
mpl.rcParams.update({'font.size': 14})
fig = plt.figure(figsize=(5, 6))

plotting_map = maps.InjectActiveCells(mesh, ind_active, np.nan)
log_model = np.log10(model)

ax1 = fig.add_axes([0.14, 0.1, 0.6, 0.85])
mesh.plotImage(
    plotting_map*log_model, ax=ax1, grid=False,
    clim=(np.log10(layer_conductivity), np.log10(pipe_conductivity))
)
ax1.set_title('Conductivity Model (Survey in red)')

ax1.plot(receiver_locations[:, 0], receiver_locations[:, 2], 'r.')

ax2 = fig.add_axes([0.76, 0.1, 0.05, 0.85])
norm = mpl.colors.Normalize(
    vmin=np.log10(layer_conductivity), vmax=np.log10(pipe_conductivity)
)
cbar = mpl.colorbar.ColorbarBase(
    ax2, norm=norm, orientation='vertical', format="$10^{%.1f}$"
)
cbar.set_label(
    'Conductivity [S/m]', rotation=270, labelpad=15, size=12
)


######################################################
# Simulation: Predicting FDEM Data
# --------------------------------
#
# Here we define the formulation for solving Maxwell's equations. Since we are
# measuring the magnetic flux density and working with a conductivity model,
# the EB formulation is the most natural. We must also remember to define
# the mapping for the conductivity model. If you defined a resistivity model,
# use the kwarg *rhoMap* instead of *sigmaMap*
#

simulation = fdem.simulation.Simulation3DMagneticFluxDensity(
    mesh, survey=survey, sigmaMap=model_map, Solver=Solver
)

######################################################
# Predict and Plot Data
# ---------------------
#
# Here we show how the simulation is used to predict data.
#

# Compute predicted data for the given model.
dpred = simulation.dpred(model)

# Data are organized by transmitter location, then component, then frequency. We had nFreq
# transmitters and each transmitter had 2 receivers (real and imaginary component). So
# first we will pick out the real and imaginary data
bz_real = dpred[0:len(dpred):2]
bz_imag = dpred[1:len(dpred):2]

# Then we will will reshape the data.
bz_real = np.reshape(bz_real, (ntx, len(frequencies)))
bz_imag = np.reshape(bz_imag, (ntx, len(frequencies)))

# Plot secondary field along the profile at f = 1 Hz
fig = plt.figure(figsize=(7, 5))
ax1 = fig.add_axes([0.15, 0.1, 0.8, 0.85])
frequencies_index = 0
ax1.plot(receiver_locations[:, 0], bz_real[:, frequencies_index], 'b', lw=3)
ax1.plot(receiver_locations[:, 0], bz_imag[:, frequencies_index], 'r', lw=3)
ax1.set_xlim((0, np.max(xtx)))
ax1.set_xlabel('Easting [m]')
ax1.set_ylabel('Bz secondary [T]')
ax1.set_title('Secondary Bz-field at 1 Hz')
ax1.legend(['Real', 'Imaginary'], loc='lower right')

# Plot FEM sounding over the pipe
fig = plt.figure(figsize=(7, 5))
ax1 = fig.add_axes([0.15, 0.1, 0.8, 0.85])
location_index = 0
ax1.semilogx(frequencies, bz_real[location_index, :], 'b', lw=3)
ax1.semilogx(frequencies, bz_imag[location_index, :], 'r', lw=3)
ax1.set_xlim((np.min(frequencies), np.max(frequencies)))
ax1.set_xlabel('Frequency [Hz]')
ax1.set_ylabel('Bz secondary [T]')
ax1.set_title('Secondary Bz-field over pipe')
ax1.legend(['Real', 'Imaginary'], loc='lower left')

