"""
Stitched 1D Forward Simulation
==============================

Here we use the module *SimPEG.electromangetics.time_domain_1d* to predict
time domain data for a set of "stitched" 1D soundings. That is, the data
for each source is predicted for a separate, user-defined 1D model.
In this tutorial, we focus on the following:

    - Defining receivers, sources and the survey for the stitched 1D case
    - Constructing a stitched model - a set of 1D vertical conductivity profiels
    - Running an EM simulation

For each sounding, our survey geometry consists of a horizontal loop source with a
radius of 10 m located 203 m above the Earth's surface. The receiver is located at the centre
of the loop and measures the vertical component of the response.


"""

#####################################################
# Import Modules
# --------------
#

import numpy as np
from scipy.spatial import Delaunay, cKDTree
import os
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from discretize import TensorMesh
from pymatsolver import PardisoSolver

from SimPEG import maps, utils
from SimPEG.utils import mkvc
import SimPEG.electromagnetics.time_domain as tdem
from SimPEG.electromagnetics.utils.em1d_utils import plot_layer, get_vertical_discretization_time, set_mesh_1d, Stitched1DModel

plt.rcParams.update({'font.size': 16})
write_output = False


#####################################################################
# Create Survey
# -------------
#
# Here we demonstrate a general way to define receivers, sources and the survey.
# For this tutorial, we define a line of equally spaced 1D soundings along the
# Easting direction. However, there is no restriction on the spacing and position
# of each sounding.
nx = 11
ny = 3
x = np.arange(nx)*50
y = np.arange(ny)*100
z = np.array([30.])

xyz = utils.ndgrid(x, y, z)
n_sounding = xyz.shape[0]
source_locations = xyz  # xyz locations for the centre of the loop
source_current = 1.
source_radius = 10.

receiver_locations = xyz   # xyz locations for the receivers
receiver_orientation = "z"            # "x", "y" or "z"
times = np.logspace(-5, -2, 16)       # time channels

# Define the waveform. In this case all sources use the same waveform.
waveform = tdem.waveforms.StepOffWaveform()

# For each sounding, we define the source and the associated receivers.
source_list = []
for ii in range(0, n_sounding):

    # Source and receiver locations
    source_location = mkvc(source_locations[ii, :])
    receiver_location = mkvc(receiver_locations[ii, :])

    # Receiver list for source i
    receiver_list = [
        tdem.receivers.PointMagneticFluxTimeDerivative(
            receiver_location, times, orientation=receiver_orientation
        )
    ]

    # Source ii
    source_list.append(
        tdem.sources.CircularLoop(
            receiver_list=receiver_list, location=source_location, waveform=waveform,
            radius=source_radius, current=source_current, i_sounding=ii
        )
    )

# Define the survey
survey = tdem.Survey(source_list)

###############################################
# Defining a Global Mesh and Model
# --------------------------------
#
# It is easy to create and visualize 2D and 3D models in SimPEG, as opposed
# to an arbitrary set of local 1D models. Here, we create a 2D model
# which represents the global conductivity structure of the Earth. In the next
# part of the tutorial, we will demonstrate how the set of local 1D models can be
# extracted and organized for the stitched 1D simulation. This process can
# be adapted easily for 3D meshes and models.
#

# line number
line = (np.arange(ny).repeat(nx)).astype(float)
# time stamp
time_stamp = np.arange(n_sounding).astype(float)
# topography
topography = np.c_[xyz[:,:2], np.zeros(n_sounding)]

fig, ax = plt.subplots(1,1, figsize=(5,5))
ax.plot(xyz[:,0], xyz[:,1], '.')
ax.set_aspect(1)
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_title("Sounding locations")

# A function for generating a wedge layer 
def get_y(x):
    y = 30/500 * x + 70.
    return y

# Conductivity values for each unit
background_conductivity = 1./100
layer_conductivity = 1./10

# Define a 1D vertical mesh
mesh_1d = set_mesh_1d(hz)
# Generate a stitched 1D model
n_layer = hz.size
conductivity = np.zeros((n_sounding, n_layer), dtype=float)

for i_sounding in range(n_sounding):
    y = get_y(xyz[i_sounding, 0])
    layer_ind = np.logical_and(mesh_1d.vectorCCx>50., mesh_1d.vectorCCx<y)
    conductivity_1d = np.ones(n_layer, dtype=float) * background_conductivity
    conductivity_1d[layer_ind] = layer_conductivity
    conductivity[i_sounding,:]=conductivity_1d

# Note: oder of the conductivity model 
stitched_conductivity_model = conductivity.flatten()

# Generate a Stitched1DModel object for plotting
model_plot = Stitched1DModel(
    hz=hz,
    line=line,
    time_stamp=time_stamp,
    topography=topography,
    physical_property=1./stitched_conductivity_model
)

_, ax, cb = model.plot_section(cmap='turbo', aspect=0.5, dx=20, i_line=2)
cb.set_label("Resistivity ($\Omega$m)")

# the optimum layer thicknesses for a set number of layers. Note that when defining
# the thicknesses, it is the number of layers minus one.
thicknesses = hz[:-1]

#######################################################################
# Define the Mapping, Forward Simulation and Predict Data
# -------------------------------------------------------
#
# Here we define the simulation and predict the TDEM data.
# The simulation requires the user define the survey, the layer thicknesses
# and a mapping from the model to the conductivities.
#
# When using the *SimPEG.electromagnetics.time_domain_1d* module, predicted
# data are organized by source (sounding), then by receiver, then by time channel.
#

# Model and mapping. Here the model is defined by the log-conductivity.
stitched_model = np.log(stitched_conductivity_model)
mapping = maps.ExpMap(nP=len(sounding_models))

# Define the simulation
simulation = tdem.Simulation1DLayeredStitched(
    survey=survey, thicknesses=thicknesses, sigmaMap=mapping,
    parallel=False, n_cpu=2, solver=PardisoSolver
)

# Predict data
dpred = simulation.dpred(stitched_model)


#######################################################################
# Plotting Results
# ----------------
#

d = np.reshape(dpred, (n_sounding, len(times)))

fig= plt.figure(figsize=(7, 7))
ax = fig.add_axes([0.15, 0.15, 0.8, 0.8])

i_line = 0
ind_line = line == i_line
for ii in range(0, len(times)):
    ax.semilogy(receiver_locations[ind_line, 0], np.abs(d[ind_line, ii]), 'k-', lw=3)
ax.set_xlabel("Sounding location (m)")
ax.set_ylabel("|dBdt| (T/s)")
ax.set_title("Line nubmer {:.0f}".format(i_line))

#######################################################################
# Write Outputs (Optional)
# ------------------------
#

if write_output:

    dir_path = os.path.dirname(__file__).split(os.path.sep)
    dir_path.extend(["outputs"])
    dir_path = os.path.sep.join(dir_path) + os.path.sep

    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

    np.random.seed(649)
    noise = 0.1*np.abs(dpred)*np.random.rand(len(dpred))
    dpred += noise
    fname = dir_path + 'em1dtm_stitched_data.txt'

    loc = np.repeat(source_locations, len(times), axis=0)
    fvec = np.kron(np.ones(n_sounding), times)

    np.savetxt(
        fname,
        np.c_[loc, fvec, dpred],
        fmt='%.4e', header='X Y Z TIME DBDT_Z'
    )
