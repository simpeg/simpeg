"""
Stitched 1D Forward Simulation
==============================

Here we use the module *SimPEG.electromangetics.frequency_domain_1d* to predict
frequency domain data for a set of "stitched" 1D soundings. That is, the data
for each source is predicted for a separate, user-defined 1D model.
In this tutorial, we focus on the following:

    - Defining receivers, sources and the survey for the stitched 1D case
    - Constructing a model on a 2D/3D mesh, then interpolating that model to create the set of local 1D models
    - The organization of the set of 1D models

For each sounding, we compute predicted data for a vertical magnetic dipole source
located 30 m above the Earth's surface. The receiver is offset
10 m horizontally from the source.


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

from SimPEG import maps
from SimPEG.utils import mkvc
import SimPEG.electromagnetics.frequency_domain_1d as em1d
from SimPEG.electromagnetics.utils.em1d_utils import (
    plot_layer, get_vertical_discretization_frequency
)

plt.rcParams.update({'font.size': 16})
save_file = True

# sphinx_gallery_thumbnail_number = 1

#####################################################################
# Create Survey
# -------------
#
# Here we demonstrate a general way to define receivers, sources and the survey.
# For this tutorial, we define a line of equally spaced 1D soundings along the
# Easting direction. However, there is no restriction on the spacing and position
# of each sounding.
#

x = np.linspace(50,4950,50)
n_sounding = len(x)
y = np.zeros(n_sounding)
z = 30 *np.ones(n_sounding)

source_locations = np.c_[x, y, z]  # xyz locations for the sources
moment_amplitude=1.

receiver_locations = np.c_[x+10., y, z]       # xyz locations for the receivers
receiver_orientation = "z"                    # "x", "y" or "z"
field_type = "ppm"                            # "secondary", "total" or "ppm"

frequencies = np.array([25., 100., 382, 1822, 7970, 35920], dtype=float)

# For each sounding, we define the source and the associated receivers.
source_list = []
for ii in range(0, n_sounding):
    
    # Source and receiver locations
    source_location = mkvc(source_locations[ii, :])
    receiver_location = mkvc(receiver_locations[ii, :])
    
    # Define receiver list for source ii
    receiver_list = []
    receiver_list.append(
        em1d.receivers.PointReceiver(
            receiver_location, frequencies, orientation=receiver_orientation,
            field_type=field_type, component="real"
        )
    )
    receiver_list.append(
        em1d.receivers.PointReceiver(
            receiver_location, frequencies, orientation=receiver_orientation,
            field_type=field_type, component="imag"
        )
    )
        
    # Define source ii
    source_list.append(
        em1d.sources.MagneticDipoleSource(
            receiver_list=receiver_list, location=source_location, orientation="z",
            moment_amplitude=moment_amplitude
        )
    )

# Survey
survey = em1d.survey.EM1DSurveyFD(source_list)


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

# Conductivity values for each unit
background_conductivity = 0.1
overburden_conductivity = 0.025
slope_conductivity = 0.4

# Define a global 2D mesh.
dx = 50.                                      # horizontal cell width
ncx = int(np.ceil((np.max(x)-np.min(x))/dx))  # number of horizontal cells
hx = np.ones(ncx) * dx                        # horizontal cell widths
hz = 10*np.ones(40)                           # vertical cell widths
mesh2D = TensorMesh([hx, hz], x0='0N')

# Define global 2D model
def PolygonInd(mesh, pts):
    hull = Delaunay(pts)
    inds = hull.find_simplex(mesh.gridCC)>=0
    return inds

model = np.ones(mesh2D.nC) * background_conductivity

layer_ind = mesh2D.gridCC[:, -1] > -30.
model[layer_ind] = overburden_conductivity

x0 = np.r_[0., -30.]
x1 = np.r_[mesh2D.nodes_x[-1], -30.]
x2 = np.r_[mesh2D.nodes_x[-1], -130.]
x3 = np.r_[0., -50.]
pts = np.vstack((x0, x1, x2, x3, x0))
poly_inds = PolygonInd(mesh2D, pts)
model[poly_inds] = slope_conductivity

# Plot global 2D model
fig = plt.figure(figsize=(9, 3))
ax1 = fig.add_axes([0.15, 0.12, 0.65, 0.78])
log_mod = np.log10(model)

mesh2D.plot_image(
    log_mod, ax=ax1, grid=False,
    clim=(np.log10(overburden_conductivity), np.log10(slope_conductivity)),
    pcolorOpts={"cmap": "viridis"},
)
ax1.set_ylim(mesh2D.vectorNy.min(), mesh2D.vectorNy.max())

ax1.set_title("Conductivity Model")
ax1.set_xlabel("x (m)")
ax1.set_ylabel("z (m)")

ax2 = fig.add_axes([0.82, 0.12, 0.03, 0.78])
norm = mpl.colors.Normalize(
    vmin=np.log10(overburden_conductivity), vmax=np.log10(slope_conductivity)
)
cbar = mpl.colorbar.ColorbarBase(
    ax2, norm=norm, cmap=mpl.cm.viridis, orientation="vertical", format="$10^{%.1f}$"
)
cbar.set_label("Conductivity [S/m]", rotation=270, labelpad=15, size=12)

################################################################
# Layer Thicknesses and Conductivities of Local 1D Models
# -------------------------------------------------------
#
# Here, we plot and organize the set of local 1D conductivity models which are used to
# simulate the response for all sounding locations. Where *mi* is a 1D array
# representing the 1D conductivity model for sounding *i*, the 1D array
# containing all local 1D models is organized as [m1,m2,m3,...]. That is,
# we organize by sounding, then by layer.
# Since the conductivity of the Earth was defined on a 2D mesh, we must interpolate
# from the global 2D model to the local 1D models for each sounding.
#


# Define layer the thicknesses used for all local 1D models.
# For a background conductivity and a set of frequencies, we can determine the
# the optimum layer thicknesses for a set number of layers. Note that when defining
# the thicknesses, it is the number of layers minus one.
n_layer = 30
thicknesses = get_vertical_discretization_frequency(
    frequencies, sigma_background=background_conductivity, n_layer=n_layer-1
)

# Nearest neighbour interpolation. We use nearest neighbour to output a 1D
# array that contains the 1D models for all soundings. This vector is organized
# by sounding location, then by layer from top to bottom.
z = np.r_[thicknesses, thicknesses[-1]]
z = -(np.cumsum(z) - z/2.)
x, z = np.meshgrid(x, z)
xz = np.c_[mkvc(x), mkvc(z)]

tree = cKDTree(mesh2D.cell_centers)
_, ind = tree.query(xz)

sounding_models = model[ind]

# Create a 2D mesh and plot the numpy array containing all the organized 1D models.
hx = np.ones(n_sounding)
hz = np.r_[thicknesses, thicknesses[-1]]
mesh_soundings = TensorMesh([hz, hx], x0='00')

# Plot the organized 1D models
fig = plt.figure(figsize=(4, 7.5))
ax1 = fig.add_axes([0.15, 0.12, 0.67, 0.78])
log_mod_sounding = np.log10(sounding_models)

mesh_soundings.plot_image(
    log_mod_sounding, ax=ax1, grid=True,
    clim=(np.log10(overburden_conductivity), np.log10(slope_conductivity)),
    pcolorOpts={"cmap": "viridis"},
)
ax1.set_ylim(mesh_soundings.vectorNy.min(), mesh_soundings.vectorNy.max())

ax1.set_xticks([])
ax1.set_yticks([])

ax1.set_title("Sounding Models")
ax1.set_xlabel("Layer")
ax1.set_ylabel("Sounding Number")

ax2 = fig.add_axes([0.85, 0.12, 0.05, 0.78])
norm = mpl.colors.Normalize(
    vmin=np.log10(overburden_conductivity), vmax=np.log10(slope_conductivity)
)
cbar = mpl.colorbar.ColorbarBase(
    ax2, norm=norm, cmap=mpl.cm.viridis, orientation="vertical", format="$10^{%.1f}$"
)
cbar.set_label("Conductivity [S/m]", rotation=270, labelpad=15, size=12)


#######################################################################
# Define the Mapping, Forward Simulation and Predict Data
# -------------------------------------------------------
#
# Here we define the simulation and predict the FDEM data.
# The simulation requires the user define the survey, the layer thicknesses
# and a mapping from the model to the conductivities.
# 
# When using the *SimPEG.electromagnetics.frequency_domain_1d* module, predicted
# data are organized by source (sounding), then by receiver, then by frequency.
#

# Model and mapping. Here the model is defined by the log-conductivity.
sounding_models = np.log(sounding_models)
mapping = maps.ExpMap(nP=len(sounding_models))

# Define the simulation
simulation = em1d.simulation.StitchedEM1DFMSimulation(
    survey=survey, thicknesses=thicknesses, sigmaMap=mapping,
    parallel=False, Solver=PardisoSolver
)

# Predict data
dpred = simulation.dpred(sounding_models)


#######################################################################
# Plotting Results
# -------------------------------------------------
#

d_plotting = np.reshape(dpred, (2*n_sounding, len(frequencies))).T

d_real = d_plotting[:, 0::2]
d_imag = d_plotting[:, 1::2]

fig, ax = plt.subplots(1,1, figsize = (7, 7))

ax.loglog(frequencies, np.abs(d_real[:, 0]), 'k-', lw=2)
ax.loglog(frequencies, np.abs(d_imag[0:len(frequencies):, 0]), 'k--', lw=2)
for ii in range(1, n_sounding):
    ax.loglog(frequencies, np.abs(d_real[:, ii]), '-', lw=2)
    ax.loglog(frequencies, np.abs(d_imag[0:len(frequencies):, ii]), '--', lw=2)

ax.grid(True)
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("|Hs/Hp| (ppm)")
ax.set_title("Secondary Magnetic Field")
ax.legend(["Real", "Imaginary"])

#######################################################################
# Write Output (Optional)
# -----------------------
#

if save_file == True:

    dir_path = os.path.dirname(em1d.__file__).split(os.path.sep)[:-3]
    dir_path.extend(["tutorials", "07-fdem", "em1dfm_stitched"])
    dir_path = os.path.sep.join(dir_path) + os.path.sep

    np.random.seed(491)
    noise = 0.1*np.abs(dpred)*np.random.rand(len(dpred))
    dpred += noise
    fname = dir_path + "em1dfm_stitched_data.obs"

    loc = np.repeat(source_locations, len(frequencies), axis=0)
    fvec = np.kron(np.ones(n_sounding), frequencies)
    dout = np.c_[mkvc(d_real), mkvc(d_imag)]

    np.savetxt(
        fname,
        np.c_[loc, fvec, dout],
        fmt='%.4e', header='X Y Z FREQUENCY HZ_REAL HZ_IMAG'
    )

