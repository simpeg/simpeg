"""
Simulation on an OcTree Mesh
============================

Here we use the module *SimPEG.electromagnetics.frequency_domain* to simulate the
FDEM response for an airborne survey using an OcTree mesh and a resistivity model.
To limit computational demant, we simulate airborne data at a single frequency
for a vertical coplanar survey geometry. This tutorial can be easily adapted to
simulate data at many frequencies. For this tutorial, we focus on the following:

    - How to define the transmitters and receivers
    - How to define the survey
    - How to define the topography
    - How to solve the FDEM problem on OcTree meshes
    - The units of the resistivity model and resulting data
    

Please note that we have used a coarse mesh to shorten the time of the simulation.
Proper discretization is required to simulate the fields at each frequency with
sufficient accuracy.


"""

#########################################################################
# Import modules
# --------------
#

from discretize import TreeMesh
from discretize.utils import mkvc, refine_tree_xyz

from SimPEG.utils import plot2Ddata, surface2ind_topo
from SimPEG import maps
import SimPEG.electromagnetics.frequency_domain as fdem

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

try:
    from pymatsolver import Pardiso as Solver
except ImportError:
    from SimPEG import SolverLU as Solver
    
save_file = False

# sphinx_gallery_thumbnail_number = 2


###############################################################
# Defining Topography
# -------------------
#
# Here we define surface topography as an (N, 3) numpy array. Topography could
# also be loaded from a file. Here we define flat topography, however more
# complex topographies can be considered.
#

xx, yy = np.meshgrid(np.linspace(-3000, 3000, 101), np.linspace(-3000, 3000, 101))
zz = np.zeros(np.shape(xx))
topo_xyz = np.c_[mkvc(xx), mkvc(yy), mkvc(zz)]

#####################################################################
# Create Airborne Survey
# ----------------------
#
# For this example, the survey consists of a uniform grid of airborne
# measurements. To save time, we will only compute the response for a single
# frequency.
#

# Frequencies being predicted
frequencies = [100, 500, 2500]

# Defining transmitter locations
N = 9
xtx, ytx, ztx = np.meshgrid(
    np.linspace(-200, 200, N), np.linspace(-200,200, N), [50]
)
source_locations = np.c_[mkvc(xtx), mkvc(ytx), mkvc(ztx)]
ntx = np.size(xtx)

# Define receiver locations
xrx, yrx, zrx = np.meshgrid(
    np.linspace(-200, 200, N), np.linspace(-200,200, N), [30]
)
receiver_locations = np.c_[mkvc(xrx), mkvc(yrx), mkvc(zrx)]

source_list = []  # Create empty list to store sources

# Each unique location and frequency defines a new transmitter
for ii in range(ntx):

    # Define receivers of different type at each location
    bzr_receiver = fdem.receivers.PointMagneticFluxDensitySecondary(
            receiver_locations[ii, :], 'z', 'real'
            )
    bzi_receiver = fdem.receivers.PointMagneticFluxDensitySecondary(
            receiver_locations[ii, :], 'z', 'imag'
            )
    receivers_list = [bzr_receiver, bzi_receiver]

    for jj in range(len(frequencies)):

        # Must define the transmitter properties and associated receivers
        source_list.append(
            fdem.sources.MagDipole(
                receivers_list, frequencies[jj], source_locations[ii], orientation='z'
            )
        )
    
survey = fdem.Survey(source_list)

###############################################################
# Create OcTree Mesh
# ------------------
#
# Here we define the OcTree mesh that is used for this example.
# We chose to design a coarser mesh to decrease the run time.
# When designing a mesh to solve practical frequency domain problems:
# 
#     - Your smallest cell size should be 10%-20% the size of your smallest skin depth
#     - The thickness of your padding needs to be 2-3 times biggest than your largest skin depth
#     - The skin depth is ~500*np.sqrt(rho/f)
#
#

dh = 25.                                                     # base cell width
dom_width = 3000.                                            # domain width
nbc = 2**int(np.round(np.log(dom_width/dh)/np.log(2.)))      # num. base cells

# Define the base mesh
h = [(dh, nbc)]
mesh = TreeMesh([h, h, h], x0='CCC')

# Mesh refinement based on topography
mesh = refine_tree_xyz(
    mesh, topo_xyz, octree_levels=[0, 0, 0, 1], method='surface', finalize=False
)

# Mesh refinement near transmitters and receivers
mesh = refine_tree_xyz(
    mesh, receiver_locations, octree_levels=[2, 4], method='radial', finalize=False
)

# Refine core mesh region
xp, yp, zp = np.meshgrid([-250., 250.], [-250., 250.], [-300., 0.])
xyz = np.c_[mkvc(xp), mkvc(yp), mkvc(zp)]
mesh = refine_tree_xyz(
    mesh, xyz, octree_levels=[0, 2, 4], method='box', finalize=False
)

mesh.finalize()

###############################################################
# Defining a Resistivity Model
# ----------------------------
#
# Here, we create the model that will be used to predict frequency
# domain data and the mapping from the model to the mesh. Here,
# the model consists of a conductive block within a more resistive
# background.
#

# Resistivity in Ohm m
air_resistivity = 1e8
background_resistivity = 1e2
block_resistivity = 1e-1

# Find cells that are active in the forward modeling (cells below surface)
ind_active = surface2ind_topo(mesh, topo_xyz)

# Define mapping from model to active cells
model_map = maps.InjectActiveCells(mesh, ind_active, air_resistivity)

# Define model. Models in SimPEG are vector arrays
model = background_resistivity*np.ones(ind_active.sum())
ind_block = (
    (mesh.gridCC[ind_active, 0] < 100.) & (mesh.gridCC[ind_active, 0] > -100.) &
    (mesh.gridCC[ind_active, 1] < 100.) & (mesh.gridCC[ind_active, 1] > -100.) &
    (mesh.gridCC[ind_active, 2] > -250.) & (mesh.gridCC[ind_active, 2] < -50.)
)
model[ind_block] = block_resistivity

# Plot Log of Resistivity Model
fig = plt.figure(figsize=(7, 6))

plotting_map = maps.InjectActiveCells(mesh, ind_active, np.nan)
log_model = np.log10(model)

ax1 = fig.add_axes([0.13, 0.1, 0.6, 0.85])
mesh.plotSlice(
    plotting_map*log_model, normal='Y', ax=ax1, ind=int(mesh.hx.size/2),
    grid=True, clim=(np.log10(block_resistivity), np.log10(background_resistivity))
)
ax1.set_title('Resistivity Model at Y = 0 m')

ax2 = fig.add_axes([0.75, 0.1, 0.05, 0.85])
norm = mpl.colors.Normalize(vmin=np.log10(block_resistivity), vmax=np.log10(background_resistivity))
cbar = mpl.colorbar.ColorbarBase(
    ax2, norm=norm, orientation='vertical', format="$10^{%.1f}$"
)
cbar.set_label(
    'Resistivity (Ohm m)', rotation=270, labelpad=15, size=12
)


#######################################################################
# Predict Data
# ------------
#
# Here we demonstrate how to simulate the frequency domain response for
# an electrical resistivity model on an OcTree mesh.
#

# Define the forward simulation
simulation = fdem.simulation.Simulation3DMagneticFluxDensity(
        mesh, survey=survey, rhoMap=model_map, Solver=Solver
        )

# Compute predicted data for a your model
mu0 = 4*np.pi*1e-7
dpred = simulation.dpred(model)/mu0

# Data are organized by transmitter then by receiver. We had nFreq transmitters
# and each transmitter had 2 receivers (real and imaginary component). So
# first we will pick out the real and imaginary data
hz_real = dpred[0:len(dpred):2]/mu0
hz_imag = dpred[1:len(dpred):2]/mu0

# Then we will will reshape the data.
hz_real = np.reshape(hz_real, (ntx, len(frequencies)))
hz_imag = np.reshape(hz_imag, (ntx, len(frequencies)))


fig = plt.figure(figsize=(10, 4))

# Real Component
frequencies_index = 0
v_max = np.max(np.abs(hz_real[:, frequencies_index]))
ax1 = fig.add_axes([0.05, 0.05, 0.35, 0.9])
plot2Ddata(
    receiver_locations[:, 0:2], hz_real[:, frequencies_index], ax=ax1,
    ncontour=30, clim=(-v_max, v_max), contourOpts={"cmap": "RdBu_r"}
    )
ax1.set_title('Re[$H_z$] at 100 Hz')

ax2 = fig.add_axes([0.41, 0.05, 0.02, 0.9])
norm = mpl.colors.Normalize(vmin=-v_max, vmax=v_max)
cbar = mpl.colorbar.ColorbarBase(
    ax2, norm=norm, orientation='vertical', cmap=mpl.cm.RdBu_r
)
cbar.set_label('$A/m$', rotation=270, labelpad=15, size=12)

# Imaginary Component
v_max = np.max(np.abs(hz_imag[:, frequencies_index]))
ax1 = fig.add_axes([0.55, 0.05, 0.35, 0.9])
plot2Ddata(
    receiver_locations[:, 0:2], hz_imag[:, frequencies_index],
    ax=ax1, ncontour=30, clim=(-v_max, v_max), contourOpts={"cmap": "RdBu_r"}
)
ax1.set_title('Im[$H_z$] at 100 Hz')

ax2 = fig.add_axes([0.91, 0.05, 0.02, 0.9])
norm = mpl.colors.Normalize(vmin=-v_max, vmax=v_max)
cbar = mpl.colorbar.ColorbarBase(
    ax2, norm=norm, orientation='vertical', cmap=mpl.cm.RdBu_r
)
cbar.set_label('$A/m$', rotation=270, labelpad=15, size=12)

plt.show()


#######################################################
# Optional: Export Data
# ---------------------
#
# Write the true model, data and topography
#


if save_file == True:

    fname = os.path.dirname(fdem.__file__) + '\\..\\..\\..\\tutorials\\assets\\fdem\\fdem_topo.txt'
    np.savetxt(fname, np.c_[topo_xyz], fmt='%.4e')

    fname = os.path.dirname(fdem.__file__) + '\\..\\..\\..\\tutorials\\assets\\fdem\\fdem_data.txt'
    hz_real = hz_real + 1e-10*np.random.rand(len(hz_real))
    hz_imag = hz_imag + 5e-11*np.random.rand(len(hz_imag))
    f_vec = frequencies*np.ones(len(hz_real))
    np.savetxt(
        fname,
        np.c_[f_vec, receiver_locations, hz_real, hz_imag],
        fmt='%.4e'
    )

    output_model = plotting_map*model
    output_model[np.isnan(output_model)] = 0.

    fname = os.path.dirname(fdem.__file__) + '\\..\\..\\..\\tutorials\\assets\\fdem\\true_model.txt'
    np.savetxt(
        fname,
        output_model,
        fmt='%.4e'
    )

