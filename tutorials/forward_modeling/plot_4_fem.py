"""
Frequency Domain EM
===================

Here we use the module *SimPEG.EM.FDEM* to predict the fields that
result from solving 3D frequency-domain EM problems. For this tutorial, we
focus on the following:

    - How to define the transmitters and receivers
    - How to define the survey
    - How to solve the FEM problem on Cylindrical and OcTree meshes
    - How to include topography
    - The units of the conductivity/resistivity model and resulting data

The tutorial contains 2 examples. In the first, we compute the FEM response for
an airborne survey using a cylindrical mesh and a conductivity model. In the
second example, we compute the FEM response using an OcTree mesh and a
resistivity model.
    

"""

#########################################################################
# Import modules
# --------------
#

from discretize import CylMesh, TreeMesh
from discretize.utils import mkvc, refine_tree_xyz

from SimPEG.utils import plot2Ddata, surface2ind_topo
from SimPEG import maps
from SimPEG.electromagnetics.frequency_domain import (
    receivers, sources, survey, simulation
    )

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

try:
    from pymatsolver import Pardiso as Solver
except ImportError:
    from SimPEG import SolverLU as Solver

# sphinx_gallery_thumbnail_number = 2

###############################################################
# CYL-MESH EXAMPLE
# ----------------
#
# Here we solve the forward 3D TDEM problem on a cylindrical mesh. We consider
# a single line of airborne data for a vertical coplanar survey geometry. Here
# the Earth's electrical properties are defined by a conductivity model
#

#####################################################################
# Create Airborne Survey
# ----------------------
#
# Here we define an airborne survey that consists of a single line of EM data
# measurements over a range of frequencies. The loop geometry is horizontal
# coplanar.
#

# Frequencies being predicted (10 Hz to 10 kHz)
freq = np.logspace(1, 4, 16)

# Defining transmitter locations
xtx, ytx, ztx = np.meshgrid(np.linspace(0, 200, 41), [0], [55])
src_locs = np.c_[mkvc(xtx), mkvc(ytx), mkvc(ztx)]
ntx = np.size(xtx)

# Define receiver locations
xrx, yrx, zrx = np.meshgrid(np.linspace(0, 200, 41), [0], [50])
rx_locs = np.c_[mkvc(xrx), mkvc(yrx), mkvc(zrx)]

source_list = []  # Create empty list to store sources

# Each unique location and frequency defines a new transmitter
for ii in range(ntx):

    # Define receivers of different types at each location. Real and imaginary
    # measurements require separate receivers. You can define the orientation of
    # the transmitters and receivers for different survey geometries.
    bzr_rec = receivers.Point_bSecondary(rx_locs[ii, :], 'z', 'real')
    bzi_rec = receivers.Point_bSecondary(rx_locs[ii, :], 'z', 'imag')
    rx_list = [bzr_rec, bzi_rec]  # must be a list

    for jj in range(len(freq)):

        # Must define the transmitter properties and associated receivers
        source_list.append(
            sources.MagDipole(rx_list, freq[jj], src_locs[ii], orientation='z')
        )

# Define the survey
#survey = survey(src_list)

###############################################################
# Create Cylindrical Mesh
# -----------------------
#
# Here we create the cylindrical mesh that will be used for this tutorial
# example.
#

hr = [(10., 50), (10., 10, 1.5)]
hz = [(10., 10, -1.5), (10., 100), (10., 10, 1.5)]

mesh = CylMesh([hr, 1, hz], x0='00C')

###############################################################
# Create Conductivity Model and Mapping for Cylindrical Mesh
# ----------------------------------------------------------
#
# Here, the model consists of a long vertical conductive pipe and a resistive
# surface layer. For this example, we will have only flat topography.
#

# Conductivity in S/m
air_val = 1e-8
background_val = 1e-1
layer_val = 1e-2
pipe_val = 1e1

# Active cells are cells below the surface
ind_active = mesh.gridCC[:, 2] < 0
mod_map = maps.InjectActiveCells(mesh, ind_active, air_val)

# Define the model
mod = background_val*np.ones(ind_active.sum())
ind_layer = (
    (mesh.gridCC[ind_active, 2] > -100.) & (mesh.gridCC[ind_active, 2] < -0)
)
mod[ind_layer] = layer_val
ind_pipe = (
    (mesh.gridCC[ind_active, 0] < 60.) &
    (mesh.gridCC[ind_active, 2] > -10000.) & (mesh.gridCC[ind_active, 2] < 0.)
)
mod[ind_pipe] = pipe_val


# Plot Conductivity Model
fig = plt.figure(figsize=(4.5, 6))

plotting_map = maps.InjectActiveCells(mesh, ind_active, np.nan)
log_mod = np.log10(mod)

ax1 = fig.add_axes([0.05, 0.05, 0.7, 0.9])
mesh.plotImage(
    plotting_map*log_mod, ax=ax1, grid=False,
    clim=(np.log10(layer_val), np.log10(pipe_val))
)
ax1.set_title('Conductivity Model (Survey in red)')

ax1.plot(rx_locs[:, 0], rx_locs[:, 2], 'r.')

ax2 = fig.add_axes([0.8, 0.05, 0.05, 0.9])
norm = mpl.colors.Normalize(vmin=np.log10(layer_val), vmax=np.log10(pipe_val))
cbar = mpl.colorbar.ColorbarBase(
    ax2, norm=norm, orientation='vertical', format="$10^{%.1f}$"
)
cbar.set_label(
    'Conductivity [S/m]', rotation=270, labelpad=15, size=12
)


######################################################
# Predicting the FEM response
# ---------------------------
#
# Here we demonstrate how to define the problem and forward model data for
# our airborne survey.
#

# Must define the mapping for the conductivity model
fwd_sim = simulation.Problem3D_b(mesh, survey=source_list, sigmaMap=mod_map, Solver=Solver)

# We defined receivers that measure the secondary B field. We must turn this
# into the H-field in A/m
mu0 = 4*np.pi*1e-7
dpred = fwd_sim.dpred(mod)/mu0

# Data are organized by transmitter then by receiver. We had nFreq transmitters
# and each transmitter had 2 receivers (real and imaginary component). So
# first we will pick out the real and imaginary data
h_real = dpred[0:len(dpred):2]
h_imag = dpred[1:len(dpred):2]

# Then we will will reshape the data.
h_real = np.reshape(h_real, (ntx, len(freq)))
h_imag = np.reshape(h_imag, (ntx, len(freq)))

# Plot the response
fig = plt.figure(figsize=(10, 5))

# Response at f = 1 Hz
ax1 = fig.add_axes([0.1, 0.05, 0.35, 0.85])
freq_ind = 0
ax1.plot(rx_locs[:, 0], h_real[:, freq_ind], 'k')
ax1.plot(rx_locs[:, 0], h_imag[:, freq_ind], 'r')
ax1.set_xlim((0, np.max(xtx)))
ax1.set_xlabel('Easting [m]')
ax1.set_ylabel('H secondary [A/m]')
ax1.set_title('Secondary field at 1 Hz')
ax1.legend(['Real', 'Imaginary'], loc='lower right')

# Response over pipe for all frequencies
ax2 = fig.add_axes([0.6, 0.05, 0.35, 0.85])
loc_ind = 0
ax2.semilogx(freq, h_real[loc_ind, :], 'k')
ax2.semilogx(freq, h_imag[loc_ind, :], 'r')
ax2.set_xlim((np.min(freq), np.max(freq)))
ax2.set_xlabel('Frequency [Hz]')
ax2.set_ylabel('H secondary [A/m]')
ax2.set_title('Secondary field over pipe')
ax2.legend(['Real', 'Imaginary'], loc='lower left')

##############################################################
# OCTREE EXAMPLE
# --------------
#
# Here we solve the 3D FDEM forward problem on an OcTree mesh for a
# resistivity model. To limit the
# computational cost, we have limited the size of the mesh. This will
# result in a decrease in numerical accuracy for the predicted fields.
#


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

fname = os.path.dirname(receivers.__file__) + '\\..\\..\\..\\tutorials\\assets\\fdem_topo.txt'
np.savetxt(fname, topo_xyz, fmt='%.4e')

#####################################################################
# Create Airborne Survey
# ----------------------
#
# For this example, the survey consists of a uniform grid of airborne
# measurements. To save time, we will only compute the response for a single
# frequency.
#

# Frequencies being predicted
freq = 100.

# Defining transmitter locations
N = 11
xtx, ytx, ztx = np.meshgrid(
    np.linspace(-200, 200, N), np.linspace(-200,200, N), [50]
)
src_locs = np.c_[mkvc(xtx), mkvc(ytx), mkvc(ztx)]
ntx = np.size(xtx)

# Define receiver locations
xrx, yrx, zrx = np.meshgrid(
    np.linspace(-200, 200, N), np.linspace(-200,200, N), [30]
)
rx_locs = np.c_[mkvc(xrx), mkvc(yrx), mkvc(zrx)]

source_list = []  # Create empty list to store sources

# Each unique location and frequency defines a new transmitter
for ii in range(ntx):

    # Define receivers of different type at each location
    bzr = receivers.Point_bSecondary(rx_locs[ii, :], 'z', 'real')
    bzi = receivers.Point_bSecondary(rx_locs[ii, :], 'z', 'imag')
    rxList = [bzr, bzi]

    source_list.append(
        sources.MagDipole(rxList, freq, src_locs[ii, :], orientation='z')
    )

###############################################################
# Create OcTree Mesh
# ------------------
#
# Here we define the OcTree mesh that is used for this example.
#

dh = 20.                                                     # base cell width
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
    mesh, rx_locs, octree_levels=[2, 4], method='radial', finalize=False
)

# Refine core mesh region
xp, yp, zp = np.meshgrid([-300., 300.], [-300., 300.], [-400., 0.])
xyz = np.c_[mkvc(xp), mkvc(yp), mkvc(zp)]
mesh = refine_tree_xyz(
    mesh, xyz, octree_levels=[0, 2, 4], method='box', finalize=False
)

mesh.finalize()

###############################################################
# Create Resistivity Model and Mapping for OcTree Mesh
# ----------------------------------------------------
#
# We can define the electrical properties of the model in terms of a 
# resistivity model. Here, the model consists of a conductive
# block within a more resistive background.
#

# Resistivity in Ohm m
air_val = 1e8
background_val = 1e2
block_val = 1e-1

# Active cells are cells below the surface
ind_active = surface2ind_topo(mesh, topo_xyz)
mod_map = maps.InjectActiveCells(mesh, ind_active, air_val)

# Define the model
mod = background_val*np.ones(ind_active.sum())
ind_block = (
    (mesh.gridCC[ind_active, 0] < 100.) & (mesh.gridCC[ind_active, 0] > -100.) &
    (mesh.gridCC[ind_active, 1] < 100.) & (mesh.gridCC[ind_active, 1] > -100.) &
    (mesh.gridCC[ind_active, 2] > -260.) & (mesh.gridCC[ind_active, 2] < -60.)
)
mod[ind_block] = block_val

# Plot Log of Resistivity Model
fig = plt.figure(figsize=(4.5, 6))

plotting_map = maps.InjectActiveCells(mesh, ind_active, np.nan)
log_mod = np.log10(mod)

ax1 = fig.add_axes([0.05, 0.05, 0.7, 0.9])
mesh.plotSlice(
    plotting_map*log_mod, normal='Y', ax=ax1, ind=int(mesh.hx.size/2),
    grid=True, clim=(np.log10(block_val), np.log10(background_val))
)
ax1.set_title('Resistivity Model at Y = 0 m')

ax2 = fig.add_axes([0.8, 0.05, 0.05, 0.9])
norm = mpl.colors.Normalize(vmin=np.log10(block_val), vmax=np.log10(background_val))
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
# Here we demonstrate that the problem and forward modeling is done with the
# exact same syntax. We need only define a resistivity mapping as opposed to
# a conductivity mapping. The object recognizes the type of mesh and solves the
# problem accordingly.
#

fwd_sim_2 = simulation.Problem3D_b(mesh, survey=source_list, rhoMap=mod_map, Solver=Solver)

mu0 = 4*np.pi*1e-7
dpred = fwd_sim_2.dpred(mod)/mu0

# Plot the response
h_real = dpred[0:-1:2]
h_imag = np.r_[dpred[1:-1:2], dpred[-1]]

fig = plt.figure(figsize=(10, 4))

# Real Component
v_max = np.max(np.abs(h_real))
ax1 = fig.add_axes([0.05, 0.05, 0.35, 0.9])
plot2Ddata(
    rx_locs[:, 0:2], h_real, ax=ax1, ncontour=30, clim=(-v_max, v_max),
    contourOpts={"cmap": "RdBu_r"}
    )
ax1.set_title('Re[$H_z$] at 100 Hz')

ax2 = fig.add_axes([0.41, 0.05, 0.02, 0.9])
norm = mpl.colors.Normalize(vmin=-v_max, vmax=v_max)
cbar = mpl.colorbar.ColorbarBase(
    ax2, norm=norm, orientation='vertical', cmap='RdBu_r'
)
cbar.set_label('$A/m$', rotation=270, labelpad=15, size=12)

# Imaginary Component
v_max = np.max(np.abs(h_imag))
ax1 = fig.add_axes([0.55, 0.05, 0.35, 0.9])
plot2Ddata(
    rx_locs[:, 0:2], h_imag, ax=ax1, ncontour=30, clim=(-v_max, v_max),
    contourOpts={"cmap": "RdBu_r"}
)
ax1.set_title('Im[$H_z$] at 100 Hz')

ax2 = fig.add_axes([0.91, 0.05, 0.02, 0.9])
norm = mpl.colors.Normalize(vmin=-v_max, vmax=v_max)
cbar = mpl.colorbar.ColorbarBase(
    ax2, norm=norm, orientation='vertical', cmap='RdBu_r'
)
cbar.set_label('$A/m$', rotation=270, labelpad=15, size=12)

plt.show()

# PRINT FOR INVERSION TUTORIAL
fname = os.path.dirname(receivers.__file__) + '\\..\\..\\..\\tutorials\\assets\\fdem_data.txt'
h_real = h_real + 1e-9*np.random.rand(len(h_real))
h_imag = h_imag + 5e-10*np.random.rand(len(h_imag))
f_vec = freq*np.ones(len(h_real))
np.savetxt(
    fname,
    np.c_[f_vec, rx_locs, h_real, h_imag],
    fmt='%.4e'
)
