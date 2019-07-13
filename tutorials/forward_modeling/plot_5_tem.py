"""
Time Domain EM
==============

Here we use the module *SimPEG.EM.TDEM* to predict the fields that
result from solving 3D time-domain EM problems. For this tutorial, we focus
on the following:
    
    - How to define the transmitters and receivers
    - How to define the transmitter waveform
    - How to define the survey
    - How to solve TDEM problems on Cylindrical and OcTree meshes
    - How to include topography
    - The units of the conductivity/resistivity model and resulting data

The tutorial contains 2 examples. In the first, we compute the TEM response for
an airborne survey using a cylindrical mesh. In the second example, we compute
the TEM response using an OcTree mesh.

"""



from discretize import CylMesh, TreeMesh
from discretize.utils import mkvc, refine_tree_xyz

from SimPEG.Utils import plot2Ddata, surface2ind_topo
from SimPEG import Maps
from SimPEG.EM import TDEM

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

try:
    from pymatsolver import Pardiso as Solver
except ImportError:
    from SimPEG import SolverLU as Solver

###############################################################
# CYL-MESH EXAMPLE
# ----------------
#
# Here we solve the forward 3D TDEM problem on a cylindrical mesh. We consider
# a single line of airborne data for a vertical coplanar survey geometry. Here
# the Earth's electrical properties are defined by a resistivity model.
#

#####################################################################
# Create Airborne Survey
# ----------------------
#
# Here we define an airborne survey our airborne survey. For time domain
# problems, we must define the geometry of the source and its waveform. For
# the receivers, we define their geometry, the type of measurement and the time
# channels at which we would like to predict the response.
#

# Observation times for response (time channels)
tc = np.logspace(-4, -2, 11)

# Defining transmitter locations
xtx, ytx, ztx = np.meshgrid(np.linspace(0, 200, 41), [0], [55])
src_locs = np.c_[mkvc(xtx), mkvc(ytx), mkvc(ztx)]
ntx = np.size(xtx)

# Define receiver locations
xrx, yrx, zrx = np.meshgrid(np.linspace(0, 200, 41), [0], [50])
rx_locs = np.c_[mkvc(xrx), mkvc(yrx), mkvc(zrx)]

# Defining the transmitter waveform. Under 'TDEM.Src', there are a multitude
# of waveforms that can be defined with SimPEG (VTEM, Ramp-off etc...)
waveform = TDEM.Src.StepOffWaveform(offTime=0.)

src_list = []  # Create empty list to store sources

# Each unique location defines a new transmitter
for ii in range(ntx):
    
    # Define receivers of different type at each location.
    dbzdt_rec = TDEM.Rx.Point_dbdt(rx_locs[ii, :], tc, 'z')
    rx_list = [dbzdt_rec]  # Make a list containing all receivers even if just one
        
    # Must define the transmitter properties and associated receivers
    src_list.append(
        TDEM.Src.MagDipole(rx_list, loc=src_locs[ii], moment=1., orientation='z')
    )

# Define the survey
survey = TDEM.Survey(src_list)

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
# Create Model and Mapping for Cylindrical Mesh
# ---------------------------------------------
#
# Here, the model consists of a long vertical conductive pipe and a resistive
# surface layer. For this example, we will have only flat topography.
#

# Conductivity in S/m
air_val = 1e-8
background_val = 1e-1
layer_val = 1e-2
pipe_val = 1e1

# Active cells are cells below the surface.
ind_active = mesh.gridCC[:, 2] < 0
mod_map = Maps.InjectActiveCells(mesh, ind_active, air_val)

# Define the model
mod = background_val*np.ones(ind_active.sum())
ind_layer = (
    (mesh.gridCC[ind_active, 2] > -200.) & (mesh.gridCC[ind_active, 2] < -0)
)
mod[ind_layer] = layer_val
ind_pipe = (
    (mesh.gridCC[ind_active, 0] < 60.) &
    (mesh.gridCC[ind_active, 2] > -10000.) & (mesh.gridCC[ind_active, 2] < 0.)
)
mod[ind_pipe] = pipe_val


# Plot Conductivity Model
fig = plt.figure(figsize=(4.5, 6))

plotting_map = Maps.InjectActiveCells(mesh, ind_active, np.nan)
log_mod = np.log10(mod)  # We will plot log-conductivity

ax1 = fig.add_axes([0.05, 0.05, 0.7, 0.9])
mesh.plotImage(
    plotting_map*log_mod, ax=ax1, grid=False,
    clim=(np.log10(layer_val), np.log10(pipe_val))
)
ax1.set_title('Log-Conductivity (Survey in red)')

ax1.plot(rx_locs[:, 0], rx_locs[:, 2], 'r.')

ax2 = fig.add_axes([0.8, 0.05, 0.05, 0.9])
norm = mpl.colors.Normalize(vmin=np.log10(layer_val), vmax=np.log10(pipe_val))
cbar = mpl.colorbar.ColorbarBase(ax2, norm=norm, orientation='vertical')
cbar.set_label(
    'Log-Conductivity (log[S/m])', rotation=270, labelpad=15, size=12
)


######################################################
# Predicting the TEM response
# ---------------------------
#
# Here we demonstrate how to define the problem and forward model data for
# our airborne survey.
#

# Create the problem
prob = TDEM.Problem3D_e(mesh, sigmaMap=mod_map, Solver=Solver)

# Define the backward Euler time-stepping. Each interval of time-stepping is
# defined by (step width, n steps).
prob.timeSteps = [(1e-05, 20), (0.0001, 20), (0.001, 21)]
prob.pair(survey)

# Predict data given a model. Data are organized by transmitter, then by
# receiver then by observation time. dBdt data are in T/s.
dbdt = survey.dpred(mod)

# Plot the response
dbdt = np.reshape(dbdt, (ntx, len(tc)))

fig = plt.figure(figsize=(8, 4))

# TDEM Profile
ax1 = fig.add_axes([0.1, 0.05, 0.35, 0.85])
for ii in range(0, len(tc)):
    ax1.plot(rx_locs[:, 0], -dbdt[:, ii], 'k')
ax1.set_xlim((0, np.max(xtx)))
ax1.set_xlabel('Easting [m]')
ax1.set_ylabel('-dBz/dt [T/s]')
ax1.set_title('Airborne TDEM Profile')

# Response over pipe for all time channels
ax2 = fig.add_axes([0.6, 0.05, 0.35, 0.85])
ax2.loglog(tc, -dbdt[0, :], 'b')
ax2.loglog(tc, -dbdt[-1, :], 'r')
ax2.set_xlim((np.min(tc), np.max(tc)))
ax2.set_xlabel('time [s]')
ax2.set_ylabel('-dBz/dt [T/s]')
ax2.set_title('Decay Curve')
ax2.legend(['Over pipe','Background'], loc='upper right')

##############################################################
# OCTREE EXAMPLE
# --------------
#
# Here we solve the 3D TDEM forward problem on an OcTree mesh. For this
# example we define a log-resistivity model, predict the H-field at all
# time channels and use the VTEM waveform. To limit the
# computational cost, we have reduced the size of the problem. This will
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


#####################################################################
# Create Airborne Survey
# ----------------------
#
# For this example, the survey consists of a uniform grid of airborne
# measurements. To save time, we will compute the response for a minimal
# range of time channels
#

# time channels
n_times = 6
tc = np.logspace(-4, -3, n_times)

# Defining transmitter locations
n_tx = 20
xtx, ytx, ztx = np.meshgrid(
    np.linspace(-190, 190, n_tx), np.linspace(-190,190, n_tx), [50]
)
src_locs = np.c_[mkvc(xtx), mkvc(ytx), mkvc(ztx)]
ntx = np.size(xtx)

# Define receiver locations
xrx, yrx, zrx = np.meshgrid(
    np.linspace(-190, 190, n_tx), np.linspace(-190,190, n_tx), [30]
)
rx_locs = np.c_[mkvc(xrx), mkvc(yrx), mkvc(zrx)]

# Defining the transmitter waveform
t_wave = np.linspace(-0.002, 0, 21)
waveform = TDEM.Src.TrapezoidWaveform(
        ramp_on=np.r_[-0.002, -0.001],  ramp_off=np.r_[-0.001, 0.], offTime=0.
)
[waveform.eval(t) for t in t_wave]


src_list = []  # Create empty list to store sources

# Each unique location and tcuency defines a new transmitter
for ii in range(ntx):
    
    # Here we define receivers that measure the h-field in A/m
    h_rec = TDEM.Rx.Point_h(rx_locs[ii, :], tc, 'z')
    rx_list = [h_rec]  # Make a list containing all receivers even if just one
        
    # Must define the transmitter properties and associated receivers
    src_list.append(
        TDEM.Src.MagDipole(rx_list, loc=src_locs[ii], moment=1., orientation='z')
    )

# Define the survey
survey2 = TDEM.Survey(src_list)

###############################################################
# Create OcTree Mesh
# ------------------
#
# Here we define the OcTree mesh that is used for this example.
#

dh = 25.                                                     # base cell width
dom_width = 1600.                                            # domain width
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
xp, yp, zp = np.meshgrid([-300., 300.], [-300., 300.], [-300., 0.])
xyz = np.c_[mkvc(xp), mkvc(yp), mkvc(zp)]
mesh = refine_tree_xyz(
    mesh, xyz, octree_levels=[0, 2, 4], method='box', finalize=False
)

mesh.finalize()

###############################################################
# Create Model and Mapping for OcTree Mesh
# ----------------------------------------
#
# Here, we define the electrical properties of the Earth as a resistivity
# model. The model consists of a long vertical conductive pipe within a more
# resistive background.
#

# Log-Resistivity in log[Ohm m]
air_val = 1e8
background_val = 1e2
block_val = 1e-1

# Active cells are cells below the surface.
ind_active = surface2ind_topo(mesh, topo_xyz)
mod_map = Maps.InjectActiveCells(mesh, ind_active, air_val)

# Define the model
mod = background_val*np.ones(ind_active.sum())
ind_block = (
    (mesh.gridCC[ind_active, 0] < 100.) & (mesh.gridCC[ind_active, 0] > -100.) &
    (mesh.gridCC[ind_active, 1] < 100.) & (mesh.gridCC[ind_active, 1] > -100.) &
    (mesh.gridCC[ind_active, 2] > -260.) & (mesh.gridCC[ind_active, 2] < -60.)
)
mod[ind_block] = block_val

# Plot log-resistivity model
fig = plt.figure(figsize=(7.5, 6))

log_mod = np.log10(mod)

plotting_map = Maps.InjectActiveCells(mesh, ind_active, np.nan)

ax1 = fig.add_axes([0.05, 0.05, 0.7, 0.9])
mesh.plotSlice(
    plotting_map*log_mod, normal='Y', ax=ax1, ind=int(mesh.hx.size/2),
    grid=True, clim=(np.min(log_mod), np.max(log_mod))
)
ax1.set_title('Log-Resistivity Model at Y = 0 m')

ax2 = fig.add_axes([0.8, 0.05, 0.05, 0.9])
norm = mpl.colors.Normalize(vmin=np.min(log_mod), vmax=np.max(log_mod))
cbar = mpl.colorbar.ColorbarBase(ax2, norm=norm, orientation='vertical')
cbar.set_label(
    'Log-Resistivity (log[Ohm m])', rotation=270, labelpad=15, size=12
)


#######################################################################
# Predict Data
# ------------
#
# Here we demonstrate that the problem and forward modeling is done with the
# exact same syntax. The object recognizes the type of mesh and solves the
# problem accordingly.
#

# We defined a waveform that is on from -0.002 s to 0 s. As a result, we need
# to set the start time for the simulation at -0.002 s.
prob2 = TDEM.Problem3D_b(mesh, rhoMap=mod_map, Solver=Solver, t0=-0.002)

# Need to define time stepping for waveform and off-time
prob2.timeSteps = [(1e-4, 20), (1e-05, 10), (1e-4, 10)]

prob2.pair(survey2)

h = survey2.dpred(mod)

# Plot
h = np.reshape(h, (n_tx**2, n_times))

fig = plt.figure(figsize=(10, 4))

# H field at early time
v_max = np.max(np.abs(h[:, 0]))
ax11 = fig.add_axes([0.05, 0.05, 0.35, 0.9])
plot2Ddata(
    rx_locs[:,0:2], h[:, 0], ax=ax11, ncontour=30, clim=(-v_max, v_max),
    contourOpts={"cmap": "RdBu_r"}
    )
ax11.set_title('Hz at 0.0001 s')

ax12 = fig.add_axes([0.42, 0.05, 0.02, 0.9])
norm1 = mpl.colors.Normalize(vmin=-v_max, vmax=v_max)
cbar1 = mpl.colorbar.ColorbarBase(
    ax12, norm=norm1, orientation='vertical', cmap='RdBu_r'
)
cbar1.set_label('$A/m$', rotation=270, labelpad=15, size=12)

# H field at later times
v_max = np.max(np.abs(h[:, -1]))
ax21 = fig.add_axes([0.55, 0.05, 0.35, 0.9])
plot2Ddata(
    rx_locs[:,0:2], h[:, -1], ax=ax21, ncontour=30, clim=(-v_max, v_max),
    contourOpts={"cmap": "RdBu_r"}
    )
ax21.set_title('Hz at 0.001 s')

ax22 = fig.add_axes([0.92, 0.05, 0.02, 0.9])
norm2 = mpl.colors.Normalize(vmin=-v_max, vmax=v_max)
cbar2 = mpl.colorbar.ColorbarBase(
    ax22, norm=norm2, orientation='vertical', cmap='RdBu_r'
)
cbar2.set_label('$A/m$', rotation=270, labelpad=15, size=12)

plt.show()









