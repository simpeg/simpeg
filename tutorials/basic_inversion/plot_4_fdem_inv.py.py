
"""
FDEM: Parametric Inversion
==========================

Here we invert total magnetic intensity (TMI) data to recover a magnetic
susceptibility model. We formulate the inverse problem as an iteratively
re-weighted least-squares (IRLS) optimization problem. For this tutorial, we
focus on the following:

    - Defining the survey from xyz formatted data
    - Generating a mesh based on survey geometry
    - Including surface topography
    - Defining the inverse problem (data misfit, regularization, directives)
    - Applying sensitivity weighting
    - Setting sparse and blocky norms
    - Plotting the recovered model and data misfit

Although we consider TMI data in this tutorial, the same approach
can be used to invert other types of geophysical data.
    

"""

#########################################################################
# Import modules
# --------------
#

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from discretize import TreeMesh
from discretize.utils import mkvc, refine_tree_xyz

from SimPEG.EM import FDEM
from SimPEG.Utils import plot2Ddata, ModelBuilder, surface2ind_topo
from SimPEG import (
    Maps, InvProblem, DataMisfit, Regularization, Optimization, Directives,
    Inversion, Utils
    )

try:
    from pymatsolver import Pardiso as Solver
except ImportError:
    from SimPEG import SolverLU as Solver

# sphinx_gallery_thumbnail_number = 4

#############################################
# Load Data and Plot
# ------------------
#
# Here we load and plot synthetic TMI data.
#

topo_name = os.path.dirname(FDEM.__file__) + '\\..\\..\\..\\tutorials\\assets\\fdem_topo.txt'
data_name = os.path.dirname(FDEM.__file__) + '\\..\\..\\..\\tutorials\\assets\\fdem_data.txt'
topo_xyz = np.loadtxt(str(topo_name))
dobs = np.loadtxt(str(data_name))

freq = dobs[:, 0]
rx_locs = dobs[:, 1:4]
dobs_real = dobs[:, -2]
dobs_imag = dobs[:, -1]
n_locs = np.shape(rx_locs)[0]

fig = plt.figure(figsize=(10, 4))

# Real Component
v_max = np.max(np.abs(dobs_real))
ax1 = fig.add_axes([0.05, 0.05, 0.35, 0.9])
plot2Ddata(
    rx_locs[:, 0:2], dobs_real, ax=ax1, ncontour=30, clim=(-v_max, v_max),
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
v_max = np.max(np.abs(dobs_imag))
ax1 = fig.add_axes([0.55, 0.05, 0.35, 0.9])
plot2Ddata(
    rx_locs[:, 0:2], dobs_imag, ax=ax1, ncontour=30, clim=(-v_max, v_max),
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

#############################################
# Assign Uncertainty
# ------------------

dunc_real = 1e-9*np.ones(len(dobs_real))
dunc_imag = 5e-10*np.ones(len(dobs_imag))

mu0 = 4*np.pi*1e-7
dobs = mu0*mkvc(np.c_[dobs_real, dobs_imag].T)
dunc = mu0*mkvc(np.c_[dunc_real, dunc_imag].T)

#############################################
# Defining the Survey
# -------------------
#
# Here, we define survey that will be used for all tutorial examples. Magnetic
# surveys are simple to create. The user needs an (N, 3) array to define
# the xyz positions of the observation locations. The user also needs to
# define the Earth's magnetic field intensity and orientation. Here, we
# create a basic airborne survey with a flight height of 10 m above the
# surface topography.
#

src_list = []  # Create empty list to store sources
# Each unique location and frequency defines a new transmitter
for ii in range(n_locs):

    # Define receivers of different type at each location
    bzr = FDEM.Rx.Point_bSecondary(rx_locs[ii, :], 'z', 'real')
    bzi = FDEM.Rx.Point_bSecondary(rx_locs[ii, :], 'z', 'imag')
    rxList = [bzr, bzi]
    
    src_loc = rx_locs[ii, :] + np.c_[0, 0, 20]
    src_list.append(
        FDEM.Src.MagDipole(rxList, freq[ii], src_loc, orientation='z')
    )

survey = FDEM.Survey(src_list)

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

########################################################
# Starting/Reference Model and Mapping on Tensor Mesh
# ---------------------------------------------------
#
# Here, we would create starting and/or reference models for the inversion as
# well as the mapping from the model space to the active cells. Starting and
# reference models can be a constant background value or contain a-priori
# structures. Here, the background is 1e-4 SI.
#

# Resistivity in Ohm m
air_val = 8
background_val = 2

# Active cells are cells below the surface
ind_active = surface2ind_topo(mesh, topo_xyz)
topo_map = Maps.InjectActiveCells(mesh, ind_active, 10**air_val)

exp_map = Maps.ExpMap()

fwd_map = topo_map*exp_map

# Define the model
m0 = background_val*np.ones(ind_active.sum())

# Plot Conductivity Model
fig = plt.figure(figsize=(8.5, 4))

plotting_map = Maps.InjectActiveCells(mesh, ind_active, np.nan)

ax1 = fig.add_axes([0.05, 0.05, 0.8, 0.9])
mesh.plotSlice(
    plotting_map*m0, normal='Y', ax=ax1, ind=int(mesh.hy.size/2),
    grid=True, clim=(-2, 4), pcolorOpts={'cmap': 'jet'}
)
ax1.set_title('Starting Model at Y = 0 m')

ax2 = fig.add_axes([0.87, 0.05, 0.05, 0.9])
norm = mpl.colors.Normalize(vmin=-2, vmax=4)
cbar = mpl.colorbar.ColorbarBase(
    ax2, norm=norm, orientation='vertical', cmap='jet', format="$10^{%.1f}$"
)
cbar.set_label(
    'Resistivity [Ohm m]', rotation=270, labelpad=15, size=12
)

plt.show()

##############################################
# Define the Physics
# ------------------
#
# Here, we define the physics of the gravity problem.
# 

# Define the problem. Define the cells below topography and the mapping
prob = FDEM.Problem3D_b(mesh, rhoMap=fwd_map, Solver=Solver)
# Pair the survey and problem
survey.pair(prob)

# Define the observed data and uncertainties
survey.dobs = dobs
survey.std = dunc


#######################################################################
# Define Inverse Problem
# ----------------------
#
# Here we define the inverse problem.
#


# Define the data misfit (Here we use weighted L2-norm)
dmis = DataMisfit.l2_DataMisfit(survey)
dmis.W = Utils.sdiag(1/dunc)

# Define the regularization (model objective function)
reg = Regularization.Simple(
    mesh, indActive=ind_active, mref=m0,
    alpha_s=1, alpha_x=1, alpha_y=1, alpha_z=1
)

# Create model weights based on sensitivity matrix (sensitivity weighting)
#wr = np.sum(prob.G**2., axis=0)**0.5
#wr = (wr/np.max(np.abs(wr)))
#reg.cell_weights = wr  # include sensitivity weighting in regularization

# Define how the optimization problem is solved.
opt = Optimization.ProjectedGNCG(
    maxIter=2, lower=-2., upper=4.,
    maxIterLS=20, maxIterCG=10, tolCG=1e-3
)

# Here we define the inverse problem that is to be solved
inv_prob = InvProblem.BaseInvProblem(dmis, reg, opt)

# Here we define any directive that are carried out during the inversion. Here,
# we apply the itertively re-weighted least squares.
#betaest = Directives.BetaSchedule()
betaest = Directives.BetaEstimate_ByEig()
saveDict = Directives.SaveOutputEveryIteration(save_txt=False)
#update_Jacobi = Directives.UpdatePreconditioner()

# Here we combine the inverse problem and the set of directives
inv = Inversion.BaseInversion(
    inv_prob, directiveList=[betaest, saveDict]
)

# Run the inversion
mrec = inv.run(m0)

############################################################
# Plotting True Model and Recovered Model
# ---------------------------------------
#

# Construct True Model

# Resistivity in Ohm m
air_val = 8
background_val = 2
block_val = -1

# Define the model
mtrue = background_val*np.ones(ind_active.sum())
ind_block = (
    (mesh.gridCC[ind_active, 0] < 100.) & (mesh.gridCC[ind_active, 0] > -100.) &
    (mesh.gridCC[ind_active, 1] < 100.) & (mesh.gridCC[ind_active, 1] > -100.) &
    (mesh.gridCC[ind_active, 2] > -260.) & (mesh.gridCC[ind_active, 2] < -60.)
)
mtrue[ind_block] = block_val

# Plot Conductivity Model
fig = plt.figure(figsize=(8.5, 4))

plotting_map = Maps.InjectActiveCells(mesh, ind_active, np.nan)

ax1 = fig.add_axes([0.05, 0.05, 0.8, 0.9])
mesh.plotSlice(
    plotting_map*mtrue, normal='Y', ax=ax1, ind=int(mesh.hy.size/2),
    grid=True, clim=(-2, 4), pcolorOpts={'cmap': 'jet'}
)
ax1.set_title('True Model at Y = 0 m')

ax2 = fig.add_axes([0.8, 0.05, 0.05, 0.9])
norm = mpl.colors.Normalize(vmin=-2, vmax=4)
cbar = mpl.colorbar.ColorbarBase(
    ax2, norm=norm, orientation='vertical', format="$10^{%.1f}$"
)
cbar.set_label(
    'Resistivity (Ohm m)', rotation=270, labelpad=15, size=12
)

plt.show()

# Plot Recovered Model
fig = plt.figure(figsize=(9, 4))
plotting_map = Maps.InjectActiveCells(mesh, ind_active, np.nan)

ax1 = fig.add_axes([0.05, 0.05, 0.78, 0.9])
mesh.plotSlice(
    plotting_map*mrec, normal='Y', ax=ax1, ind=int(mesh.hy.size/2), grid=True,
    clim=(-2, 4), pcolorOpts={'cmap': 'jet'}
)
ax1.set_title('Model slice at y = 0 m')

ax2 = fig.add_axes([0.85, 0.05, 0.05, 0.9])
norm = mpl.colors.Normalize(vmin=-2, vmax=4)
cbar = mpl.colorbar.ColorbarBase(
    ax2, norm=norm, orientation='vertical', cmap='jet', format='%.1e'
)
cbar.set_label('Resistivity [Ohm m]',rotation=270, labelpad=15, size=12)

plt.show()

###################################################################
# Plotting Predicted Data and Misfit
# ----------------------------------
#
#
dpred = inv_prob.dpred
d_misfit = (dobs-dpred)/dunc
data_array = np.c_[
        dobs[0:-1:2], dpred[0:-1:2], d_misfit[0:-1:2],
        np.r_[dobs[1:-1:2], dobs[-1]], np.r_[dpred[1:-1:2], dpred[-1]], np.r_[d_misfit[1:-1:2], d_misfit[-1]]
        ]

fig = plt.figure(figsize=(18, 10))
plot_title=[
        'Observed (Re[$H_z$])', 'Predicted (Re[$H_z$])', 'Normalized Misfit (Re[$H_z$])',
        'Observed (Im[$H_z$])', 'Predicted (Im[$H_z$])', 'Normalized Misfit (Im[$H_z$])'
        ]
plot_units=['nT', 'nT', '', 'nT', 'nT', '']

ax1 = 6*[None]
ax2 = 6*[None]
norm = 6*[None]
cbar = 6*[None]
cplot = 6*[None]
v_lim = [
    np.max(np.abs(data_array[:, 0])), np.max(np.abs(data_array[:, 1])), 2,
    np.max(np.abs(data_array[:, 3])), np.max(np.abs(data_array[:, 4])), 2
    ]

p = [0, 0.33, 0.66, 0, 0.33, 0.66]
q = [0.55, 0.55, 0.55, 0.05, 0.05, 0.05]

for ii in range(0, 6):
    
    ax1[ii] = fig.add_axes([p[ii]+0.03, q[ii]+0.03, 0.25, 0.4])
    cplot[ii] = plot2Ddata(
        rx_locs, data_array[:, ii], ax=ax1[ii], ncontour=30,
        clim=(-v_lim[ii], v_lim[ii]), contourOpts={"cmap": "RdBu_r"}
    )
    ax1[ii].set_title(plot_title[ii])
    
    ax2[ii] = fig.add_axes([p[ii]+0.28, q[ii]+0.03, 0.01, 0.4])
    norm[ii] = mpl.colors.Normalize(vmin=-v_lim[ii], vmax=v_lim[ii])
    cbar[ii] = mpl.colorbar.ColorbarBase(
        ax2[ii], norm=norm[ii], orientation='vertical', cmap='RdBu_r'
    )
    cbar[ii].set_label(plot_units[ii], rotation=270, labelpad=15, size=12)

plt.show()





