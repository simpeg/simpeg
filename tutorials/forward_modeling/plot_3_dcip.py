# -*- coding: utf-8 -*-
"""
DCIP
====

Here we use the module *SimPEG.EM.Static.DC* to predict DC resistivity data
and the module *SimPEG.EM.Static.IP* to predict IP data for a dipole-dipole
survey. In this tutorial, we focus on the following: 
    
    - How to define the survey
    - How to define the problem
    - How to predict DC resistivity data for a synthetic resistivity model
    - How to predict IP data for a synthetic chargeability model
    - How to include surface topography
    - The units of the models and resulting data

This tutorial is split into two parts. First we create a resistivity model and
predict the static electrical response. Next we create a chargeability model.
We use the static

"""

#########################################################################
# Import modules
# --------------
#


from discretize import TensorMesh, CylMesh, TreeMesh
from discretize.utils import mkvc, refine_tree_xyz

from SimPEG.Utils import plot2Ddata, ModelBuilder, surface2ind_topo, matutils
from SimPEG import Maps
from SimPEG.EM.Static import DC, IP
from SimPEG.EM.Static.Utils import gen_DCIPsurvey, convertObs_DC3D_to_2D, plot_pseudoSection

import numpy as np
from scipy.interpolate import LinearNDInterpolator
import matplotlib as mpl
import matplotlib.pyplot as plt
from discretize.utils import meshutils

try:
    from pymatsolver import Pardiso as Solver
except ImportError:
    from SimPEG import SolverLU as Solver


###############################################################
# Defining Topography
# -------------------
#
# Here we define surface topography as an (N, 3) numpy array. Topography could
# also be loaded from a file. In our case, our survey takes place in a valley
# that runs North-South.
#

xx, yy = np.meshgrid(np.linspace(-3000, 3000, 101), np.linspace(-3000, 3000, 101))
zz = (1/np.pi)*80*(-np.pi/2 + np.arctan((np.abs(xx) - 500.)/30.))
xx, yy, zz = mkvc(xx), mkvc(yy), mkvc(zz)
topo_xyz = np.c_[xx, yy, zz]


#####################################################################
# Create Dipole-Dipole Survey
# ---------------------------
#

# Define all electrode locations (Src and Rx) as an (N, 3) numpy array
N = 31  # Number of points along EW profile
xr = np.linspace(-300., 300., N)
yr = 0.
xr, yr = np.meshgrid(xr, yr)
xr, yr = mkvc(xr.T), mkvc(yr.T)
fun_interp = LinearNDInterpolator(np.c_[xx, yy], zz)
zr = fun_interp(np.c_[xr, yr]) - 4.  # Ensure electrodes inside Earth
electrode_locs = np.c_[xr, yr, zr]


k = [0] * N
k = [ii for ii in range(N)]

ii = 1
src_list = []

while ii < N:
    
    a_loc = electrode_locs[ii-1, :]
    b_loc = electrode_locs[ii, :]
    
    k = np.zeros(N, dtype='bool')
    k[ii-1] = True
    k[ii] = True
    
    n_locs = electrode_locs[k == False]
    m_locs = n_locs[0:N-3, :]
    n_locs = n_locs[1:N-2, :]
    
    rx_list = [DC.RxDC.Dipole(m_locs, n_locs, rxType='phi')]
    src_list.append(DC.SrcDC.Dipole(rx_list, a_loc, b_loc))
    
    ii = ii + 1
    
survey = DC.Survey(src_list)


###############################################################
# Create OcTree Mesh
# ------------------
#
#

dh = 20.                                                    # base cell width
dom_width_x = 3000.                                         # domain width x
dom_width_y = 3000.                                         # domain width y
dom_width_z = 1500.                                         # domain width z
nbcx = 2**int(np.round(np.log(dom_width_x/dh)/np.log(2.)))  # num. base cells x
nbcy = 2**int(np.round(np.log(dom_width_y/dh)/np.log(2.)))  # num. base cells y
nbcz = 2**int(np.round(np.log(dom_width_z/dh)/np.log(2.)))  # num. base cells z

# Define the base mesh
hx = [(dh, nbcx)]
hy = [(dh, nbcy)]
hz = [(dh, nbcz)]
mesh = TreeMesh([hx, hy, hz], x0='CCN')

# Mesh refinement based on topography
mesh = refine_tree_xyz(
    mesh, topo_xyz, octree_levels=[0, 0, 0, 0, 1], method='surface', finalize=False
)

# Mesh refinement near transmitters and receivers
mesh = refine_tree_xyz(
    mesh, electrode_locs, octree_levels=[2, 4], method='radial', finalize=False
)

# Refine core mesh region
xp, yp, zp = np.meshgrid([-600., 600.], [-300., 300.], [-300., 0.])
xyz = np.c_[mkvc(xp), mkvc(yp), mkvc(zp)]
mesh = refine_tree_xyz(
    mesh, xyz, octree_levels=[0, 2, 2], method='box', finalize=False
    )

mesh.finalize()

###############################################################
# Create Conductivity Model and Mapping for OcTree Mesh
# -----------------------------------------------------
#
#

# Define density contrast values for each unit in g/cc
air_val = 1e-8
background_val = 1e-2
conductor_val = 1e-1
resistor_val = 1e-4

# Find active cells in forward modeling (cell below surface)
ind_active = surface2ind_topo(mesh, topo_xyz)

# Define mapping from model to active cells
nC = int(ind_active.sum())
mod_map = Maps.InjectActiveCells(mesh, ind_active, air_val)

# Define model
mod = background_val*np.ones(nC)

ind_conductor = ModelBuilder.getIndicesSphere(
    np.r_[-120., 0., -180.], 60., mesh.gridCC
)
ind_conductor = ind_conductor[ind_active]
mod[ind_conductor] = conductor_val

ind_resistor = ModelBuilder.getIndicesSphere(
    np.r_[120., 0., -180.], 60., mesh.gridCC
)
ind_resistor = ind_resistor[ind_active]
mod[ind_resistor] = resistor_val


# Plot Conductivity Model
fig = plt.figure(figsize=(8.5, 4))

plotting_map = Maps.InjectActiveCells(mesh, ind_active, np.nan)
log_mod = np.log10(mod)

ax1 = fig.add_axes([0.05, 0.05, 0.8, 0.9])
mesh.plotSlice(
    plotting_map*log_mod, normal='Y', ax=ax1, ind=int(mesh.hy.size/2),
    grid=True, clim=(np.log10(resistor_val), np.log10(conductor_val))
)
ax1.set_title('Log-Conductivity Model at Y = 0 m')

ax2 = fig.add_axes([0.87, 0.05, 0.05, 0.9])
norm = mpl.colors.Normalize(vmin=np.log10(resistor_val), vmax=np.log10(conductor_val))
cbar = mpl.colorbar.ColorbarBase( ax2, norm=norm, orientation='vertical')
cbar.set_label(
    'Log-Conductivity (log[S/m])', rotation=270, labelpad=15, size=12
)


#######################################################################
# Predict DC Resistivity Data
# ---------------------------
#

prob = DC.Problem3D_N(mesh, sigmaMap=mod_map, Solver=Solver)
prob.pair(survey)

data_dc = survey.dpred(mod)

# Plot
fig = plt.figure(figsize=(11, 5))

ax1 = fig.add_axes([0.05, 0.05, 0.8, 0.9])
plot_pseudoSection(
    survey, dobs=data_dc, ax=ax1, survey_type='dipole-dipole',
    data_type='appConductivity', space_type='half-space',
    scale='log', sameratio=True, pcolorOpts={}
)
ax1.set_title('Apparent Conductivity [S/m]')

plt.show()


#######################################################################
# Predict IP Resistivity Data
# ---------------------------
#

survey_ip = IP.from_dc_to_ip_survey(survey, dim="3D")


###############################################################
# Create Chargeability Model and Mapping for OcTree Mesh
# ------------------------------------------------------
#
#

# Define density contrast values for each unit in g/cc
air_val = 0.
background_val = 0.
chargeable_val = 1e-1

# Find active cells in forward modeling (cell below surface)
ind_active = surface2ind_topo(mesh, topo_xyz)

# Define mapping from model to active cells
nC = int(ind_active.sum())
modata_ip_map = Maps.InjectActiveCells(mesh, ind_active, air_val)

# Define chargeability model
modata_ip = background_val*np.ones(nC)

ind_chargeable = ModelBuilder.getIndicesSphere(
    np.r_[-120., 0., -180.], 60., mesh.gridCC
)
ind_chargeable = ind_chargeable[ind_active]
modata_ip[ind_chargeable] = chargeable_val

# Plot Chargeability Model
fig = plt.figure(figsize=(8.5, 4))

ax1 = fig.add_axes([0.05, 0.05, 0.8, 0.9])
mesh.plotSlice(
    plotting_map*modata_ip, normal='Y', ax=ax1, ind=int(mesh.hy.size/2),
    grid=True
)
ax1.set_title('Chargeability at Y = 0 m')

ax2 = fig.add_axes([0.87, 0.05, 0.05, 0.9])
norm = mpl.colors.Normalize(vmin=0, vmax=chargeable_val)
cbar = mpl.colorbar.ColorbarBase(ax2, norm=norm, orientation='vertical')
cbar.set_label(
    'Chargeability (mV/V)', rotation=270, labelpad=15, size=12
)

#######################################################################
# Predict IP Data
# ---------------
#

rho_map = Maps.InjectActiveCells(mesh, ind_active, 8)


prob_ip = IP.Problem3D_N(
    mesh, etaMap=modata_ip_map, sigma = mod_map*mod, 
        Solver=Solver
)
prob_ip.pair(survey_ip)

data_ip = survey_ip.dpred(modata_ip)

# Plot
fig = plt.figure(figsize=(11, 5))

ax1 = fig.add_axes([0.05, 0.05, 0.8, 0.9])
plot_pseudoSection(
    survey, dobs=data_ip, V0=data_dc, ax=ax1, survey_type='dipole-dipole',
    data_type='appChargeability', space_type='half-space',
    scale='linear', clim=None,
)
ax1.set_title('Apparent Chargeability')

plt.show()
















