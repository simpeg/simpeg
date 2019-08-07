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
predict DC resistivity data. Next we create a chargeability model and a
background conductivity model to compute IP data.
    

"""

#########################################################################
# Import modules
# --------------
#


from discretize import TreeMesh
from discretize.utils import mkvc, refine_tree_xyz

from SimPEG.Utils import ModelBuilder, surface2ind_topo
from SimPEG import Maps
from SimPEG.EM.Static import DC, IP
from SimPEG.EM.Static.Utils import plot_pseudoSection

import os
import numpy as np
from scipy.interpolate import LinearNDInterpolator
import matplotlib as mpl
import matplotlib.pyplot as plt

try:
    from pymatsolver import Pardiso as Solver
except ImportError:
    from SimPEG import SolverLU as Solver

# sphinx_gallery_thumbnail_number = 2

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

fname = os.path.dirname(DC.__file__) + '\\..\\..\\..\\..\\tutorials\\assets\\dcip_topo.txt'
np.savetxt(fname, topo_xyz, fmt='%.4e')

#####################################################################
# Create Dipole-Dipole Survey
# ---------------------------
#
# Here we define a single EW survey line that uses a dipole-dipole configuration.
# For the source, we must define the AB electrode locations. For the receivers
# we must define the MN electrode locations. When creating DCIP surveys, it
# is important for the electrode locations NOT to lie within air cells. Here
# we shift the vertical locations of the electrodes down by a constant. The
# user may choose to do something more complex.
#

# Define all electrode locations (Src and Rx) as an (N, 3) numpy array
n_loc = 16  # Number of electrode locations along EW profile
xr = np.linspace(-300., 300., n_loc)
yr = 0.
xr, yr = np.meshgrid(xr, yr)
xr, yr = mkvc(xr.T), mkvc(yr.T)
fun_interp = LinearNDInterpolator(np.c_[xx, yy], zz)
zr = fun_interp(np.c_[xr, yr]) - 4.  # Ensure electrodes inside Earth
electrode_locs = np.c_[xr, yr, zr]

# An indexing array
k = [0] * n_loc
k = [ii for ii in range(n_loc)]

ii = 1
src_list = []  # create empty array for sources to live

while ii < n_loc:

    # AB electrode locations for source
    a_loc = electrode_locs[ii-1, :]
    b_loc = electrode_locs[ii, :]

    # MN electrode locations for receivers
    k = np.zeros(n_loc, dtype='bool')
    k[ii-1] = True
    k[ii] = True

    n_locs = electrode_locs[k == False]
    m_locs = n_locs[0:n_loc-3, :]
    n_locs = n_locs[1:n_loc-2, :]

    # Create receivers list. Define as pole or dipole. Can choose to
    # measured potential or components of electric field.
    rx_list = [DC.RxDC.Dipole(m_locs, n_locs, rxType='phi')]

    # Define the source properties and associated receivers
    src_list.append(DC.SrcDC.Dipole(rx_list, a_loc, b_loc))

    ii = ii + 1

# Define survey
survey_dc = DC.Survey(src_list)


###############################################################
# Create OcTree Mesh
# ------------------
#
# Here, we create the OcTree mesh that will be used to predict both DC
# resistivity and IP data.
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
xp, yp, zp = np.meshgrid([-600., 600.], [-300., 300.], [-500., 0.])
xyz = np.c_[mkvc(xp), mkvc(yp), mkvc(zp)]
mesh = refine_tree_xyz(
    mesh, xyz, octree_levels=[0, 2, 2], method='box', finalize=False
    )

mesh.finalize()

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
air_val = 1e-8
background_val = 1e-2
conductor_val = 1e-1
resistor_val = 1e-4

# Find active cells in forward modeling (cell below surface)
ind_active = surface2ind_topo(mesh, topo_xyz)

# Define mapping from model to active cells
nC = int(ind_active.sum())
mod_dc_map = Maps.InjectActiveCells(mesh, ind_active, air_val)

# Define model
mod_dc = background_val*np.ones(nC)

ind_conductor = ModelBuilder.getIndicesSphere(
    np.r_[-120., 0., -180.], 60., mesh.gridCC
)
ind_conductor = ind_conductor[ind_active]
mod_dc[ind_conductor] = conductor_val

ind_resistor = ModelBuilder.getIndicesSphere(
    np.r_[120., 0., -180.], 60., mesh.gridCC
)
ind_resistor = ind_resistor[ind_active]
mod_dc[ind_resistor] = resistor_val


# Plot Conductivity Model
fig = plt.figure(figsize=(8.5, 4))

plotting_map = Maps.InjectActiveCells(mesh, ind_active, np.nan)
log_mod = np.log10(mod_dc)

ax1 = fig.add_axes([0.05, 0.05, 0.8, 0.9])
mesh.plotSlice(
    plotting_map*log_mod, normal='Y', ax=ax1, ind=int(mesh.hy.size/2),
    grid=True, clim=(np.log10(resistor_val), np.log10(conductor_val))
)
ax1.set_title('Conductivity Model at Y = 0 m')

ax2 = fig.add_axes([0.87, 0.05, 0.05, 0.9])
norm = mpl.colors.Normalize(vmin=np.log10(resistor_val), vmax=np.log10(conductor_val))
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

prob = DC.Problem3D_N(mesh, sigmaMap=mod_dc_map, Solver=Solver)
prob.pair(survey_dc)

dpred_dc = survey_dc.dpred(mod_dc)

# Plot apparent conductivity pseudo-section
fig = plt.figure(figsize=(11, 5))

ax1 = fig.add_axes([0.05, 0.05, 0.8, 0.9])
plot_pseudoSection(
    survey_dc, dobs=dpred_dc, ax=ax1, survey_type='dipole-dipole',
    data_type='appConductivity', space_type='half-space',
    scale='log', sameratio=True, pcolorOpts={}
)
ax1.set_title('Apparent Conductivity [S/m]')

plt.show()

#######################################################################
# Write out dpred
# ---------------
#

survey_dc.getABMN_locations()

data_array = np.c_[
    survey_dc.a_locations,
    survey_dc.b_locations,
    survey_dc.m_locations,
    survey_dc.n_locations,
    dpred_dc*(1 + 0.05*np.random.rand(len(dpred_dc)))
    ]

fname = os.path.dirname(DC.__file__) + '\\..\\..\\..\\..\\tutorials\\assets\\dc_data.txt'
np.savetxt(fname, data_array, fmt='%.4e')


#######################################################################
# Predict IP Resistivity Data
# ---------------------------
#
# The geometry of the survey was defined earlier. Here, we use SimPEG functionality
# to make a copy for predicting IP data.
#

survey_ip = IP.from_dc_to_ip_survey(survey_dc, dim="3D")


###############################################################
# Create Chargeability Model and Mapping for OcTree Mesh
# ------------------------------------------------------
#
# Here we define the chargeability model that will be used to predict IP data.
# Here we assume that the conductive sphere is also chargeable but the resistive
# sphere is not. Here, the chargeability is defined as mV/V.
#

# Define chargeability model in mV/V
air_val = 0.
background_val = 0.
chargeable_val = 1e-1

# Find active cells in forward modeling (cell below surface)
ind_active = surface2ind_topo(mesh, topo_xyz)

# Define mapping from model to active cells
nC = int(ind_active.sum())
mod_ip_map = Maps.InjectActiveCells(mesh, ind_active, air_val)

# Define chargeability model
mod_ip = background_val*np.ones(nC)

ind_chargeable = ModelBuilder.getIndicesSphere(
    np.r_[-120., 0., -180.], 60., mesh.gridCC
)
ind_chargeable = ind_chargeable[ind_active]
mod_ip[ind_chargeable] = chargeable_val

# Plot Chargeability Model
fig = plt.figure(figsize=(8.5, 4))

ax1 = fig.add_axes([0.05, 0.05, 0.8, 0.9])
mesh.plotSlice(
    plotting_map*mod_ip, normal='Y', ax=ax1, ind=int(mesh.hy.size/2),
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
# Here we use a chargeability model and a background conductivity/resistivity
# model to predict IP data.
#

# We use the keyword argument *sigma* to define the background conductivity on
# the mesh. We could use the keyword argument *rho* to accomplish the same thing
# using a background resistivity model.
prob_ip = IP.Problem3D_N(
    mesh, etaMap=mod_ip_map, sigma=mod_dc_map*mod_dc, Solver=Solver
)
prob_ip.pair(survey_ip)

dpred_ip = survey_ip.dpred(mod_ip)

# Plot
fig = plt.figure(figsize=(11, 5))

ax1 = fig.add_axes([0.05, 0.05, 0.8, 0.9])
plot_pseudoSection(
    survey_ip, dobs=dpred_ip, V0=dpred_dc, ax=ax1, survey_type='dipole-dipole',
    data_type='appChargeability', space_type='half-space',
    scale='linear', clim=None,
)
ax1.set_title('Apparent Chargeability (mV/V)')

plt.show()

#######################################################################
# Write out dpred
# ---------------
#

survey_ip.getABMN_locations()

data_array = np.c_[
    survey_ip.a_locations,
    survey_ip.b_locations,
    survey_ip.m_locations,
    survey_ip.n_locations,
    dpred_ip*(1 + 0.05*np.random.rand(len(dpred_ip)))
    ]

fname = os.path.dirname(DC.__file__) + '\\..\\..\\..\\..\\tutorials\\assets\\ip_data.txt'
np.savetxt(fname, data_array, fmt='%.4e')