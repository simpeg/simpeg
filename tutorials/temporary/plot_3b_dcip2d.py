# -*- coding: utf-8 -*-
"""
DC and IP in 2.5D
=================

Here we use the module *SimPEG.electromagnetics.static.resistivity* to predict
DC resistivity data and the module *SimPEG.electromagnetics.static.induced_polarization*
to predict IP data for a dipole-dipole survey. In this tutorial, we focus on
the following:

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

from SimPEG.utils import ModelBuilder, surface2ind_topo
from SimPEG import maps, data
from SimPEG.electromagnetics.static import resistivity as dc
from SimPEG.electromagnetics.static import induced_polarization as ip
from SimPEG.electromagnetics.static.utils import generate_dcip_survey_line, plot_pseudoSection

import os
import numpy as np
from scipy.interpolate import interp1d
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

x_topo, y_topo = np.meshgrid(np.linspace(-3000, 3000, 101), np.linspace(-3000, 3000, 101))
z_topo = (1/np.pi)*80*(-np.pi/2 + np.arctan((np.abs(x_topo) - 500.)/30.))
x_topo, y_topo, z_topo = mkvc(x_topo), mkvc(y_topo), mkvc(z_topo)
xyz_topo = np.c_[x_topo, y_topo, z_topo]


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


survey_type = 'dipole-dipole'
end_locations = np.r_[-400., 400]
xyz_topo = 0.
station_separation = 40.
dipole_separation = 20.
n = 8 


dc_survey = generate_dcip_survey_line(
    survey_type, end_locations, xyz_topo,
    station_separation, dipole_separation, n,
    dim_flag='2.5D', sources_only=False
    )

dc_survey.getABMN_locations()
unique_locations = np.unique(
        np.r_[dc_survey.a_locations, dc_survey.b_locations], axis=0
        )

###############################################################
# Create OcTree Mesh
# ------------------
#
# Here, we create the OcTree mesh that will be used to predict both DC
# resistivity and IP data.
#

dh = 5.                                                    # base cell width
dom_width_x = 3000.                                         # domain width x
dom_width_z = 1500.                                         # domain width z
nbcx = 2**int(np.round(np.log(dom_width_x/dh)/np.log(2.)))  # num. base cells x
nbcz = 2**int(np.round(np.log(dom_width_z/dh)/np.log(2.)))  # num. base cells z

# Define the base mesh
hx = [(dh, nbcx)]
hz = [(dh, nbcz)]
mesh = TreeMesh([hx, hz], x0='CN')


# Mesh refinement near transmitters and receivers
mesh = refine_tree_xyz(
    mesh, unique_locations, octree_levels=[2, 4], method='radial', finalize=False
)

# Refine core mesh region
xp, zp = np.meshgrid([-600., 600.], [-500., 0.])
xyz = np.c_[mkvc(xp), mkvc(zp)]
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
resistor_val = 1e-5

# Find active cells in forward modeling (cell below surface)
ind_active = np.ones(mesh.nC, dtype="bool")

# Define mapping from model to active cells
nC = int(ind_active.sum())
mod_dc_map = maps.InjectActiveCells(mesh, ind_active, air_val)

# Define model
mod_dc = background_val*np.ones(nC)

ind_conductor = ModelBuilder.getIndicesSphere(
    np.r_[-120., -100.], 60., mesh.gridCC
)
ind_conductor = ind_conductor[ind_active]
mod_dc[ind_conductor] = conductor_val

ind_resistor = ModelBuilder.getIndicesSphere(
    np.r_[120., -100.], 60., mesh.gridCC
)
ind_resistor = ind_resistor[ind_active]
mod_dc[ind_resistor] = resistor_val


# Plot Conductivity Model
fig = plt.figure(figsize=(8.5, 4))

plotting_map = maps.InjectActiveCells(mesh, ind_active, np.nan)
log_mod = np.log10(mod_dc)

ax1 = fig.add_axes([0.05, 0.05, 0.8, 0.9])
mesh.plotImage(
    plotting_map*log_mod, ax=ax1, grid=False,
    clim=(np.log10(resistor_val), np.log10(conductor_val))
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

dc_simulation = dc.simulation_2d.Problem2D_N(mesh, survey=dc_survey, sigmaMap=mod_dc_map, Solver=Solver)

dpred_dc = dc_simulation.dpred(mod_dc)

dc_data = data.Data(dc_survey, dobs=dpred_dc)


# Plot apparent conductivity pseudo-section
fig = plt.figure(figsize=(11, 5))

ax1 = fig.add_axes([0.05, 0.05, 0.8, 0.9])
plot_pseudoSection(
    dc_data, ax=ax1, survey_type='dipole-dipole',
    data_type='appConductivity', space_type='half-space', scale='log'
)
ax1.set_title('Apparent Conductivity [S/m]')

plt.show()

#######################################################################
# Write out dpred
# ---------------
#

#survey_dc.getABMN_locations()
#
#data_array = np.c_[
#    survey_dc.a_locations,
#    survey_dc.b_locations,
#    survey_dc.m_locations,
#    survey_dc.n_locations,
#    dpred_dc*(1 + 0.05*np.random.rand(len(dpred_dc)))
#    ]
#
#fname = os.path.dirname(dc.__file__) + '\\..\\..\\..\\..\\tutorials\\assets\\dc2D_data.txt'
#np.savetxt(fname, data_array, fmt='%.4e')


#######################################################################
# Predict IP Resistivity Data
# ---------------------------
#
# The geometry of the survey was defined earlier. Here, we use SimPEG functionality
# to make a copy for predicting IP data.
#

ip_survey = ip.from_dc_to_ip_survey(dc_survey, dim="2.5D")


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
ind_active = np.ones(mesh.nC, dtype="bool")

# Define mapping from model to active cells
nC = int(ind_active.sum())
mod_ip_map = maps.InjectActiveCells(mesh, ind_active, air_val)

# Define chargeability model
mod_ip = background_val*np.ones(nC)

ind_chargeable = ModelBuilder.getIndicesSphere(
    np.r_[-120., -180.], 60., mesh.gridCC
)
ind_chargeable = ind_chargeable[ind_active]
mod_ip[ind_chargeable] = chargeable_val

# Plot Chargeability Model
fig = plt.figure(figsize=(8.5, 4))

ax1 = fig.add_axes([0.05, 0.05, 0.8, 0.9])
mesh.plotImage(plotting_map*mod_ip, ax=ax1, grid=False)
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
simulation_ip = ip.simulation.Problem3D_N(
    mesh, survey=ip_survey, etaMap=mod_ip_map, sigma=mod_dc_map*mod_dc,
    Solver=Solver
)

dpred_ip = simulation_ip.dpred(mod_ip)

ip_data = data.Data(ip_survey, dobs=dpred_ip)

# Plot
fig = plt.figure(figsize=(11, 5))

ax1 = fig.add_axes([0.05, 0.05, 0.8, 0.9])
plot_pseudoSection(
    ip_survey, dobs=dpred_ip, V0=dpred_dc, ax=ax1, survey_type='dipole-dipole',
    data_type='appChargeability', space_type='half-space',
    scale='linear', clim=None,
)
ax1.set_title('Apparent Chargeability (mV/V)')

plt.show()

#######################################################################
# Write out dpred
# ---------------
#

#survey_ip.getABMN_locations()
#
#data_array = np.c_[
#    survey_ip.a_locations,
#    survey_ip.b_locations,
#    survey_ip.m_locations,
#    survey_ip.n_locations,
#    dpred_ip*(1 + 0.05*np.random.rand(len(dpred_ip)))
#    ]
#
#fname = os.path.dirname(DC.__file__) + '\\..\\..\\..\\..\\tutorials\\assets\\ip_data.txt'
#np.savetxt(fname, data_array, fmt='%.4e')