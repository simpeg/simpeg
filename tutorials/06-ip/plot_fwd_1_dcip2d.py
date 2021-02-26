# -*- coding: utf-8 -*-
"""
2.5D Forward Simulation of a DCIP Line
======================================

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
predict DC resistivity data as measured voltages. Next we create a chargeability
model and a background conductivity model to compute IP data defined as
secondary potentials. We show how DC and IP in units of Volts can be plotted on
pseudo-sections as apparent conductivities and apparent chargeabilities.


"""

#########################################################################
# Import modules
# --------------
#

from discretize import TreeMesh
from discretize.utils import mkvc, refine_tree_xyz

from SimPEG.utils import model_builder, surface2ind_topo
from SimPEG import maps, data
from SimPEG.electromagnetics.static import resistivity as dc
from SimPEG.electromagnetics.static import induced_polarization as ip
from SimPEG.electromagnetics.static.utils.static_utils import (
    generate_dcip_sources_line,
    plot_2d_pseudosection,
    convert_volts_to_resisitivities
)

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

try:
    from pymatsolver import Pardiso as Solver
except ImportError:
    from SimPEG import SolverLU as Solver

mpl.rcParams.update({'font.size': 16})
save_file = True

# sphinx_gallery_thumbnail_number = 5


###############################################################
# Defining Topography
# -------------------
#
# Here we define surface topography as an (N, 3) numpy array. Topography could
# also be loaded from a file. In our case, our survey takes place within a set
# of valleys that run North-South.
#

x_topo, y_topo = np.meshgrid(
    np.linspace(-3000, 3000, 601), np.linspace(-3000, 3000, 101)
)
z_topo = 40.*np.sin(2*np.pi*x_topo/800) - 40.
x_topo, y_topo, z_topo = mkvc(x_topo), mkvc(y_topo), mkvc(z_topo)
xyz_topo = np.c_[x_topo, y_topo, z_topo]

# Create 2D topography. Since our 3D topography only changes in the x direction,
# it is easy to define the 2D topography projected along the survey line. For
# arbitrary topography and for an arbitrary survey orientation, the user must
# define the 2D topography along the survey line.
topo_2d = np.unique(xyz_topo[:, [0, 2]], axis=0)

#####################################################################
# Create Dipole-Dipole Survey
# ---------------------------
#
# Here we define a single EW survey line that uses a dipole-dipole configuration.
# For the source, we must define the AB electrode locations. For the receivers
# we must define the MN electrode locations. Instead of creating the survey
# from scratch (see 1D example), we will use the *generat_dcip_survey_line* utility.
#

# Define survey line parameters
survey_type = "dipole-dipole"
dimension_type = "2.5D"
data_type = "volt"
end_locations = np.r_[-400., 400.]
station_separation = 40.0
num_rx_per_src = 10

# Generate source list for DC survey line
source_list = generate_dcip_sources_line(
    survey_type,
    data_type,
    dimension_type,
    end_locations,
    xyz_topo,
    num_rx_per_src,
    station_separation
)

# Define survey
dc_survey = dc.survey.Survey(source_list, survey_type=survey_type)


###############################################################
# Create OcTree Mesh
# ------------------
#
# Here, we create the OcTree mesh that will be used to predict both DC
# resistivity and IP data.
#

dh = 8  # base cell width
dom_width_x = 2400.0  # domain width x
dom_width_z = 1200.0  # domain width z
nbcx = 2 ** int(np.round(np.log(dom_width_x / dh) / np.log(2.0)))  # num. base cells x
nbcz = 2 ** int(np.round(np.log(dom_width_z / dh) / np.log(2.0)))  # num. base cells z

# Define the base mesh
hx = [(dh, nbcx)]
hz = [(dh, nbcz)]
mesh = TreeMesh([hx, hz], x0="CN")

# Mesh refinement based on topography
mesh = refine_tree_xyz(
    mesh, xyz_topo[:, [0, 2]], octree_levels=[0, 2], method="surface", finalize=False
)

# Mesh refinement near transmitters and receivers. First we need to obtain the
# set of unique electrode locations.
electrode_locations = np.c_[
    dc_survey.locations_a,
    dc_survey.locations_b,
    dc_survey.locations_m,
    dc_survey.locations_n,
]

unique_locations = np.unique(
    np.reshape(electrode_locations, (4 * dc_survey.nD, 2)), axis=0
)

mesh = refine_tree_xyz(
    mesh, unique_locations, octree_levels=[2, 4], method="radial", finalize=False
)

# Refine core mesh region
xp, zp = np.meshgrid([-800.0, 800.0], [-800.0, 0.0])
xyz = np.c_[mkvc(xp), mkvc(zp)]
mesh = refine_tree_xyz(mesh, xyz, octree_levels=[0, 2, 2], method="box", finalize=False)

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
air_conductivity = 1e-8
background_conductivity = 1e-2
conductor_conductivity = 1e-1
resistor_conductivity = 1e-3

# Find active cells in forward modeling (cell below surface)
ind_active = surface2ind_topo(mesh, xyz_topo[:, [0, 2]])

# Define mapping from model to active cells
nC = int(ind_active.sum())
conductivity_map = maps.InjectActiveCells(mesh, ind_active, air_conductivity)

# Define model
conductivity_model = background_conductivity * np.ones(nC)

ind_conductor = model_builder.getIndicesSphere(np.r_[-120.0, -160.0], 60.0, mesh.gridCC)
ind_conductor = ind_conductor[ind_active]
conductivity_model[ind_conductor] = conductor_conductivity

ind_resistor = model_builder.getIndicesSphere(np.r_[120.0, -100.0], 60.0, mesh.gridCC)
ind_resistor = ind_resistor[ind_active]
conductivity_model[ind_resistor] = resistor_conductivity


# Plot Conductivity Model
fig = plt.figure(figsize=(9, 4))

plotting_map = maps.InjectActiveCells(mesh, ind_active, np.nan)
log_mod = np.log10(conductivity_model)

ax1 = fig.add_axes([0.14, 0.15, 0.68, 0.7])
mesh.plotImage(
    plotting_map * log_mod,
    ax=ax1,
    grid=False,
    clim=(np.log10(resistor_conductivity), np.log10(conductor_conductivity)),
    pcolor_opts={"cmap": "viridis"},
)
ax1.set_title("Conductivity Model")
ax1.set_xlabel("x (m)")
ax1.set_ylabel("z (m)")

ax2 = fig.add_axes([0.84, 0.15, 0.03, 0.7])
norm = mpl.colors.Normalize(
    vmin=np.log10(resistor_conductivity), vmax=np.log10(conductor_conductivity)
)
cbar = mpl.colorbar.ColorbarBase(
    ax2, norm=norm, cmap=mpl.cm.viridis, orientation="vertical", format="$10^{%.1f}$"
)
cbar.set_label("Conductivity [S/m]", rotation=270, labelpad=15, size=12)


###############################################################
# Project Survey to Discretized Topography
# ----------------------------------------
#
# It is important that electrodes are not modeled as being in the air. Even if the
# electrodes are properly located along surface topography, they may lie above
# the discretized topography. This step is carried out to ensure all electrodes
# lie on the discretized surface.
#

dc_survey.drape_electrodes_on_topography(mesh, ind_active, option="top")


#######################################################################
# Predict DC Resistivity Data
# ---------------------------
#
# Here we predict DC resistivity data. If the keyword argument *sigmaMap* is
# defined, the simulation will expect a conductivity model. If the keyword
# argument *rhoMap* is defined, the simulation will expect a resistivity model.
#

dc_simulation = dc.simulation_2d.Simulation2DNodal(
    mesh, survey=dc_survey, sigmaMap=conductivity_map, Solver=Solver
)

# Predict the data by running the simulation. The data are the raw voltage in
# units of volts.
dpred_dc = dc_simulation.dpred(conductivity_model)

#######################################################################
# Plotting DC Data in Pseudo-Section
# ----------------------------------
#
# Here, we demonstrate how to plot 2D DC data in pseudo-section.
# First, we plot the voltages in pseudo-section as a scatter plot. This
# allows us to visualize the pseudo-sensitivity locations for our survey.
# Next, we plot the apparent conductivities in pseudo-section as a filled
# contour plot.
#

# Plot voltages pseudo-section
fig = plt.figure(figsize=(12, 5))
ax1 = fig.add_axes([0.1, 0.15, 0.75, 0.78])
plot_2d_pseudosection(
    dc_survey,
    np.abs(dpred_dc),
    'scatter',
    ax=ax1,
    scale="log",
    units="V/A",
    scatter_opts={"cmap": "viridis"},
)
ax1.set_title("Normalized Voltages")
plt.show()

# Get apparent conductivities from volts and survey geometry
apparent_conductivities = 1/convert_volts_to_resisitivities(dc_survey, dpred_dc)

# Plot apparent conductivity pseudo-section
fig = plt.figure(figsize=(12, 5))
ax1 = fig.add_axes([0.1, 0.15, 0.75, 0.78])
plot_2d_pseudosection(
    dc_survey,
    apparent_conductivities,
    'tricontourf',
    ax=ax1,
    scale="log",
    units="S/m",
    tricontourf_opts={"levels": 20, "cmap": "viridis"},
)
ax1.set_title("Apparent Conductivity")
plt.show()

#######################################################################
# Optional: Write out dpred
# -------------------------
#
# Write DC resistivity data, topography and true model
#

if save_file:

    dir_path = os.path.dirname(dc.__file__).split(os.path.sep)[:-4]
    dir_path.extend(["tutorials", "06-ip", "dcip2d"])
    dir_path = os.path.sep.join(dir_path) + os.path.sep

    # Add 5% Gaussian noise to each datum
    np.random.seed(225)
    dc_noise = 0.05 * np.abs(dpred_dc) * np.random.rand(len(dpred_dc))

    # Write out data at their original electrode locations (not shifted)
    data_array = np.c_[electrode_locations, dpred_dc + dc_noise]

    fname = dir_path + "dc_data.obs"
    np.savetxt(fname, data_array, fmt="%.4e")

    fname = dir_path + "true_conductivity.txt"
    np.savetxt(fname, conductivity_map * conductivity_model, fmt="%.4e")

    fname = dir_path + "xyz_topo.txt"
    np.savetxt(fname, xyz_topo, fmt="%.4e")

#######################################################################
# Define IP Survey
# ----------------
#
# The geometry of the survey was defined earlier. We will define the IP
# data as the secondary potential. Thus the data type is still 'volts'.
#

# Generate source list for DC survey line
source_list = generate_dcip_sources_line(
    survey_type,
    data_type,
    dimension_type,
    end_locations,
    xyz_topo,
    num_rx_per_src,
    station_separation
)

# Define survey
ip_survey = ip.survey.Survey(source_list, survey_type=survey_type)


###############################################################
# Create Chargeability Model and Mapping for OcTree Mesh
# ------------------------------------------------------
#
# Here we define the chargeability model that will be used to predict IP data.
# Here we assume that the conductive sphere is also chargeable but the resistive
# sphere is not. Here, the chargeability is defined as mV/V.
#

# Define chargeability model as intrinsic chargeability (V/V).
air_chargeability = 0.0
background_chargeability = 0.0
sphere_chargeability = 1e-1

# Find active cells in forward modeling (cells below surface)
ind_active = surface2ind_topo(mesh, xyz_topo[:, [0, 2]])

# Define mapping from model to active cells
nC = int(ind_active.sum())
chargeability_map = maps.InjectActiveCells(mesh, ind_active, air_chargeability)

# Define chargeability model
chargeability_model = background_chargeability * np.ones(nC)

ind_chargeable = model_builder.getIndicesSphere(
    np.r_[-120.0, -160.0], 60.0, mesh.gridCC
)
ind_chargeable = ind_chargeable[ind_active]
chargeability_model[ind_chargeable] = sphere_chargeability

# Plot Chargeability Model
fig = plt.figure(figsize=(9, 4))

ax1 = fig.add_axes([0.14, 0.15, 0.68, 0.7])
mesh.plotImage(
    plotting_map * chargeability_model,
    ax=ax1,
    grid=False,
    clim=(background_chargeability, sphere_chargeability),
    pcolor_opts={"cmap": "plasma"},
)
ax1.set_title("Intrinsic Chargeability")
ax1.set_xlabel("x (m)")
ax1.set_ylabel("z (m)")

ax2 = fig.add_axes([0.84, 0.15, 0.03, 0.7])
norm = mpl.colors.Normalize(vmin=background_chargeability, vmax=sphere_chargeability)
cbar = mpl.colorbar.ColorbarBase(
    ax2, norm=norm, orientation="vertical", cmap=mpl.cm.plasma
)
cbar.set_label("Intrinsic Chargeability (V/V)", rotation=270, labelpad=15, size=12)

plt.show()

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
simulation_ip = ip.simulation_2d.Simulation2DNodal(
    mesh,
    survey=ip_survey,
    etaMap=chargeability_map,
    sigma=conductivity_map * conductivity_model,
    Solver=Solver,
)

# Run forward simulation and predicted IP data. The data are the voltage (V)
dpred_ip = simulation_ip.dpred(chargeability_model)


###############################################
# Plot 2D IP Data in Pseudosection
# --------------------------------
#
# We want to plot apparent chargeability. To accomplish this, we must normalize the IP
# voltage by the DC voltage. This is then multiplied by 1000 so that our
# apparent chargeability is in units mV/V.

fig = plt.figure(figsize=(12, 12))

# Plot apparent conductivity
ax1 = fig.add_axes([0.1, 0.575, 0.72, 0.4])
cax1 = fig.add_axes([0.84, 0.575, 0.05, 0.4])
plot_2d_pseudosection(
    dc_survey,
    apparent_conductivities,
    'tricontourf',
    ax=ax1,
    cax=cax1,
    scale="log",
    units="S/m",
    mask_topography=True,
    tricontourf_opts={"levels": 20, "cmap": "viridis"},
)
ax1.set_title("Apparent Conductivity")

# Convert from voltage measurement to apparent chargeability by normalizing by
# the DC voltage
apparent_chargeability = dpred_ip / dpred_dc

ax2 = fig.add_axes([0.1, 0.05, 0.72, 0.4])
cax2 = fig.add_axes([0.84, 0.075, 0.05, 0.4])
plot_2d_pseudosection(
    ip_survey,
    apparent_chargeability,
    'tricontourf',
    ax=ax2,
    cax=cax2,
    scale="linear",
    units="V/V",
    mask_topography=True,
    tricontourf_opts={"levels": 20, "cmap": "plasma"},
)
ax2.set_title("Apparent Chargeability (V/V)")

plt.show()

#######################################################################
# Write out dpred
# ---------------
#
# Write data and true model
#

if save_file:

    # Add 1% Gaussian noise based on the DC data (not the IP data)
    ip_noise = 0.01 * np.abs(dpred_dc) * np.random.rand(len(dpred_ip))

    data_array = np.c_[electrode_locations, dpred_ip + ip_noise]

    fname = dir_path + "ip_data.obs"
    np.savetxt(fname, data_array, fmt="%.4e")

    fname = dir_path + "true_chargeability.txt"
    np.savetxt(fname, chargeability_map * chargeability_model, fmt="%.4e")
