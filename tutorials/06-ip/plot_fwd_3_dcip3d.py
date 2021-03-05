# -*- coding: utf-8 -*-
"""
DC Resistivity Forward Simulation in 3D
=======================================

Here we use the module *SimPEG.electromagnetics.static.resistivity* to predict
DC resistivity data on an OcTree mesh. In this tutorial, we focus on the following:

    - How to define the survey
    - How to definine a tree mesh based on the survey geometry
    - How to define the forward simulation
    - How to predict DC resistivity data for a synthetic conductivity model
    - How to include surface topography
    - The units of the model and resulting data
    - Plotting DC resistivity data in 3D


In this case, we simulate dipole-dipole data for one East-West line and two
North-South lines.

"""

#########################################################################
# Import modules
# --------------
#

from discretize import TreeMesh
from discretize.utils import mkvc, refine_tree_xyz

from SimPEG.utils import model_builder, surface2ind_topo
from SimPEG.utils.io_utils.io_utils_electromagnetics import write_dcip_xyz
from SimPEG import maps, data
from SimPEG.electromagnetics.static import resistivity as dc
from SimPEG.electromagnetics.static import induced_polarization as ip
from SimPEG.electromagnetics.static.utils.static_utils import (
    generate_dcip_sources_line,
    apparent_resistivity_from_voltage
)

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

try:
    from pymatsolver import Pardiso as Solver
except ImportError:
    from SimPEG import SolverLU as Solver

try:
    import plotly
    from SimPEG.electromagnetics.static.utils.static_utils import plot_3d_pseudosection
    has_plotly = True
except:
    has_plotly = False
    pass

mpl.rcParams.update({"font.size": 16})
save_file = True

# sphinx_gallery_thumbnail_number = 4


###############################################################
# Defining Topography
# -------------------
#
# Here we define surface topography as an (N, 3) numpy array. Topography could
# also be loaded from a file. In our case, our survey takes place in circular
# depression.
#

x_topo, y_topo = np.meshgrid(
    np.linspace(-2000, 2000, 161), np.linspace(-2000, 2000, 161)
)
s = np.sqrt(x_topo**2 + y_topo**2)
z_topo = (1 / np.pi) * 140 * (-np.pi / 2 + np.arctan((s - 600.0) / 80.0))
x_topo, y_topo, z_topo = mkvc(x_topo), mkvc(y_topo), mkvc(z_topo)
xyz_topo = np.c_[x_topo, y_topo, z_topo]


#####################################################################
# Create Dipole-Dipole Survey
# ---------------------------
#
# Here we define 3 survey lines that use a dipole-dipole electrode configuration.
# For each source, we must define the AB electrode locations. For each receiver
# we must define the MN electrode location. Instead of creating the survey
# from scratch (see 1D example), we will use the *generat_dcip_sources_line* utility.
#

# Define the parameters for each survey line
survey_type = "dipole-dipole"
data_type = "volt"
dimension_type = "3D"
end_locations_list = [
    np.r_[-1000.0, 1000.0, 0.0, 0.0],
    np.r_[-350.0, -350.0, -1000.0, 1000.0],
    np.r_[350.0, 350.0, -1000.0, 1000.0],
]
station_separation = 100.0
num_rx_per_src = 10

# The source lists for each line can be appended to create the source
# list for the whole survey.
source_list = []
for ii in range(0, len(end_locations_list)):
    source_list += generate_dcip_sources_line(
        survey_type,
        data_type,
        dimension_type,
        end_locations_list[ii],
        xyz_topo,
        num_rx_per_src,
        station_separation,
    )

# Define the survey
dc_survey = dc.survey.Survey(source_list)


###############################################################
# Create OcTree Mesh
# ------------------
#
# Here, we create the OcTree mesh that will be used to predict DC data.
#

dh = 25.0  # base cell width
dom_width_x = 6000.0  # domain width x
dom_width_y = 6000.0  # domain width y
dom_width_z = 4000.0  # domain width z
nbcx = 2 ** int(np.round(np.log(dom_width_x / dh) / np.log(2.0)))  # num. base cells x
nbcy = 2 ** int(np.round(np.log(dom_width_y / dh) / np.log(2.0)))  # num. base cells y
nbcz = 2 ** int(np.round(np.log(dom_width_z / dh) / np.log(2.0)))  # num. base cells z

# Define the base mesh
hx = [(dh, nbcx)]
hy = [(dh, nbcy)]
hz = [(dh, nbcz)]
mesh = TreeMesh([hx, hy, hz], x0="CCN")

# Mesh refinement based on topography
k = np.sqrt(np.sum(xyz_topo[:, 0:2]**2, axis=1)) < 1200
mesh = refine_tree_xyz(
    mesh, xyz_topo[k, :], octree_levels=[0, 4, 8, 4], method="surface", finalize=False
)

# Mesh refinement near sources and receivers. First we need to obtain the
# set of unique electrode locations.
electrode_locations = np.r_[
    dc_survey.locations_a, dc_survey.locations_b, dc_survey.locations_m, dc_survey.locations_n
]

unique_locations = np.unique(electrode_locations, axis=0)

mesh = refine_tree_xyz(
    mesh, unique_locations, octree_levels=[4, 8, 4], method="radial", finalize=False
)

mesh.finalize()

###############################################################
# Create Conductivity Model and Mapping for OcTree Mesh
# -----------------------------------------------------
#
# Here we define the conductivity model that will be used to predict DC
# resistivity data. The model consists of a conductive block and a
# resistive block within a moderately conductive background. Note that
# you can carry through this work flow with a resistivity model if desired.
#

# Define conductivity model in S/m (or resistivity model in Ohm m)
air_value = 1e-8
background_value = 1e-2
conductor_value = 1e-1
resistor_value = 1e-4

# Find active cells in forward modeling (cell below surface)
ind_active = surface2ind_topo(mesh, xyz_topo)

# Define mapping from model to active cells
nC = int(ind_active.sum())
conductivity_map = maps.InjectActiveCells(mesh, ind_active, air_value)

# Define model
conductivity_model = background_value * np.ones(nC)

ind_conductor = (
    (mesh.gridCC[ind_active, 0] > -500.0)
    & (mesh.gridCC[ind_active, 0] < -150.0)
    & (mesh.gridCC[ind_active, 1] > -400.0)
    & (mesh.gridCC[ind_active, 1] < 400.0)
    & (mesh.gridCC[ind_active, 2] > -550.0)
    & (mesh.gridCC[ind_active, 2] < -200.0)
)
conductivity_model[ind_conductor] = conductor_value

ind_resistor = (
    (mesh.gridCC[ind_active, 0] > 150.0)
    & (mesh.gridCC[ind_active, 0] < 500.0)
    & (mesh.gridCC[ind_active, 1] > -400.0)
    & (mesh.gridCC[ind_active, 1] < 400.0)
    & (mesh.gridCC[ind_active, 2] > -550.0)
    & (mesh.gridCC[ind_active, 2] < -200.0)
)
conductivity_model[ind_resistor] = resistor_value

# Plot Conductivity Model
fig = plt.figure(figsize=(11, 4))

plotting_map = maps.InjectActiveCells(mesh, ind_active, np.nan)
log_mod = np.log10(conductivity_model)

ax1 = fig.add_axes([0.1, 0.12, 0.7, 0.78])
mesh.plotSlice(
    plotting_map * log_mod,
    ax=ax1,
    normal="Y",
    ind=int(len(mesh.hy) / 2),
    grid=True,
    clim=(np.log10(resistor_value), np.log10(conductor_value)),
    pcolor_opts={"cmap": mpl.cm.viridis},
)
ax1.set_title("Conductivity Model")
ax1.set_xlabel("x (m)")
ax1.set_ylabel("z (m)")
ax1.set_xlim([-2000, 2000])
ax1.set_ylim([-2000, 0])

ax2 = fig.add_axes([0.83, 0.12, 0.03, 0.78])
norm = mpl.colors.Normalize(
    vmin=np.log10(resistor_value), vmax=np.log10(conductor_value)
)
cbar = mpl.colorbar.ColorbarBase(
    ax2, cmap=mpl.cm.viridis, norm=norm, orientation="vertical", format="$10^{%.1f}$"
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

dc_simulation = dc.simulation.Simulation3DNodal(
    mesh, survey=dc_survey, sigmaMap=conductivity_map, Solver=Solver
)

# Predict the data by running the simulation. The data are the raw voltage in
# units of volts.
dpred_dc = dc_simulation.dpred(conductivity_model)

#######################################################################
# Plot DC3D Pseudosection
# -----------------------
#
# Here we demonstrate how 3D DC resistivity data can be represented on a 3D
# pseudosection plot. To use this utility, you must have Python's *plotly*
# package. Here, we represent the data as apparent conductivities.
#

# Convert predicted data to apparent conductivities
apparent_conductivity = 1 / apparent_resistivity_from_voltage(
    dc_survey, dpred_dc,
)

if has_plotly:

    fig = plot_3d_pseudosection(
        dc_survey,
        apparent_conductivity,
        scale='log',
        units='S/m',
    )

    fig.update_layout(
        title_text='Apparent Conductivity',
        title_x=0.5,
        title_font_size=24,
        width=650,
        height=500,
        scene_camera=dict(center=dict(x=0, y=0, z=-0.4))
    )
        
    plotly.io.show(fig)

else:
    print("INSTALL 'PLOTLY' TO VISUALIZE 3D PSEUDOSECTIONS")

#######################################################################
# Optional: Write Predicted DC Data
# ---------------------------------
#
# Write DC resistivity data, topography and true model
#

if save_file:

    dir_path = os.path.dirname(dc.__file__).split(os.path.sep)[:-4]
    dir_path.extend(["tutorials", "06-ip", "dcip3d"])
    dir_path = os.path.sep.join(dir_path) + os.path.sep

    # Add 5% Gaussian noise to each datum
    np.random.seed(433)
    std = 0.05 * np.abs(dpred_dc)
    noise = std * np.random.rand(len(dpred_dc))
    dobs = dpred_dc + noise
    
    # Create dictionary that stores line IDs
    N = int(dc_survey.nD/3)
    lineID = np.r_[np.ones(N), 2*np.ones(N), 3*np.ones(N)]
    out_dict = {'LINEID': lineID}
    
    # Write out data at their original electrode locations (not shifted)
    data_obj = data.Data(dc_survey, dobs=dobs, standard_deviation=std)
    
    fname = dir_path + "dc_data.xyz"
    write_dcip_xyz(
        fname,
        data_obj,
        data_header='V/A',
        uncertainties_header='UNCERT',
        out_dict=out_dict
    )

    fname = dir_path + "true_conductivity.txt"
    np.savetxt(fname, conductivity_map * conductivity_model, fmt="%.4e")

    fname = dir_path + "xyz_topo.txt"
    np.savetxt(fname, xyz_topo, fmt="%.4e")


#######################################################################
# Define IP Survey
# ----------------
#
# The geometry of the survey was defined earlier. We will define the IP
# data as apparent chargeabilities. Thus the data type is still 'V/V'.
#

# Generate source list for IP survey lines
source_list = []
for ii in range(0, len(end_locations_list)):
    source_list += generate_dcip_sources_line(
        survey_type,
        "apparent_chargeability",
        dimension_type,
        end_locations_list[ii],
        xyz_topo,
        num_rx_per_src,
        station_separation,
    )

# Define survey
ip_survey = ip.survey.Survey(source_list, survey_type=survey_type)

# Drape to discretized topography as before
ip_survey.drape_electrodes_on_topography(mesh, ind_active, option="top")

###############################################################
# Create Chargeability Model and Mapping for OcTree Mesh
# ------------------------------------------------------
#
# Here we define the chargeability model that will be used to predict IP data.
# Here we assume that the conductive sphere is also chargeable but the resistive
# sphere is not. Here, the chargeability is defined as mV/V.
#

# Define conductivity model in S/m (or resistivity model in Ohm m)
air_value = 1e-8
background_value = 1e-6
chargeable_value = 1e-1

# Define mapping from model to active cells
chargeability_map = maps.InjectActiveCells(mesh, ind_active, air_value)

# Define model
chargeability_model = background_value * np.ones(nC)

ind_chargeable = (
    (mesh.gridCC[ind_active, 0] > -500.0)
    & (mesh.gridCC[ind_active, 0] < -150.0)
    & (mesh.gridCC[ind_active, 1] > -400.0)
    & (mesh.gridCC[ind_active, 1] < 400.0)
    & (mesh.gridCC[ind_active, 2] > -550.0)
    & (mesh.gridCC[ind_active, 2] < -200.0)
)
chargeability_model[ind_conductor] = chargeable_value

# Plot Conductivity Model
fig = plt.figure(figsize=(11, 4))

plotting_map = maps.InjectActiveCells(mesh, ind_active, np.nan)

ax1 = fig.add_axes([0.1, 0.12, 0.7, 0.78])
mesh.plotSlice(
    plotting_map * chargeability_model,
    ax=ax1,
    normal="Y",
    ind=int(len(mesh.hy) / 2),
    grid=True,
    clim=(background_value, chargeable_value),
    pcolor_opts={"cmap": mpl.cm.plasma},
)
ax1.set_title("Chargeability Model")
ax1.set_xlabel("x (m)")
ax1.set_ylabel("z (m)")
ax1.set_xlim([-2000, 2000])
ax1.set_ylim([-2000, 0])

ax2 = fig.add_axes([0.83, 0.12, 0.03, 0.78])
norm = mpl.colors.Normalize(
    vmin=background_value, vmax=chargeable_value
)
cbar = mpl.colorbar.ColorbarBase(
    ax2, cmap=mpl.cm.plasma, norm=norm, orientation="vertical", format="%.2f"
)
cbar.set_label("Intrinsic Chargeability [V/V]", rotation=270, labelpad=15, size=12)

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
ip_simulation = ip.simulation.Simulation3DNodal(
    mesh,
    survey=ip_survey,
    etaMap=chargeability_map,
    sigma=conductivity_map * conductivity_model,
    Solver=Solver,
)

# Run forward simulation and predicted IP data. The data are the voltage (V)
dpred_ip = ip_simulation.dpred(chargeability_model)

#######################################################################
# Plot IP3D Pseudosection
# -----------------------
#
# Here we demonstrate how 3D IP data can be represented on a 3D
# pseudosection plot. To use this utility, you must have Python's *plotly*
# package. Here, we represent the data as apparent chargeabilities.
#

if has_plotly:

    fig = plot_3d_pseudosection(
        ip_survey,
        dpred_ip,
        vlim=[0., np.max(dpred_ip)],
        scale='linear',
        units='V/V',
        marker_opts={'colorscale': 'plasma'}
    )

    fig.update_layout(
        title_text='Apparent Chargeability',
        title_x=0.5,
        title_font_size=24,
        width=650,
        height=500,
        scene_camera=dict(center=dict(x=0, y=0, z=-0.4))
    )
    
    plotly.io.show(fig)

else:
    print("INSTALL 'PLOTLY' TO VISUALIZE 3D PSEUDOSECTIONS")

#######################################################################
# Optional: Write predicted IP data
# ---------------------------------
#
# Write data and true model
#

if save_file:

    # Add 1% Gaussian noise based on the DC data (not the IP data)
    np.random.seed(444)
    std = 0.01 * np.abs(dpred_dc) 
    noise = std * np.random.rand(len(dpred_ip))
    dobs = dpred_ip + noise
    
    # Write out data at their original electrode locations (not shifted)
    data_obj = data.Data(ip_survey, dobs=dobs, standard_deviation=std)
    
    fname = dir_path + "ip_data.xyz"
    write_dcip_xyz(
        fname,
        data_obj,
        data_header='APP_CHG',
        uncertainties_header='UNCERT',
        out_dict=out_dict
    )

    fname = dir_path + "true_chargeability.txt"
    np.savetxt(fname, chargeability_map * chargeability_model, fmt="%.4e")
