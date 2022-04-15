# -*- coding: utf-8 -*-
"""
DC Resistivity Forward Simulation in 3D
=======================================

Here we use the module *SimPEG.electromagnetics.static.resistivity* to predict
DC resistivity data on an OcTree mesh. In this tutorial, we focus on the following:

    - How to define the survey
    - How to definine a tree mesh based on the survey geometry
    - How to define the forward simulations
    - How to predict DC data for a synthetic conductivity model
    - How to include surface topography
    - The units of the model and resulting data
    - Plotting DC data in 3D


In this case, we simulate dipole-dipole data for three East-West lines and two
North-South lines.


"""

##############################################################
# Import modules
# --------------
#
#

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from discretize import TreeMesh
from discretize.utils import mkvc, refine_tree_xyz

from SimPEG import maps, data
from SimPEG.utils import model_builder, surface2ind_topo
from SimPEG.utils.io_utils.io_utils_electromagnetics import write_dcip_xyz
from SimPEG.electromagnetics.static import resistivity as dc
from SimPEG.electromagnetics.static.utils.static_utils import (
    generate_dcip_sources_line,
    apparent_resistivity_from_voltage,
)

# To plot DC data in 3D, the user must have the plotly package
try:
    import plotly
    from SimPEG.electromagnetics.static.utils.static_utils import plot_3d_pseudosection

    has_plotly = True
except:
    has_plotly = False
    pass

try:
    from pymatsolver import Pardiso as Solver
except ImportError:
    from SimPEG import SolverLU as Solver

mpl.rcParams.update({"font.size": 16})
write_output = False

# sphinx_gallery_thumbnail_number = 2

#########################################################################
# Defining Topography
# -------------------
#
# Here we define surface topography as an (N, 3) numpy array. Topography could
# also be loaded from a file. In our case, our survey takes place within a circular
# depression.
#

x_topo, y_topo = np.meshgrid(
    np.linspace(-2100, 2100, 141), np.linspace(-2000, 2000, 141)
)
s = np.sqrt(x_topo ** 2 + y_topo ** 2)
z_topo = 10 + (1 / np.pi) * 140 * (-np.pi / 2 + np.arctan((s - 600.0) / 160.0))
x_topo, y_topo, z_topo = mkvc(x_topo), mkvc(y_topo), mkvc(z_topo)
topo_xyz = np.c_[x_topo, y_topo, z_topo]

#########################################################################
# Construct the DC Survey
# -----------------------
#
# Here we define 5 DC lines that use a dipole-dipole electrode configuration;
# three lines along the East-West direction and 2 lines along the North-South direction.
# For each source, we must define the AB electrode locations. For each receiver
# we must define the MN electrode locations. Instead of creating the survey
# from scratch (see 1D example), we will use the *generat_dcip_sources_line* utility.
# This utility will give us the source list for a given DC/IP line. We can append
# the sources for multiple lines to create the survey.
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
num_rx_per_src = 8

# The source lists for each line can be appended to create the source
# list for the whole survey.
source_list = []
for ii in range(0, len(end_locations_list)):
    source_list += generate_dcip_sources_line(
        survey_type,
        data_type,
        dimension_type,
        end_locations_list[ii],
        topo_xyz,
        num_rx_per_src,
        station_separation,
    )

# Define the survey
survey = dc.survey.Survey(source_list)

#################################################################
# Create OcTree Mesh
# ------------------
#
# Here, we create the OcTree mesh that will be used to predict DC data.
#
#

# Defining domain side and minimum cell size
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
k = np.sqrt(np.sum(topo_xyz[:, 0:2] ** 2, axis=1)) < 1200
mesh = refine_tree_xyz(
    mesh, topo_xyz[k, :], octree_levels=[0, 6, 8], method="surface", finalize=False
)

# Mesh refinement near sources and receivers.
electrode_locations = np.r_[
    survey.locations_a, survey.locations_b, survey.locations_m, survey.locations_n
]
unique_locations = np.unique(electrode_locations, axis=0)
mesh = refine_tree_xyz(
    mesh, unique_locations, octree_levels=[4, 6, 4], method="radial", finalize=False
)

# Finalize the mesh
mesh.finalize()

################################################################
# Create Conductivity Model and Mapping for OcTree Mesh
# -----------------------------------------------------
#
# Here we define the conductivity model that will be used to predict DC
# resistivity data. The model consists of a conductive sphere and a
# resistive sphere within a moderately conductive background. Note that
# you can carry through this work flow with a resistivity model if desired.
#

# Define conductivity model in S/m (or resistivity model in Ohm m)
air_value = 1e-8
background_value = 1e-2
conductor_value = 1e-1
resistor_value = 1e-3

# Find active cells in forward modeling (cell below surface)
ind_active = surface2ind_topo(mesh, topo_xyz)

# Define mapping from model to active cells
nC = int(ind_active.sum())
conductivity_map = maps.InjectActiveCells(mesh, ind_active, air_value)

# Define model
conductivity_model = background_value * np.ones(nC)

ind_conductor = model_builder.getIndicesSphere(
    np.r_[-350.0, 0.0, -300.0], 160.0, mesh.cell_centers[ind_active, :]
)
conductivity_model[ind_conductor] = conductor_value

ind_resistor = model_builder.getIndicesSphere(
    np.r_[350.0, 0.0, -300.0], 160.0, mesh.cell_centers[ind_active, :]
)
conductivity_model[ind_resistor] = resistor_value

# Plot Conductivity Model
fig = plt.figure(figsize=(10, 4))

plotting_map = maps.InjectActiveCells(mesh, ind_active, np.nan)
log_mod = np.log10(conductivity_model)

ax1 = fig.add_axes([0.15, 0.15, 0.68, 0.75])
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
ax1.set_xlim([-1000, 1000])
ax1.set_ylim([-1000, 0])

ax2 = fig.add_axes([0.84, 0.15, 0.03, 0.75])
norm = mpl.colors.Normalize(
    vmin=np.log10(resistor_value), vmax=np.log10(conductor_value)
)
cbar = mpl.colorbar.ColorbarBase(
    ax2, cmap=mpl.cm.viridis, norm=norm, orientation="vertical", format="$10^{%.1f}$"
)
cbar.set_label("Conductivity [S/m]", rotation=270, labelpad=15, size=12)

##########################################################
# Project Survey to Discretized Topography
# ----------------------------------------
#
# It is important that electrodes are not modeled as being in the air. Even if the
# electrodes are properly located along surface topography, they may lie above
# the *discretized* topography. This step is carried out to ensure all electrodes
# lie on the discretized surface.
#
#

survey.drape_electrodes_on_topography(mesh, ind_active, option="top")

############################################################
# Predict DC Resistivity Data
# ---------------------------
#
# Here we predict DC resistivity data. If the keyword argument *sigmaMap* is
# defined, the simulation will expect a conductivity model. If the keyword
# argument *rhoMap* is defined, the simulation will expect a resistivity model.
#
#
#

# Define the DC simulation
simulation = dc.simulation.Simulation3DNodal(
    mesh, survey=survey, sigmaMap=conductivity_map, solver=Solver
)

# Predict the data by running the simulation. The data are the measured voltage
# normalized by the source current in units of V/A.
dpred = simulation.dpred(conductivity_model)

#########################################################
# Plot DC Data in 3D Pseudosection
# --------------------------------
#
# Here we demonstrate how 3D DC resistivity data can be represented on a 3D
# pseudosection plot. To use this utility, you must have Python's *plotly*
# package. Here, we represent the data as apparent conductivities.
#
# The *plot_3d_pseudosection* utility allows the user to plot all pseudosection
# points, or plot the pseudosection plots that lie within some distance of
# one or more planes.
#

# Since the data are normalized voltage, we must convert predicted
# to apparent conductivities.
apparent_conductivity = 1 / apparent_resistivity_from_voltage(
    survey,
    dpred,
)

# For large datasets or for surveys with unconventional electrode geometry,
# interpretation can be challenging if we plot every datum. Here, we plot
# 3 out of the 5 survey lines to better image anomalous structures.
# To plot ALL of the data, simply remove the keyword argument *plane_points*
# when calling *plot_3d_pseudosection*.
plane_points = []
p1, p2, p3 = np.array([-1000, 0, 0]), np.array([1000, 0, 0]), np.array([0, 0, -1000])
plane_points.append([p1, p2, p3])
p1, p2, p3 = (
    np.array([-350, -1000, 0]),
    np.array([-350, 1000, 0]),
    np.array([-350, 0, -1000]),
)
plane_points.append([p1, p2, p3])
p1, p2, p3 = (
    np.array([350, -1000, 0]),
    np.array([350, 1000, 0]),
    np.array([350, 0, -1000]),
)
plane_points.append([p1, p2, p3])

if has_plotly:

    fig = plot_3d_pseudosection(
        survey,
        apparent_conductivity,
        scale="log",
        units="S/m",
        plane_points=plane_points,
        plane_distance=15,
    )

    fig.update_layout(
        title_text="Apparent Conductivity",
        title_x=0.5,
        title_font_size=24,
        width=650,
        height=500,
        scene_camera=dict(center=dict(x=0.05, y=0, z=-0.4)),
    )

    plotly.io.show(fig)

else:
    print("INSTALL 'PLOTLY' TO VISUALIZE 3D PSEUDOSECTIONS")

########################################################
# Optional: Write Predicted DC Data
# ---------------------------------
#
# Write DC resistivity data, topography and true model
#

if write_output:

    dir_path = os.path.dirname(__file__).split(os.path.sep)
    dir_path.extend(["outputs"])
    dir_path = os.path.sep.join(dir_path) + os.path.sep

    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

    # Add 5% Gaussian noise to each datum
    np.random.seed(433)
    std = 0.1 * np.abs(dpred)
    noise = std * np.random.rand(len(dpred))
    dobs = dpred + noise

    # Create dictionary that stores line IDs
    N = int(survey.nD / len(end_locations_list))
    lineID = np.r_[np.ones(N), 2 * np.ones(N), 3 * np.ones(N)]
    out_dict = {"LINEID": lineID}

    # Create a survey with the original electrode locations
    # and not the shifted ones
    source_list = []
    for ii in range(0, len(end_locations_list)):
        source_list += generate_dcip_sources_line(
            survey_type,
            data_type,
            dimension_type,
            end_locations_list[ii],
            topo_xyz,
            num_rx_per_src,
            station_separation,
        )
    survey_original = dc.survey.Survey(source_list)

    # Write out data at their original electrode locations (not shifted)
    data_obj = data.Data(survey_original, dobs=dobs, standard_deviation=std)

    fname = dir_path + "dc_data.xyz"
    write_dcip_xyz(
        fname,
        data_obj,
        data_header="V/A",
        uncertainties_header="UNCERT",
        out_dict=out_dict,
    )

    fname = dir_path + "topo_xyz.txt"
    np.savetxt(fname, topo_xyz, fmt="%.4e")
