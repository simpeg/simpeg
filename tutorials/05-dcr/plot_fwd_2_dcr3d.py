# -*- coding: utf-8 -*-
"""
DC Resistivity Forward Simulation in 3D
=======================================

Here we use the module *SimPEG.electromagnetics.static.resistivity* to predict
DC resistivity data on an OcTree mesh. In this tutorial, we focus on the following:

    - How to define the survey
    - How to definine a tree mesh based on the survey geometry
    - How to define the forward simulation
    - How to predict DC resistivity data for a synthetic resistivity model
    - How to include surface topography
    - The units of the model and resulting data
    - Plotting DC resistivity data in 3D


In this case, we simulate dipole-dipole data for both an East-West and a
North-South survey line.

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
from SimPEG.electromagnetics.static.utils.static_utils import (
    generate_dcip_sources_line,
    apparent_resistivity,
    plot_3d_pseudosection,
)

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
# also be loaded from a file. In our case, our survey takes place in a valley
# that runs North-South.
#

x_topo, y_topo = np.meshgrid(
    np.linspace(-3000, 3000, 101), np.linspace(-3000, 3000, 101)
)
z_topo = (1 / np.pi) * 85 * (-np.pi / 2 + np.arctan((np.abs(x_topo) - 600.0) / 50.0))
x_topo, y_topo, z_topo = mkvc(x_topo), mkvc(y_topo), mkvc(z_topo)
xyz_topo = np.c_[x_topo, y_topo, z_topo]


#####################################################################
# Create Dipole-Dipole Survey
# ---------------------------
#
# Here we define a single EW survey line that uses a dipole-dipole configuration.
# For the source, we must define the AB electrode locations. For the receivers
# we must define the MN electrode locations. Instead of creating the survey
# from scratch (see 1D example), we will use the *generat_dcip_survey_line* utility.
#

# West to East survey line
survey_type = "dipole-dipole"
data_type = "volt"
dimension_type = "3D"
end_locations = np.r_[-1000.0, 1000.0, 0., 0.]
station_separation = 100.0
dipole_separation = 100.0
num_rx_per_src = 10

source_list = generate_dcip_sources_line(
    survey_type,
    data_type,
    dimension_type,
    end_locations,
    xyz_topo,
    num_rx_per_src,
    station_separation
)

# South to North survey line
end_locations = np.r_[0., 0., -1000.0, 1000.0]

source_list = source_list + generate_dcip_sources_line(
    survey_type,
    data_type,
    dimension_type,
    end_locations,
    xyz_topo,
    num_rx_per_src,
    station_separation
)

survey = dc.survey.Survey(source_list)


###############################################################
# Create OcTree Mesh
# ------------------
#
# Here, we create the OcTree mesh that will be used to predict both DC
# resistivity and IP data.
#

dh = 40.0  # base cell width
dom_width_x = 10000.0  # domain width x
dom_width_y = 10000.0  # domain width y
dom_width_z = 2000.0  # domain width z
nbcx = 2 ** int(np.round(np.log(dom_width_x / dh) / np.log(2.0)))  # num. base cells x
nbcy = 2 ** int(np.round(np.log(dom_width_y / dh) / np.log(2.0)))  # num. base cells y
nbcz = 2 ** int(np.round(np.log(dom_width_z / dh) / np.log(2.0)))  # num. base cells z

# Define the base mesh
hx = [(dh, nbcx)]
hy = [(dh, nbcy)]
hz = [(dh, nbcz)]
mesh = TreeMesh([hx, hy, hz], x0="CCN")

# Mesh refinement based on topography
mesh = refine_tree_xyz(
    mesh, xyz_topo, octree_levels=[0, 0, 1], method="surface", finalize=False
)

# Mesh refinement near transmitters and receivers. First we need to obtain the
# set of unique electrode locations.
electrode_locations = np.r_[
    survey.locations_a, survey.locations_b, survey.locations_m, survey.locations_n
]

unique_locations = np.unique(electrode_locations, axis=0)

mesh = refine_tree_xyz(
    mesh, unique_locations, octree_levels=[2, 4], method="radial", finalize=False
)

# Refine core mesh region
xp, yp, zp = np.meshgrid([-1200.0, 1200.0], [-1200.0, 1200.0], [-800.0, 0.0])
xyz = np.c_[mkvc(xp), mkvc(yp), mkvc(zp)]
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
block_conductivity = 1e-1

# Find active cells in forward modeling (cell below surface)
ind_active = surface2ind_topo(mesh, xyz_topo)

# Define mapping from model to active cells
nC = int(ind_active.sum())
conductivity_map = maps.InjectActiveCells(mesh, ind_active, air_conductivity)

# Define model
conductivity_model = background_conductivity * np.ones(nC)

ind_block = (
    (mesh.gridCC[ind_active, 0] > -200.0)
    & (mesh.gridCC[ind_active, 0] < 200.0)
    & (mesh.gridCC[ind_active, 1] > -400.0)
    & (mesh.gridCC[ind_active, 1] < 400.0)
    & (mesh.gridCC[ind_active, 2] > -640.0)
    & (mesh.gridCC[ind_active, 2] < -240.0)
)

conductivity_model[ind_block] = block_conductivity


# Plot Conductivity Model
fig = plt.figure(figsize=(11, 4))

plotting_map = maps.InjectActiveCells(mesh, ind_active, np.nan)
log_mod = np.log10(conductivity_model)

ax1 = fig.add_axes([0.1, 0.12, 0.7, 0.78])
mesh.plotSlice(
    plotting_map * log_mod,
    ax=ax1,
    normal='Y',
    ind=int(len(mesh.hy)/2),
    grid=True,
    clim=(np.log10(background_conductivity), np.log10(block_conductivity)),
    pcolorOpts={"cmap": "viridis"},
)
ax1.set_title("Conductivity Model")
ax1.set_xlabel("x (m)")
ax1.set_ylabel("z (m)")
ax1.set_xlim([-2000, 2000])
ax1.set_ylim([-2000, 0])

ax2 = fig.add_axes([0.85, 0.12, 0.05, 0.78])
norm = mpl.colors.Normalize(
    vmin=np.log10(background_conductivity), vmax=np.log10(block_conductivity)
)
cbar = mpl.colorbar.ColorbarBase(
    ax2, norm=norm, cmap=mpl.cm.viridis, orientation="vertical", format="$10^{%.1f}$"
)
cbar.set_label("Conductivity [S/m]", rotation=270, labelpad=15, size=12)


###############################################################
# Project Survey to Discretized Topography
# ----------------------------------------
#
# It is important that electrodes are not model as being in the air. Even if the
# electrodes are properly located along surface topography, they may lie above
# the discretized topography. This step is carried out to ensure all electrodes
# like on the discretized surface.
#

survey.drape_electrodes_on_topography(mesh, ind_active, option="top")


#######################################################################
# Predict DC Resistivity Data
# ---------------------------
#
# Here we predict DC resistivity data. If the keyword argument *sigmaMap* is
# defined, the simulation will expect a conductivity model. If the keyword
# argument *rhoMap* is defined, the simulation will expect a resistivity model.
#

simulation = dc.simulation.Simulation3DNodalFictitiousSources(
    mesh, survey=survey, sigmaMap=conductivity_map, Solver=Solver
)

# Predict the data by running the simulation. The data are the raw voltage in
# units of volts.
dpred = simulation.dpred(conductivity_model)

#######################################################################
# Plot 3D Pseudosection
# ---------------------
#

dc_data = data.Data(survey, dobs=dpred)

# Define plane
plane_points = []

p1, p2, p3 = np.array([-1000, 0, 0]), np.array([1000, 0, 0]), np.array([1000, 0, -1000])
plane_points.append([p1,p2,p3])

p1, p2, p3 = np.array([0, -1000, 0]), np.array([0, 1000,0]), np.array([0, 1000, -1000])
plane_points.append([p1,p2,p3])

fig = plt.figure(figsize=(12, 4))
ax1 = fig.add_axes([0.05, 0.1, 0.8, 0.8], projection='3d', azim=-60, elev=45)
cax1 = fig.add_axes([0.85, 0.15, 0.02, 0.7])

app_res = apparent_resistivity(
    dc_data, space_type="half space", dobs=dpred, eps=1e-10,
)

vlim = [app_res.min(), app_res.max()]

plot_3d_pseudosection(
    survey, app_res, s=150, ax=ax1, scale='log', vlim=vlim, cax=cax1,
    plane_points=plane_points, plane_distance=40., units='$\Omega m$'
)

# ax1.set_xlim([-1000., 1000.])
# ax1.set_ylim([-1000., 1000.])

#######################################################################
# Optional: Write out dpred
# -------------------------
#
# Write DC resistivity data, topography and true model
#

if save_file:

    dir_path = os.path.dirname(dc.__file__).split(os.path.sep)[:-4]
    dir_path.extend(["tutorials", "assets", "dcr2d"])
    dir_path = os.path.sep.join(dir_path) + os.path.sep

    # Add 5% Gaussian noise to each datum
    noise = 0.05 * dpred * np.random.rand(len(dpred))

    # Write out data at their original electrode locations (not shifted)
    data_array = np.c_[electrode_locations, dpred + noise]

    fname = dir_path + "dc_data.obs"
    np.savetxt(fname, data_array, fmt="%.4e")

    fname = dir_path + "true_conductivity.txt"
    np.savetxt(fname, conductivity_map * conductivity_model, fmt="%.4e")

    fname = dir_path + "xyz_topo.txt"
    np.savetxt(fname, xyz_topo, fmt="%.4e")
