"""
Forward Simulation of Stitched SkyTEM Data
==========================================





"""

#####################################################
# Import Modules
# --------------
#

import numpy as np
import os
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from discretize import TensorMesh
from pymatsolver import PardisoSolver

from SimPEG import maps
from SimPEG.utils import mkvc
import SimPEG.electromagnetics.time_domain_1d as em1d
from SimPEG.electromagnetics.utils.em1d_utils import (
    plot_layer,
    get_vertical_discretization_time,
)
from SimPEG.electromagnetics.time_domain_1d.known_waveforms import (
    skytem_HM_2015,
    skytem_LM_2015,
)

plt.rcParams.update({"font.size": 16})
save_file = True


#####################################################################
# topography
# -------------
#
#

x = np.linspace(50, 4950, 50)
# x = np.linspace(50,250,3)
y = np.zeros_like(x)
z = np.zeros_like(x)
topo = np.c_[x, y, z].astype(float)


#####################################################################
# Create Survey
# -------------
#
#

wave_HM = skytem_HM_2015()
wave_LM = skytem_LM_2015()
time_HM = wave_HM.time_gate_center[0::2]
time_LM = wave_LM.time_gate_center[0::2]

time_input_currents_HM = wave_HM.current_times[-7:]
input_currents_HM = wave_HM.currents[-7:]
time_input_currents_LM = wave_LM.current_times[-13:]
input_currents_LM = wave_LM.currents[-13:]


x = np.linspace(50, 4950, 50)
# x = np.linspace(50,250,3)
n_sounding = len(x)

source_locations = np.c_[x, np.zeros(n_sounding), 30.0 * np.ones(n_sounding)]
source_current = 1.0
source_orientation = "z"
receiver_offset_r = 13.25
receiver_offset_z = 2.0

receiver_locations = np.c_[
    x + receiver_offset_r,
    np.zeros(n_sounding),
    30.0 * np.ones(n_sounding) + receiver_offset_z,
]
receiver_orientation = "z"  # "x", "y" or "z"

source_list = []

for ii in range(0, n_sounding):

    source_location = mkvc(source_locations[ii, :])
    receiver_location = mkvc(receiver_locations[ii, :])

    receiver_list = [
        em1d.receivers.PointReceiver(
            receiver_location,
            times=time_HM,
            times_dual_moment=time_LM,
            orientation=receiver_orientation,
            component="dbdt",
        )
    ]

    #     Sources
    source_list.append(
        em1d.sources.MagneticDipoleSource(
            receiver_list=receiver_list,
            location=source_location,
            moment_amplitude=source_current,
            orientation=source_orientation,
            wave_type="general",
            moment_type="dual",
            time_input_currents=time_input_currents_HM,
            input_currents=input_currents_HM,
            n_pulse=1,
            base_frequency=25.0,
            time_input_currents_dual_moment=time_input_currents_LM,
            input_currents_dual_moment=input_currents_LM,
            base_frequency_dual_moment=210,
        )
    )

# Survey
survey = em1d.survey.EM1DSurveyTD(source_list)


###############################################
# Defining a Global Mesh
# ----------------------
#


n_layer = 25
thicknesses = get_vertical_discretization_time(
    np.r_[time_HM, time_LM], sigma_background=0.1, n_layer=n_layer - 1
)

dx = 100.0
hx = np.ones(n_sounding) * dx
hz = np.r_[thicknesses, thicknesses[-1]]
mesh2D = TensorMesh([hx, np.flipud(hz)], x0="0N")
mesh_soundings = TensorMesh([hz, hx], x0="00")

n_param = n_layer * n_sounding


###############################################
# Defining a Model
# ----------------------
#

from scipy.spatial import Delaunay


def PolygonInd(mesh, pts):
    hull = Delaunay(pts)
    inds = hull.find_simplex(mesh2D.gridCC) >= 0
    return inds


background_conductivity = 0.1
overburden_conductivity = 0.025
slope_conductivity = 0.4

model = np.ones(n_param) * background_conductivity

layer_ind = mesh2D.gridCC[:, -1] > -30.0
model[layer_ind] = overburden_conductivity


x0 = np.r_[0.0, -30.0]
x1 = np.r_[dx * n_sounding, -30.0]
x2 = np.r_[dx * n_sounding, -130.0]
x3 = np.r_[0.0, -50.0]
pts = np.vstack((x0, x1, x2, x3, x0))
poly_inds = PolygonInd(mesh2D, pts)
model[poly_inds] = slope_conductivity

mapping = maps.ExpMap(nP=n_param)

# MODEL TO SOUNDING MODELS METHOD 1
# sounding_models = model.reshape(mesh2D.vnC, order='F')
# sounding_models = np.fliplr(sounding_models)
# sounding_models = mkvc(sounding_models.T)

# MODEL TO SOUNDING MODELS METHOD 2
sounding_models = model.reshape(mesh_soundings.vnC, order="C")
sounding_models = np.flipud(sounding_models)
sounding_models = mkvc(sounding_models)

# FROM SOUNDING MODEL TO REGULAR
# temp_model = sounding_models.reshape(mesh2D.vnC, order='C')
# temp_model = np.fliplr(temp_model)
# temp_model = mkvc(temp_model)

chi = np.zeros_like(sounding_models)


fig = plt.figure(figsize=(9, 3))
ax1 = fig.add_axes([0.1, 0.12, 0.73, 0.78])
log_mod = np.log10(model)
# log_mod = np.log10(temp_model)

mesh2D.plotImage(
    log_mod,
    ax=ax1,
    grid=True,
    clim=(np.log10(overburden_conductivity), np.log10(slope_conductivity)),
    pcolorOpts={"cmap": "viridis"},
)
ax1.set_ylim(mesh2D.vectorNy.min(), mesh2D.vectorNy.max())

ax1.set_title("Conductivity Model")
ax1.set_xlabel("x (m)")
ax1.set_ylabel("z (m)")

ax2 = fig.add_axes([0.85, 0.12, 0.05, 0.78])
norm = mpl.colors.Normalize(
    vmin=np.log10(overburden_conductivity), vmax=np.log10(slope_conductivity)
)
cbar = mpl.colorbar.ColorbarBase(
    ax2, norm=norm, cmap=mpl.cm.viridis, orientation="vertical", format="$10^{%.1f}$"
)
cbar.set_label("Conductivity [S/m]", rotation=270, labelpad=15, size=12)


fig = plt.figure(figsize=(4, 8))
ax1 = fig.add_axes([0.1, 0.12, 0.73, 0.78])
log_mod_sounding = np.log10(sounding_models)
sounding_models = np.log(sounding_models)

mesh_soundings.plotImage(
    log_mod_sounding,
    ax=ax1,
    grid=True,
    clim=(np.log10(overburden_conductivity), np.log10(slope_conductivity)),
    pcolorOpts={"cmap": "viridis"},
)
ax1.set_ylim(mesh_soundings.vectorNy.min(), mesh_soundings.vectorNy.max())

ax1.set_title("Ordered Sounding Models")
ax1.set_xlabel("hz (m)")
ax1.set_ylabel("Profile Distance (m)")

ax2 = fig.add_axes([0.85, 0.12, 0.05, 0.78])
norm = mpl.colors.Normalize(
    vmin=np.log10(overburden_conductivity), vmax=np.log10(slope_conductivity)
)
cbar = mpl.colorbar.ColorbarBase(
    ax2, norm=norm, cmap=mpl.cm.viridis, orientation="vertical", format="$10^{%.1f}$"
)
cbar.set_label("Conductivity [S/m]", rotation=270, labelpad=15, size=12)


#######################################################################
# Define the Forward Simulation and Predic Data
# ----------------------------------------------
#


# Simulate response for static conductivity
simulation = em1d.simulation.StitchedEM1DTMSimulation(
    survey=survey,
    thicknesses=thicknesses,
    sigmaMap=mapping,
    chi=chi,
    topo=topo,
    parallel=False,
    n_cpu=2,
    verbose=True,
    Solver=PardisoSolver,
)

# simulation.model = sounding_models
#
# ARGS = simulation.input_args(0)
# print("Number of arguments")
# print(len(ARGS))
# print("Print arguments")
# for ii in range(0, len(ARGS)):
#    print(ARGS[ii])

dpred = simulation.dpred(sounding_models)


#######################################################################
# Plotting Results
# -------------------------------------------------
#
#

n_time = np.r_[time_LM, time_HM].size
d = np.reshape(dpred, (n_sounding, n_time))

fig, ax = plt.subplots(1, 1, figsize=(7, 7))

for ii in range(0, n_time):
    ax.semilogy(x, np.abs(d[:, ii]), "-", lw=2)

ax.set_xlabel("Times (s)")
ax.set_ylabel("|dBdt| (T/s)")
plt.show()


if save_file == True:

    dir_path = os.path.dirname(em1d.__file__).split(os.path.sep)[:-3]
    dir_path.extend(["tutorials", "08-tdem", "em1dtm_stitched_skytem"])
    dir_path = os.path.sep.join(dir_path) + os.path.sep

    noise = 0.1 * np.abs(dpred) * np.random.rand(len(dpred))
    dpred += noise
    fname = dir_path + "em1dtm_stitched_skytem_data.obs"

    loc = np.repeat(source_locations, n_time, axis=0)
    fvec = np.kron(np.ones(n_sounding), np.r_[time_HM, time_LM])

    np.savetxt(fname, np.c_[loc, fvec, dpred], fmt="%.4e")
