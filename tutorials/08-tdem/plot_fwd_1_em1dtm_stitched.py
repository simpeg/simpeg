"""
Stitched 1D Forward Simulation
==============================

Here we use the module *SimPEG.electromangetics.time_domain_1d* to predict
the stepoff responses for an independent set of 1D soundings.
In this tutorial, we focus on the following:

    - Defining receivers, waveforms, sources and the survey
    - Defining a 2D or 3D conductivity model and using the sounding locations to construct the set of local 1D models
    - Defining and running the stitched 1D simulation

Our survey geometry consists of a horizontal loop source with a radius of 6 m
located 20 m above the Earth's surface. The receiver is located at the centre
of the loop and measures the vertical component of the response.


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
from SimPEG.electromagnetics.utils.em1d_utils import plot_layer, get_vertical_discretization_time

plt.rcParams.update({'font.size': 16})
save_file = True


#####################################################################
# topography
# -------------
#
# Here we define surface topography as an (N, 3) numpy array. Topography could
# also be loaded from a file.
#

x = np.linspace(50,4950,50)
y = np.zeros_like(x)
z = np.zeros_like(x)
topo = np.c_[x, y, z].astype(float)


#####################################################################
# Create Survey
# -------------
#
#

x = np.linspace(50,4950,50)
n_sounding = len(x)

source_locations = np.c_[x, np.zeros(n_sounding), 20.*np.ones(n_sounding)]
source_current = 1.
source_radius = 5.

receiver_locations = np.c_[x, np.zeros(n_sounding), 20.*np.ones(n_sounding)]
receiver_orientation = "z"  # "x", "y" or "z"

times = np.logspace(-5, -2, 16)

source_list = []

for ii in range(0, n_sounding):
    
    source_location = mkvc(source_locations[ii, :])
    receiver_location = mkvc(receiver_locations[ii, :])
    
    receiver_list = [
        em1d.receivers.PointReceiver(
            receiver_location, times, orientation=receiver_orientation,
            component="dbdt"
        )
    ]

#     Sources
    source_list.append(
        em1d.sources.HorizontalLoopSource(
            receiver_list=receiver_list, location=source_location, a=source_radius,
            I=source_current
        )
    )
    
    # source_list.append(
    #     em1d.sources.MagneticDipoleSource(
    #         receiver_list=receiver_list, location=source_location, orientation="z",
    #         I=source_current
    #     )
    # )

# Survey
survey = em1d.survey.EM1DSurveyTD(source_list)


###############################################
# Defining a Global Mesh
# ----------------------
#


n_layer = 25
thicknesses = get_vertical_discretization_time(
    times, sigma_background=0.1, n_layer=n_layer-1
)

dx = 100.
hx = np.ones(n_sounding) * dx
hz = np.r_[thicknesses, thicknesses[-1]]
mesh2D = TensorMesh([hx, np.flipud(hz)], x0='0N')
mesh_soundings = TensorMesh([hz, hx], x0='00')

n_param = n_layer*n_sounding



###############################################
# Defining a Model
# ----------------------
#

from scipy.spatial import Delaunay
def PolygonInd(mesh, pts):
    hull = Delaunay(pts)
    inds = hull.find_simplex(mesh2D.gridCC)>=0
    return inds


background_conductivity = 0.1
overburden_conductivity = 0.025
slope_conductivity = 0.4

model = np.ones(n_param) * background_conductivity

layer_ind = mesh2D.gridCC[:, -1] > -30.
model[layer_ind] = overburden_conductivity


x0 = np.r_[0., -30.]
x1 = np.r_[dx*n_sounding, -30.]
x2 = np.r_[dx*n_sounding, -130.]
x3 = np.r_[0., -50.]
pts = np.vstack((x0, x1, x2, x3, x0))
poly_inds = PolygonInd(mesh2D, pts)
model[poly_inds] = slope_conductivity

mapping = maps.ExpMap(nP=n_param)

fig = plt.figure(figsize=(9, 3))
ax1 = fig.add_axes([0.1, 0.12, 0.73, 0.78])
log_mod = np.log10(model)
# log_mod = np.log10(temp_model)

mesh2D.plotImage(
    log_mod, ax=ax1, grid=True,
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


###############################################
# Reorganize to a set of Sounding Models
# --------------------------------------
#




# MODEL TO SOUNDING MODELS METHOD 2
sounding_models = model.reshape(mesh_soundings.vnC, order='C')
sounding_models = np.flipud(sounding_models)
sounding_models = mkvc(sounding_models)

fig = plt.figure(figsize=(4, 7.5))
ax1 = fig.add_axes([0.15, 0.12, 0.67, 0.78])
log_mod_sounding = np.log10(sounding_models)
sounding_models = np.log(sounding_models)

mesh_soundings.plotImage(
    log_mod_sounding, ax=ax1, grid=True,
    clim=(np.log10(overburden_conductivity), np.log10(slope_conductivity)),
    pcolorOpts={"cmap": "viridis"},
)
ax1.set_ylim(mesh_soundings.vectorNy.min(), mesh_soundings.vectorNy.max())

ax1.set_xticks([])
ax1.set_yticks([])

ax1.set_title("Sounding Models")
ax1.set_xlabel("Layer")
ax1.set_ylabel("Sounding Number")

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
    survey=survey, thicknesses=thicknesses, sigmaMap=mapping,
    topo=topo, parallel=False, n_cpu=2, Solver=PardisoSolver
)

#simulation.model = sounding_models
#
#ARGS = simulation.input_args(0)
#print("Number of arguments")
#print(len(ARGS))
#print("Print arguments")
#for ii in range(0, len(ARGS)):
#    print(ARGS[ii])

dpred = simulation.dpred(sounding_models)


#######################################################################
# Plotting Results
# -------------------------------------------------
#
#


d = np.reshape(dpred, (n_sounding, len(times)))

fig, ax = plt.subplots(1,1, figsize = (7, 7))

for ii in range(0, len(times)):
    ax.semilogy(x, np.abs(d[:, ii]), 'k-', lw=3)
    
ax.set_xlabel("Times (s)")
ax.set_ylabel("|dBdt| (T/s)")






if save_file == True:

    dir_path = os.path.dirname(em1d.__file__).split(os.path.sep)[:-3]
    dir_path.extend(["tutorials", "08-tdem", "em1dtm_stitched"])
    dir_path = os.path.sep.join(dir_path) + os.path.sep

    noise = 0.1*np.abs(dpred)*np.random.rand(len(dpred))
    dpred += noise
    fname = dir_path + 'em1dtm_stitched_data.obs'
    
    loc = np.repeat(source_locations, len(times), axis=0)
    fvec = np.kron(np.ones(n_sounding), times)
    
    np.savetxt(
        fname,
        np.c_[loc, fvec, dpred],
        fmt='%.4e', header='X Y Z TIME DBDT_Z'
    )























