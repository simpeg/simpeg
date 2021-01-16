"""
Stitched Forward Simulation for a Set of 1D Soundings
=====================================================





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
import SimPEG.electromagnetics.frequency_domain_1d as em1d
from SimPEG.electromagnetics.utils.em1d_utils import (
    plot_layer, get_vertical_discretization_frequency
)

plt.rcParams.update({'font.size': 16})
save_file = True

# sphinx_gallery_thumbnail_number = 3

#####################################################################
# topography
# -------------
#
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

source_locations = np.c_[x, np.zeros(n_sounding), 30 *np.ones(n_sounding)]
source_current = 1.
source_radius = 5.
moment_amplitude=1.

receiver_locations = np.c_[x+10., np.zeros(n_sounding), 30*np.ones(n_sounding)]
receiver_orientation = "z"  # "x", "y" or "z"
field_type = "ppm"  # "secondary", "total" or "ppm"

frequencies = np.array([25., 100., 382, 1822, 7970, 35920], dtype=float)

source_list = []

for ii in range(0, n_sounding):

    source_location = mkvc(source_locations[ii, :])
    receiver_location = mkvc(receiver_locations[ii, :])

    receiver_list = []

    receiver_list.append(
        em1d.receivers.PointReceiver(
            receiver_location, frequencies, orientation=receiver_orientation,
            field_type=field_type, component="real"
        )
    )
    receiver_list.append(
        em1d.receivers.PointReceiver(
            receiver_location, frequencies, orientation=receiver_orientation,
            field_type=field_type, component="imag"
        )
    )

    source_list.append(
        em1d.sources.MagneticDipoleSource(
            receiver_list=receiver_list, location=source_location, orientation="z",
            moment_amplitude=moment_amplitude
        )
    )

# Survey
survey = em1d.survey.EM1DSurveyFD(source_list)


###############################################
# Defining a Global Mesh
# ----------------------
#

n_layer = 30
thicknesses = get_vertical_discretization_frequency(
    frequencies, sigma_background=0.1, n_layer=n_layer-1
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
    inds = hull.find_simplex(mesh.gridCC)>=0
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
ax1 = fig.add_axes([0.15, 0.12, 0.65, 0.78])
log_mod = np.log10(model)

mesh2D.plotImage(
    log_mod, ax=ax1, grid=True,
    clim=(np.log10(overburden_conductivity), np.log10(slope_conductivity)),
    pcolorOpts={"cmap": "viridis"},
)
ax1.set_ylim(mesh2D.vectorNy.min(), mesh2D.vectorNy.max())

ax1.set_title("Conductivity Model")
ax1.set_xlabel("x (m)")
ax1.set_ylabel("z (m)")

ax2 = fig.add_axes([0.82, 0.12, 0.03, 0.78])
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
# Define the Forward Simulation and Predict Data
# ----------------------------------------------
#
# source, then receiver, then frequency


# Simulate response for static conductivity
simulation = em1d.simulation.StitchedEM1DFMSimulation(
    survey=survey, thicknesses=thicknesses, sigmaMap=mapping,
    topo=topo, parallel=False, Solver=PardisoSolver
)

dpred = simulation.dpred(sounding_models)


#######################################################################
# Plotting Results
# -------------------------------------------------
#
#

N = n_sounding

d_plotting = np.reshape(dpred, (2*n_sounding, len(frequencies))).T

d_real = d_plotting[:, 0::2]
d_imag = d_plotting[:, 1::2]

fig, ax = plt.subplots(1,1, figsize = (7, 7))

for ii in range(0, n_sounding):
    ax.loglog(frequencies, np.abs(d_real[:, ii]), 'b-', lw=2)
    ax.loglog(frequencies, np.abs(d_imag[0:len(frequencies):, ii]), 'r--', lw=2)

ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("|Hs/Hp| (ppm)")
ax.set_title("Secondary Magnetic Field")
ax.legend(["Real", "Imaginary"])




if save_file == True:

    dir_path = os.path.dirname(em1d.__file__).split(os.path.sep)[:-3]
    dir_path.extend(["tutorials", "07-fdem", "em1dfm_stitched"])
    dir_path = os.path.sep.join(dir_path) + os.path.sep

    noise = 0.1*np.abs(dpred)*np.random.rand(len(dpred))
    dpred += noise
    fname = dir_path + "em1dfm_stitched_data.obs"

    loc = np.repeat(source_locations, len(frequencies), axis=0)
    fvec = np.kron(np.ones(n_sounding), frequencies)
    dout = np.c_[mkvc(d_real), mkvc(d_imag)]

    np.savetxt(
        fname,
        np.c_[loc, fvec, dout],
        fmt='%.4e', header='X Y Z FREQUENCY HZ_REAL HZ_IMAG'
    )























