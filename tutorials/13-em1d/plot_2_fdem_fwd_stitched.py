"""
Forward Simulation of Stitched Frequency-Domain Data
====================================================





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
import simpegEM1D as em1d
from simpegEM1D.utils import plotLayer, get_vertical_discretization_frequency

plt.rcParams.update({'font.size': 16})
save_file = False


#####################################################################
# topography
# -------------
#
#
x = np.linspace(50,4950,50)
#x = np.linspace(50,250,3)
y = np.zeros_like(x)
z = np.zeros_like(x)
topo = np.c_[x, y, z].astype(float)





#####################################################################
# Create Survey
# -------------
#
#
x = np.linspace(50,4950,50)
#x = np.linspace(50,250,3)
n_sounding = len(x)

source_locations = np.c_[x, np.zeros(n_sounding), 30 *np.ones(n_sounding)]
source_current = 1.
source_radius = 5.
moment_amplitude=1.

receiver_locations = np.c_[x+10., np.zeros(n_sounding), 30 *np.ones(n_sounding)]
receiver_orientation = "z"  # "x", "y" or "z"
field_type = "ppm"  # "secondary", "total" or "ppm"

frequencies = np.array([25., 100., 382, 1822, 7970, 35920], dtype=float)

source_list = []

for ii in range(0, n_sounding):

    source_location = mkvc(source_locations[ii, :])
    receiver_location = mkvc(receiver_locations[ii, :])

    receiver_list = []

    receiver_list.append(
        em1d.receivers.HarmonicPointReceiver(
            receiver_location, frequencies, orientation=receiver_orientation,
            field_type=field_type, component="both"
        )
    )
    # receiver_list.append(
    #     em1d.receivers.HarmonicPointReceiver(
    #         receiver_location, frequencies, orientation=receiver_orientation,
    #         field_type=field_type, component="imag"
    #     )
    # )

#     Sources
#    source_list = [
#        em1d.sources.HarmonicHorizontalLoopSource(
#            receiver_list=receiver_list, location=source_location, a=source_radius,
#            I=source_current
#        )
#    ]

    source_list.append(
        em1d.sources.HarmonicMagneticDipoleSource(
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

# MODEL TO SOUNDING MODELS METHOD 1
# sounding_models = model.reshape(mesh2D.vnC, order='F')
# sounding_models = np.fliplr(sounding_models)
# sounding_models = mkvc(sounding_models.T)

# MODEL TO SOUNDING MODELS METHOD 2
sounding_models = model.reshape(mesh_soundings.vnC, order='C')
sounding_models = np.flipud(sounding_models)
sounding_models = mkvc(sounding_models)

chi = np.zeros_like(sounding_models)






fig = plt.figure(figsize=(9, 3))
ax1 = fig.add_axes([0.1, 0.12, 0.73, 0.78])
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
    log_mod_sounding, ax=ax1, grid=True,
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
simulation = em1d.simulation.StitchedEM1DFMSimulation(
    survey=survey, thicknesses=thicknesses, sigmaMap=mapping, chi=chi,
    topo=topo, parallel=False, verbose=True, Solver=PardisoSolver
)

# simulation = em1d.simulation.StitchedEM1DFMSimulation(
#     survey=survey, thicknesses=thicknesses, sigmaMap=mapping, chi=chi,
#     topo=topo, parallel=True, n_cpu=2, verbose=True, Solver=PardisoSolver
# )


dpred = simulation.dpred(sounding_models)


#######################################################################
# Plotting Results
# -------------------------------------------------
#
#


d = np.reshape(dpred, (n_sounding, 2*len(frequencies))).T

fig, ax = plt.subplots(1,1, figsize = (7, 7))

for ii in range(0, n_sounding):
    ax.loglog(frequencies, np.abs(d[0:len(frequencies), ii]), '-', lw=2)
    ax.loglog(frequencies, np.abs(d[len(frequencies):, ii]), '--', lw=2)

ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("|Hs/Hp| (ppm)")
ax.set_title("Magnetic Field as a Function of Frequency")
ax.legend(["real", "imaginary"])

#
#d = np.reshape(dpred, (n_sounding, 2*len(frequencies)))
#fig = plt.figure(figsize = (10, 5))
#ax1 = fig.add_subplot(121)
#ax2 = fig.add_subplot(122)
#
#for ii in range(0, n_sounding):
#    ax1.semilogy(x, np.abs(d[:, 0:len(frequencies)]), 'k-', lw=2)
#    ax2.semilogy(x, np.abs(d[:, len(frequencies):]), 'k--', lw=2)



if save_file == True:

    noise = 0.1*np.abs(dpred)*np.random.rand(len(dpred))
    dpred += noise
    fname = os.path.dirname(em1d.__file__) + '\\..\\tutorials\\assets\\em1dfm_stitched_data.obs'

    loc = np.repeat(source_locations, len(frequencies), axis=0)
    fvec = np.kron(np.ones(n_sounding), frequencies)
    dout = np.c_[dpred[0::2], dpred[1::2]]

    np.savetxt(
        fname,
        np.c_[loc, fvec, dout],
        fmt='%.4e'
    )























