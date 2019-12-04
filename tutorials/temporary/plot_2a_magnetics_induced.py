
"""
Magnetic Simulation on a Tensor Mesh
====================================

Here we use the module *SimPEG.potential_fields.magnetics* to predict magnetic
data for a magnetic susceptibility model. We simulate the data on a tensor mesh.
For this tutorial, we focus on the following:

    - How to define the survey
    - How to predict magnetic data for a susceptibility model
    - How to include surface topography
    - The units of the physical property model and resulting data
    

"""

#########################################################################
# Import Modules
# --------------
#

import numpy as np
from scipy.interpolate import LinearNDInterpolator
import matplotlib as mpl
import matplotlib.pyplot as plt

from discretize import TensorMesh
from discretize.utils import mkvc
from SimPEG.utils import plot2Ddata, ModelBuilder
from SimPEG import maps
from SimPEG.potential_fields import magnetics

# sphinx_gallery_thumbnail_number = 4


#############################################
# Topography
# ----------
#
# Here we define surface topography as an (N, 3) numpy array. Topography could
# also be loaded from a file.
#

[x_topo, y_topo] = np.meshgrid(
        np.linspace(-200, 200, 41), np.linspace(-200, 200, 41)
        )
z_topo = -15*np.exp(-(x_topo**2 + y_topo**2) / 80**2)
x_topo, y_topo, z_topo = mkvc(x_topo), mkvc(y_topo), mkvc(z_topo)
xyz_topo = np.c_[x_topo, y_topo, z_topo]

#############################################
# Defining the Survey
# -------------------
#
# Here, we define the survey that will be used for both forward simulations.
# Magnetic surveys are simple to create. The user needs an (N, 3) array to define
# the xyz positions of the observation locations. The user also needs to
# define the Earth's magnetic field intensity and orientation. Here, we
# create a basic airborne survey with a flight height of 10 m above the
# surface topography.
#

# Define the observation locations as an (N, 3) numpy array or load them.
x = np.linspace(-80., 80., 17)
y = np.linspace(-80., 80., 17)
x, y = np.meshgrid(x, y)
x, y = mkvc(x.T), mkvc(y.T)
fun_interp = LinearNDInterpolator(np.c_[x_topo, y_topo], z_topo)
z = fun_interp(np.c_[x, y]) + 10  # Flight height 10 m above surface.
receiver_locations = np.c_[x, y, z]

# Define the receivers. Here the user may define the receiver to measure
# total magnetic intensity, Cartesian components of the anomalous field or
# gradient components of the magnetic field (for magnetic gradiometry)
receiver_list = magnetics.receivers.point_receiver(receiver_locations, "tmi")

# Define the inducing field H0 = (intensity [nT], inclination [deg], declination [deg])
inclination = 90
declination = 0
strength = 50000

inducing_field = (strength, inclination, declination)
source_field = magnetics.sources.SourceField(
    receiver_list=[receiver_list], parameters=inducing_field
    )

# Define the survey
survey = magnetics.survey.MagneticSurvey(source_field)


#############################################
# Defining a Tensor Mesh
# ----------------------
#
# Here, we create the tensor mesh that will be used to predict magnetic
# for our first forward simulation.
#

dh = 5.
hx = [(dh, 5, -1.3), (dh, 40), (dh, 5, 1.3)]
hy = [(dh, 5, -1.3), (dh, 40), (dh, 5, 1.3)]
hz = [(dh, 5, -1.3), (dh, 15)]
mesh = TensorMesh([hx, hy, hz], 'CCN')


#############################################
# Defining a Susceptibility Model
# -------------------------------
#
# Here, we create the model that will be used to predict magnetic data
# and the mapping from the model to the mesh. The model
# consists of a susceptible sphere in a less susceptible host.
#

# Define susceptibility values for each unit in SI
background_value = 0.0001
sphere_value = 0.01

# Find cells that are active in the forward modeling (cells below surface)
ind_active = surface2ind_topo(mesh, xyz_topo)

# Define mapping from model to active cells
nC = int(ind_active.sum())
model_map = maps.IdentityMap(nP=nC)  # model is a vlue for each active cell

# Define model
model = background_value*np.ones(ind_active.sum())
ind_sphere = ModelBuilder.getIndicesSphere(
    np.r_[0., 0., -45.], 15., mesh.gridCC
)
ind_sphere = ind_sphere[ind_active]
model[ind_sphere] = sphere_value

# Plot Model
fig = plt.figure(figsize=(9, 4))

plotting_map = maps.InjectActiveCells(mesh, ind_active, np.nan)
ax1 = fig.add_axes([0.05, 0.05, 0.78, 0.9])
mesh.plotSlice(
    plotting_map*model, normal='Y', ax=ax1, ind=int(mesh.nCy/2), grid=True,
    clim=(np.min(model), np.max(model))
)
ax1.set_title('Model slice at y = 0 m')

ax2 = fig.add_axes([0.85, 0.05, 0.05, 0.9])
norm = mpl.colors.Normalize(vmin=np.min(model), vmax=np.max(model))
cbar = mpl.colorbar.ColorbarBase(
    ax2, norm=norm, orientation='vertical'
)
cbar.set_label(
    'Magnetic Susceptibility (SI)',
    rotation=270, labelpad=15, size=12
)

plt.show()


###################################################################
# Simulation: TMI Data for a Susceptibility Model
# -----------------------------------------------
#
# Here we demonstrate how to predict magnetic data for a magnetic
# susceptibility model.
#

# Define the forward simluation
simulation = magnetics.simulation.MagneticIntegralSimulation(
    survey=survey, mesh=mesh,
    modelType='susceptibility', chiMap=model_map,
    actInd=ind_active, forward_only=True
)

# Compute predicted data for a susceptibility model
dpred = simulation.dpred(model)

# Plot
fig = plt.figure(figsize=(6, 5))
v_max = np.max(np.abs(dpred))

ax1 = fig.add_axes([0.05, 0.05, 0.8, 0.9])
plot2Ddata(
    receiver_list.locs, dpred, ax=ax1, ncontour=30, clim=(-v_max, v_max),
    contourOpts={"cmap": "RdBu_r"}
)
ax1.set_title('TMI Anomaly')

ax2 = fig.add_axes([0.85, 0.05, 0.05, 0.9])
norm = mpl.colors.Normalize(
        vmin=-np.max(np.abs(dpred)), vmax=np.max(np.abs(dpred))
)
cbar = mpl.colorbar.ColorbarBase(
        ax2, norm=norm, orientation='vertical', cmap=mpl.cm.RdBu_r
)
cbar.set_label('$nT$', rotation=270, labelpad=15, size=12)

plt.show()
