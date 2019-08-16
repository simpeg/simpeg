
"""
Gravity
=======

Here we use the module *SimPEG.PF.Gravity* to predict gravity anomaly data for
synthetic density contrast models. For this tutorial, we focus on the following:

    - How to define the survey
    - How to define the problem
    - How to predict gravity anomaly data and gravity gradiometry data for a synthetic model
    - How to include surface topography
    - The units of the density model and resulting data

This tutorial contains two examples. In the first, we forward model gravity
anomaly data using a tensor mesh. For the second example, we forward model
gravity gradiometry data on an OcTree mesh.
    

"""

#########################################################################
# Import modules
# --------------
#

import numpy as np
from scipy.interpolate import LinearNDInterpolator
import matplotlib as mpl
import matplotlib.pyplot as plt

from discretize import TensorMesh, TreeMesh
from discretize.utils import mkvc, refine_tree_xyz
from SimPEG.utils import plot2Ddata, ModelBuilder, surface2ind_topo
from SimPEG import maps
from SimPEG import PF
import os

# sphinx_gallery_thumbnail_number = 2

#############################################
# Defining Topography
# -------------------
#
# Here we define surface topography as an (N, 3) numpy array. Topography could
# also be loaded from a file. This is used for both examples.
#

[xx, yy] = np.meshgrid(np.linspace(-200, 200, 41), np.linspace(-200, 200, 41))
zz = -15*np.exp(-(xx**2 + yy**2) / 80**2)
xx, yy, zz = mkvc(xx), mkvc(yy), mkvc(zz)
topo = np.c_[xx, yy, zz]

fname = os.path.dirname(PF.__file__) + '\\..\\..\\tutorials\\assets\\gravity_topo.txt'
np.savetxt(
    fname,
    np.c_[topo],
    fmt='%.4e'
)

#############################################
# Defining the Survey
# -------------------
#
# Here, we define survey that will be used for all tutorial examples. Gravity
# surveys are simple to create. The user only needs an (N, 3) array to define
# the xyz locations of the observation locations. From this, the user can
# define the receivers and the source field.
#

# Define the observation locations as an (N, 3) numpy array or load them
xr = np.linspace(-80., 80., 17)
yr = np.linspace(-80., 80., 17)
xr, yr = np.meshgrid(xr, yr)
xr, yr = mkvc(xr.T), mkvc(yr.T)
fun_interp = LinearNDInterpolator(np.c_[xx, yy], zz)
zr = fun_interp(np.c_[xr, yr]) + 2.
rx_locs = np.c_[xr, yr, zr]

# Define the survey
rx_list = PF.BaseGrav.RxObs(rx_locs)            # Define receivers
src_field = PF.BaseGrav.SrcField([rx_list])     # Define the source field
survey = PF.BaseGrav.LinearSurvey(src_field)    # Define the survey
survey2 = PF.BaseGrav.LinearSurvey(src_field)   # Define the survey


#############################################
# Defining a Tensor Mesh
# ----------------------
#
# Here, we create the tensor mesh that will be used to predict gravity anomaly
# data..
#

dh = 5.
hx = [(dh, 5, -1.3), (dh, 40), (dh, 5, 1.3)]
hy = [(dh, 5, -1.3), (dh, 40), (dh, 5, 1.3)]
hz = [(dh, 5, -1.3), (dh, 15)]
mesh = TensorMesh([hx, hy, hz], 'CCN')

#############################################
# Density Contrast Model and Mapping on Tensor Mesh
# -------------------------------------------------
#
# Here, we create the density contrast model that will be used to predict
# gravity anomaly data
# and the mapping from the model to the mesh. The model
# consists of a less dense block and a more dense sphere.
#

# Define density contrast values for each unit in g/cc
background_val = 0.
block_val = -0.1
sphere_val = 0.1

# Find the indecies of the active cells in forward model (ones below surface)
ind_active = surface2ind_topo(mesh, topo)

# Define mapping from model to active cells
nC = int(ind_active.sum())
mod_map = maps.IdentityMap(nP=nC)  # model consists of a value for each cell

# Define model
mod = background_val*np.ones(nC)

ind_block = (
    (mesh.gridCC[ind_active, 0] > -50.) & (mesh.gridCC[ind_active, 0] < -20.) &
    (mesh.gridCC[ind_active, 1] > -15.) & (mesh.gridCC[ind_active, 1] < 15.) &
    (mesh.gridCC[ind_active, 2] > -50.) & (mesh.gridCC[ind_active, 2] < -30.)
)
mod[ind_block] = block_val

ind_sphere = ModelBuilder.getIndicesSphere(
    np.r_[35., 0., -40.], 15., mesh.gridCC
)
ind_sphere = ind_sphere[ind_active]
mod[ind_sphere] = sphere_val

# Plot Density Contrast Model
fig = plt.figure(figsize=(9, 4))
plotting_map = maps.InjectActiveCells(mesh, ind_active, np.nan)

ax1 = fig.add_axes([0.05, 0.05, 0.78, 0.9])
mesh.plotSlice(
    plotting_map*mod, normal='Y', ax=ax1, ind=int(mesh.nCy/2), grid=True,
    clim=(np.min(mod), np.max(mod)), pcolorOpts={'cmap': 'jet'}
    )
ax1.set_title('Model slice at y = 0 m')

ax2 = fig.add_axes([0.85, 0.05, 0.05, 0.9])
norm = mpl.colors.Normalize(vmin=np.min(mod), vmax=np.max(mod))
cbar = mpl.colorbar.ColorbarBase(
        ax2, norm=norm, orientation='vertical', cmap='jet'
        )
cbar.set_label(
    '$g/cm^3$',
    rotation=270, labelpad=15, size=12
)

plt.show()


##############################################
# Gravity Anomaly Data on Tensor Mesh
# -----------------------------------
#
# Here we demonstrate how to predict gravity anomaly data using the integral
# formulation.
# 

# Define the problem. Define the cells below topography and the mapping
prob = PF.Gravity.GravityIntegral(mesh, rhoMap=mod_map, actInd=ind_active)

# Pair the survey and problem
survey.pair(prob)

# Compute predicted data for some model
dpred = prob.fields(mod)

# THIS IS TO WRITE THE DATA OUT FOR NOW FOR INVERSION
dpred = dpred + 5e-4*np.random.rand(len(dpred))
fname = os.path.dirname(PF.__file__) + '\\..\\..\\tutorials\\assets\\gravity_data.txt'
np.savetxt(
    fname,
    np.c_[rx_locs, dpred],
    fmt='%.4e'
)

# Plot
fig = plt.figure(figsize=(7, 5))

ax1 = fig.add_axes([0.05, 0.05, 0.8, 0.9])
plot2Ddata(rx_list.locs, dpred, ax=ax1, contourOpts={"cmap": "RdBu_r"})
ax1.set_title('Gravity Anomaly')

ax2 = fig.add_axes([0.82, 0.05, 0.03, 0.9])
norm = mpl.colors.Normalize(
    vmin=-np.max(np.abs(dpred)), vmax=np.max(np.abs(dpred))
)
cbar = mpl.colorbar.ColorbarBase(
        ax2, norm=norm, orientation='vertical', cmap='RdBu_r', format='%.1e'
        )
cbar.set_label('$mgal$', rotation=270, labelpad=15, size=12)

plt.show()

##########################################################
# Defining an OcTree Mesh
# -----------------------
#
# Here, we create the OcTree mesh that will be used to predict gravity
# gradiometry data.
# 

dx = 5    # minimum cell width (base mesh cell width) in x
dy = 5    # minimum cell width (base mesh cell width) in y
dz = 5    # minimum cell width (base mesh cell width) in z

x_length = 240.     # domain width in x
y_length = 240.     # domain width in y
z_length = 120.     # domain width in y

# Compute number of base mesh cells required in x and y
nbcx = 2**int(np.round(np.log(x_length/dx)/np.log(2.)))
nbcy = 2**int(np.round(np.log(y_length/dy)/np.log(2.)))
nbcz = 2**int(np.round(np.log(z_length/dz)/np.log(2.)))

# Define the base mesh
hx = [(dx, nbcx)]
hy = [(dy, nbcy)]
hz = [(dz, nbcz)]
mesh = TreeMesh([hx, hy, hz], x0='CCN')

# Refine based on surface topography
mesh = refine_tree_xyz(
    mesh, topo, octree_levels=[2, 2], method='surface', finalize=False
)

# Refine box based on region of interest
xp, yp, zp = np.meshgrid([-100., 100.], [-100., 100.], [-80., 0.])
xyz = np.c_[mkvc(xp), mkvc(yp), mkvc(zp)]

mesh = refine_tree_xyz(
    mesh, xyz, octree_levels=[2, 2], method='box', finalize=False
)

mesh.finalize()

#######################################################
# Density Contrast Model and Mapping on OcTree Mesh
# -------------------------------------------------
#
# Here, we create the density contrast model that will be used to predict gravity gradiometry
# data and the mapping from the model to the mesh. The model
# consists of a less dense block and a more dense sphere.
#

# Define density contrast values for each unit in g/cc
background_val = 0.
block_val = -0.1
sphere_val = 0.1

# Define active cells in the forward model (ones below surface)
ind_active = surface2ind_topo(mesh, topo)

# Define mapping from model to active cells
nC = int(ind_active.sum())
mod_map = maps.IdentityMap(nP=nC)  # model will be value of active cells

# Define model
mod = background_val*np.ones(nC)

ind_block = (
    (mesh.gridCC[ind_active, 0] > -50.) & (mesh.gridCC[ind_active, 0] < -20.) &
    (mesh.gridCC[ind_active, 1] > -15.) & (mesh.gridCC[ind_active, 1] < 15.) &
    (mesh.gridCC[ind_active, 2] > -50.) & (mesh.gridCC[ind_active, 2] < -30.)
)
mod[ind_block] = block_val

ind_sphere = ModelBuilder.getIndicesSphere(
    np.r_[35., 0., -40.], 15., mesh.gridCC
)
ind_sphere = ind_sphere[ind_active]
mod[ind_sphere] = sphere_val

# Plot Density Contrast Model
fig = plt.figure(figsize=(9, 4))
plotting_map = maps.InjectActiveCells(mesh, ind_active, np.nan)

ax1 = fig.add_axes([0.05, 0.05, 0.75, 0.9])
mesh.plotSlice(
    plotting_map*mod, normal='Y', ax=ax1, ind=int(mesh.hy.size/2), grid=True,
    clim=(np.min(mod), np.max(mod)), pcolorOpts={'cmap': 'jet'}
)
ax1.set_title('Model slice at y = 0 m')

ax2 = fig.add_axes([0.85, 0.05, 0.05, 0.9])
norm = mpl.colors.Normalize(vmin=np.min(mod), vmax=np.max(mod))
cbar = mpl.colorbar.ColorbarBase(
        ax2, norm=norm, orientation='vertical', cmap='jet'
)
cbar.set_label('$g/cm^3$', rotation=270, labelpad=15, size=12)

plt.show()

##############################################
# Gravity Gradiometry Data on OcTree Mesh
# ---------------------------------------
#
# Here we demonstrate how to predict gravity gradiometry data using the integral
# formulation. Here we have flat topography and all mesh cells are used in
# the forward model. The model consists of a less dense block and a more
# dense sphere than the background.
# 








