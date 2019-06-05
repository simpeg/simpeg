

"""
Tensor Meshes
=============

Here we demonstrate various ways that models can be defined and mapped to
tensor meshes. Some things we consider are:

    - Surface topography
    - Adding structures of various shape to the model
    - Parameterized models
    - Models with 2 or more physical properties
    

"""

#########################################################################
# Import modules
# --------------
#

from SimPEG import Mesh
from SimPEG import Utils
from SimPEG import Maps
import numpy as np
import matplotlib.pyplot as plt

# sphinx_gallery_thumbnail_number = 3

#############################################
# Defining the mesh
# -----------------
#
# Here, we create the tensor mesh that will be used for all examples.
#


def makeExampleMesh():

    dh = 5.
    hx = [(dh, 5, -1.3), (dh, 20), (dh, 5, 1.3)]
    hy = [(dh, 5, -1.3), (dh, 20), (dh, 5, 1.3)]
    hz = [(dh, 5, -1.3), (dh, 20), (dh, 5, 1.3)]
    mesh = Mesh.TensorMesh([hx, hy, hz], 'CCC')

    return mesh


#############################################
# Halfspace model with topography at z = 0
# ----------------------------------------
#
# In this example we generate a half-space model. Since air cells remain
# constant during geophysical inversion, the number of model values we define
# should be equal to the number of cells lying below the surface. Here, we
# define the model (*mod* ) as well as the mapping (*modMap* ) that goes from
# the model-space to the entire mesh.
#

mesh = makeExampleMesh()

m_val = 100.

# Find cells below topography and define mapping
m_air = 0.
actv = mesh.gridCC[:, 2] < 0.
modMap = Maps.InjectActiveCells(mesh, actv, m_air)

# Define the model
mod = m_val*np.ones(actv.sum())

# We can plot a slice of the model at Y=-2.5
Fig = plt.figure(figsize=(5, 5))
Ax = Fig.add_subplot(111)
mesh.plotSlice(modMap*mod, normal='Y', ax=Ax, ind=int(mesh.nCy/2), grid=True)
Ax.set_title('Model slice at y = -2.5 m')
plt.show()

#############################################
# Topography, a block and a vertical dyke
# ---------------------------------------
#
# In this example we create a model containing a block and a vertical dyke
# that strikes along the y direction. The utility *surface2ind_topo* is used
# to find the cells which lie below a set of xyz points defining a surface.
#

mesh = makeExampleMesh()

m_back = 100.
m_dyke = 40.
m_block = 70.

# Define surface topography as an (N, 3) np.array. You could also load a file
# containing the xyz points
[xx, yy] = np.meshgrid(mesh.vectorNx, mesh.vectorNy)
zz = -3*np.exp((xx**2 + yy**2) / 75**2) + 40.
topo = np.c_[Utils.mkvc(xx), Utils.mkvc(yy), Utils.mkvc(zz)]

# Find cells below topography and define mapping
m_air = 0.
actv = Utils.surface2ind_topo(mesh, topo, 'N')
modMap = Maps.InjectActiveCells(mesh, actv, m_air)

# Define the model on subsurface cells
mod = m_back*np.ones(actv.sum())
k_dyke = (mesh.gridCC[actv, 0] > 20.) & (mesh.gridCC[actv, 0] < 40.)
mod[k_dyke] = m_dyke
k_block = ((mesh.gridCC[actv, 0] > -40.) & (mesh.gridCC[actv, 0] < -10.) &
           (mesh.gridCC[actv, 1] > -30.) & (mesh.gridCC[actv, 1] < 30.) &
           (mesh.gridCC[actv, 2] > -40.) & (mesh.gridCC[actv, 2] < 0.)
           )
mod[k_block] = m_block

# Plot
Fig = plt.figure(figsize=(5, 5))
Ax = Fig.add_subplot(111)
mesh.plotSlice(modMap*mod, normal='Y', ax=Ax, ind=int(mesh.nCy/2), grid=True)
Ax.set_title('Model slice at y = -2.5 m')
plt.show()


#############################################
# Combo Maps
# ----------
#
# Here we demonstrate how combo maps can be used to create a single mapping
# from the model to the mesh. In this case, our model consists of
# log-conductivity values but we want to plot the resistivity. To accomplish
# this we must take the exponent of our model values, then take the reciprocal,
# then map from below surface cell to the mesh.
#

mesh = makeExampleMesh()

m_back = np.log(1./100.)
m_dyke = np.log(1./40.)
m_block = np.log(1./70.)

# Define surface topography
[xx, yy] = np.meshgrid(mesh.vectorNx, mesh.vectorNy)
zz = -3*np.exp((xx**2 + yy**2) / 75**2) + 40.
topo = np.c_[Utils.mkvc(xx), Utils.mkvc(yy), Utils.mkvc(zz)]

# Find cells below topography
m_air = 0.
actv = Utils.surface2ind_topo(mesh, topo, 'N')
actvMap = Maps.InjectActiveCells(mesh, actv, m_air)

# Define the model on subsurface cells
mod = m_back*np.ones(actv.sum())
k_dyke = (mesh.gridCC[actv, 0] > 20.) & (mesh.gridCC[actv, 0] < 40.)
mod[k_dyke] = m_dyke
k_block = ((mesh.gridCC[actv, 0] > -40.) & (mesh.gridCC[actv, 0] < -10.) &
           (mesh.gridCC[actv, 1] > -30.) & (mesh.gridCC[actv, 1] < 30.) &
           (mesh.gridCC[actv, 2] > -40.) & (mesh.gridCC[actv, 2] < 0.)
           )
mod[k_block] = m_block

# Define a single mapping from model to mesh
expMap = Maps.ExpMap()
recMap = Maps.ReciprocalMap()
modMap = Maps.ComboMap([actvMap, recMap, expMap])

# Plot
Fig = plt.figure(figsize=(5, 5))
Ax = Fig.add_subplot(111)
mesh.plotSlice(modMap*mod, normal='Y', ax=Ax, ind=int(mesh.nCy/2), grid=True)
Ax.set_title('Model slice at y = -2.5 m')
plt.show()


#############################################
# Models with arbitrary shapes
# ----------------------------
#
# Here we show how model building utilities are used to make more complicated
# structural models. The process of adding a new unit is twofold: 1) we must
# find the indicies for mesh cells that lie within the new unit, 2) we
# replace the prexisting physical property value for those cells.
#

mesh = makeExampleMesh()

m_back = 100.
m_dyke = 40.
m_sph = 70.

# Define surface topography
[xx, yy] = np.meshgrid(mesh.vectorNx, mesh.vectorNy)
zz = -3*np.exp((xx**2 + yy**2) / 75**2) + 40.
topo = np.c_[Utils.mkvc(xx), Utils.mkvc(yy), Utils.mkvc(zz)]

# Set active cells and define unit values
m_air = 0.
actv = Utils.surface2ind_topo(mesh, topo, 'N')
modMap = Maps.InjectActiveCells(mesh, actv, m_air)

# Define model for cells under the surface topography
mod = m_back*np.ones(actv.sum())

# Add a sphere
sphInd = Utils.ModelBuilder.getIndicesSphere(np.r_[-25., 0., -15.],
                                             20., mesh.gridCC)
sphInd = sphInd[actv]  # So it's same size and order as model
mod[sphInd] = m_sph

# Add dyke defined by a set of points
xp = np.kron(np.ones((2)), [-10., 10., 45., 25.])
yp = np.kron([-1000., 1000.], np.ones((4)))
zp = np.kron(np.ones((2)), [-120., -120., 35., 35.])
xyz_pts = np.c_[Utils.mkvc(xp), Utils.mkvc(yp), Utils.mkvc(zp)]
polyInd = Utils.ModelBuilder.PolygonInd(mesh, xyz_pts)
polyInd = polyInd[actv]  # So same size and order as model
mod[polyInd] = m_dyke

# Plot
Fig = plt.figure(figsize=(5, 5))
Ax = Fig.add_subplot(111)
mesh.plotSlice(modMap*mod, normal='Y', ax=Ax, ind=int(mesh.nCy/2), grid=True)
Ax.set_title('Model slice at y = -2.5 m')
plt.show()


#############################################
# Parameterized block model
# -------------------------
#
# Instead of defining a model value for each sub-surface cell, we can define
# the model in terms of a small number of parameters. Here we parameterize the
# model as a block in a half-space. We then create a mapping which projects
# this model onto the mesh.
#

mesh = makeExampleMesh()

m_0 = 100.                   # background value
m_blk = 40.                  # block value
xc, yc, zc = -25., 0., -20.  # center of block
dx, dy, dz = 30., 40., 30.   # dimensions in x,y,z

# Define surface topography
[xx, yy] = np.meshgrid(mesh.vectorNx, mesh.vectorNy)
zz = -3*np.exp((xx**2 + yy**2) / 75**2) + 40.
topo = np.c_[Utils.mkvc(xx), Utils.mkvc(yy), Utils.mkvc(zz)]

# Set active cells and define unit values
m_air = 0.
actv = Utils.surface2ind_topo(mesh, topo, 'N')
actvMap = Maps.InjectActiveCells(mesh, actv, m_air)

# Define the model on subsurface cells
mod = np.r_[m_0, m_blk, xc, dx, yc, dy, zc, dz]
paramMap = Maps.ParametricBlock(mesh, indActive=actv, epsilon=1e-10, p=5.)

# Define a single mapping from model to mesh
modMap = Maps.ComboMap([actvMap, paramMap])

# Plot
Fig = plt.figure(figsize=(5, 5))
Ax = Fig.add_subplot(111)
mesh.plotSlice(modMap*mod, normal='Y', ax=Ax, ind=int(mesh.nCy/2), grid=True)
Ax.set_title('Model slice at y = -2.5 m')
plt.show()


#############################################
# Using Wire Maps
# ---------------
#
# Wire maps are needed when the model is comprised of two or more parameter
# types (e.g. conductivity and magnetic permeability). Because the model
# vector contains all values for all parameter types, we need to use "wires"
# to extract the values for a particular parameter type.
#
# Here we will define a model consisting of log-conductivity values and
# magnetic permeability values. We wish to plot the conductivity and
# permeability on the mesh. Wires are used to keep track of the mapping
# between the model vector and a particular physical property type.
#

mesh = makeExampleMesh()

sig_back = np.log(100.)
sig_sph = np.log(70.)
sig_dyke = np.log(40.)
mu_back = 1.
mu_sph = 1.25

# Define surface topography
[xx, yy] = np.meshgrid(mesh.vectorNx, mesh.vectorNy)
zz = -3*np.exp((xx**2 + yy**2) / 75**2) + 40.
topo = np.c_[Utils.mkvc(xx), Utils.mkvc(yy), Utils.mkvc(zz)]

# Set active cells
m_air = 0.
actv = Utils.surface2ind_topo(mesh, topo, 'N')
actvMap = Maps.InjectActiveCells(mesh, actv, m_air)

# Define model for cells under the surface topography
N = int(actv.sum())
mod = np.kron(np.ones((N, 1)), np.c_[sig_back, mu_back])

# Add a conductive and permeable sphere
sphInd = Utils.ModelBuilder.getIndicesSphere(np.r_[-25., 0., -15.],
                                             20., mesh.gridCC)
sphInd = sphInd[actv]  # So same size and order as model
mod[sphInd, :] = np.c_[sig_sph, mu_sph]

# Add a conductive and non-permeable dyke
xp = np.kron(np.ones((2)), [-10., 10., 45., 25.])
yp = np.kron([-1000., 1000.], np.ones((4)))
zp = np.kron(np.ones((2)), [-120., -120., 35., 35.])
xyz_pts = np.c_[Utils.mkvc(xp), Utils.mkvc(yp), Utils.mkvc(zp)]
polyInd = Utils.ModelBuilder.PolygonInd(mesh, xyz_pts)
polyInd = polyInd[actv]  # So same size and order as model
mod[polyInd, 0] = sig_dyke

# Create model vector and wires
mod = Utils.mkvc(mod)
wireMap = Maps.Wires(('logsig', N), ('mu', N))

# Use combo maps to map from model to mesh
sigMap = Maps.ComboMap([actvMap, Maps.ExpMap(), wireMap.logsig])
muMap = Maps.ComboMap([actvMap, wireMap.mu])

# Plot
Fig = plt.figure(figsize=(5, 5))
Ax = Fig.add_subplot(111)
mesh.plotSlice(sigMap*mod, normal='Y', ax=Ax, ind=int(mesh.nCy/2), grid=True)
Ax.set_title('Model slice at y = -2.5 m')
plt.show()
