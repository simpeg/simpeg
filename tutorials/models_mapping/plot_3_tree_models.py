
"""
Tree Meshes
===========

Here we demonstrate various ways that models can be defined and mapped to
OcTree meshes. Some things we consider are:

    - Mesh refinement near surface topography
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
from discretize.utils import meshutils
import numpy as np
import matplotlib.pyplot as plt

# sphinx_gallery_thumbnail_number = 3

#############################################
# Defining the mesh
# -----------------
#
# Here, we create the OcTree mesh that will be used for all examples.
#

def makeExampleMesh():

    # Base mesh parameters
    dh = 5.   # base cell size
    nbc = 32  # total width of mesh in terms of number of base mesh cells
    h = dh*np.ones(nbc)

    mesh = Mesh.TreeMesh([h, h, h], x0='CCC')

    # Refine to largest possible cell size
    mesh.refine(3, finalize=False)

    return mesh


def refineTopography(mesh):

    # Define topography and refine
    [xx, yy] = np.meshgrid(mesh.vectorNx, mesh.vectorNy)
    zz = -3*np.exp((xx**2 + yy**2) / 60**2) + 45.
    topo = np.c_[Utils.mkvc(xx), Utils.mkvc(yy), Utils.mkvc(zz)]

    mesh = meshutils.refine_tree_xyz(mesh, topo,
                                     octree_levels=[3, 2],
                                     method='surface',
                                     finalize=False
                                     )

    return mesh


def refineBox(mesh):

    # Refine for sphere
    xp, yp, zp = np.meshgrid([-55., 50.], [-50., 50.], [-40., 20.])
    xyz = np.c_[Utils.mkvc(xp), Utils.mkvc(yp), Utils.mkvc(zp)]

    mesh = meshutils.refine_tree_xyz(mesh, xyz,
                                     octree_levels=[2],
                                     method='box',
                                     finalize=False
                                     )

    return mesh


#############################################
# Topography, a block and a vertical dyke
# ---------------------------------------
#
# In this example we create a model containing a block and a vertical dyke
# that strikes along the y direction. The utility *surface2ind_topo* is used
# to find the cells which lie below a set of xyz points defining a surface.
# The model consists of all cells which lie below the surface.
#

mesh = makeExampleMesh()
mesh = refineTopography(mesh)
mesh = refineBox(mesh)
mesh.finalize()

m_back = 100.
m_dyke = 40.
m_block = 70.

# Define surface topography as an (N, 3) np.array. You could also load a file
# containing the xyz points
[xx, yy] = np.meshgrid(mesh.vectorNx, mesh.vectorNy)
zz = -3*np.exp((xx**2 + yy**2) / 60**2) + 45.
topo = np.c_[Utils.mkvc(xx), Utils.mkvc(yy), Utils.mkvc(zz)]

# Find cells below topography and define mapping
m_air = 0.
actv = Utils.surface2ind_topo(mesh, topo)
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

# We can plot a slice of the model at Y=-2.5
Fig = plt.figure(figsize=(5, 5))
Ax = Fig.add_subplot(111)
mesh.plotSlice(modMap*mod, normal='Y', ax=Ax,
               ind=int(mesh.hy.size/2), grid=True)
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
mesh = refineTopography(mesh)
mesh = refineBox(mesh)
mesh.finalize()

m_back = np.log(1./100.)
m_dyke = np.log(1./40.)
m_block = np.log(1./70.)

# Define surface topography
[xx, yy] = np.meshgrid(mesh.vectorNx, mesh.vectorNy)
zz = -3*np.exp((xx**2 + yy**2) / 60**2) + 45.
topo = np.c_[Utils.mkvc(xx), Utils.mkvc(yy), Utils.mkvc(zz)]

# Find cells below topography
m_air = 0.
actv = Utils.surface2ind_topo(mesh, topo)
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
mesh.plotSlice(modMap*mod, normal='Y', ax=Ax,
               ind=int(mesh.hy.size/2), grid=True)
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
mesh = refineTopography(mesh)
mesh = refineBox(mesh)
mesh.finalize()

m_back = 100.
m_dyke = 40.
m_sph = 70.

# Define surface topography
[xx, yy] = np.meshgrid(mesh.vectorNx, mesh.vectorNy)
zz = -3*np.exp((xx**2 + yy**2) / 60**2) + 45.
topo = np.c_[Utils.mkvc(xx), Utils.mkvc(yy), Utils.mkvc(zz)]

# Set active cells and define unit values
m_air = 0.
actv = Utils.surface2ind_topo(mesh, topo)
modMap = Maps.InjectActiveCells(mesh, actv, m_air)

# Define model for cells under the surface topography
mod = m_back*np.ones(actv.sum())

# Add a sphere
sphInd = Utils.ModelBuilder.getIndicesSphere(np.r_[-25., 0., -15.],
                                             20., mesh.gridCC)
sphInd = sphInd[actv]  # So same size and order as model
mod[sphInd] = m_sph

# Add dyke defined by a set of points
xp = np.kron(np.ones((2)), [-10., 10., 55., 35.])
yp = np.kron([-1000., 1000.], np.ones((4)))
zp = np.kron(np.ones((2)), [-120., -120., 45., 45.])
xyz_pts = np.c_[Utils.mkvc(xp), Utils.mkvc(yp), Utils.mkvc(zp)]
polyInd = Utils.ModelBuilder.PolygonInd(mesh, xyz_pts)
polyInd = polyInd[actv]  # So same size and order as model
mod[polyInd] = m_dyke

# Plot
Fig = plt.figure(figsize=(5, 5))
Ax = Fig.add_subplot(111)
mesh.plotSlice(modMap*mod, normal='Y', ax=Ax,
               ind=int(mesh.hy.size/2), grid=True)
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
mesh = refineTopography(mesh)
mesh = refineBox(mesh)
mesh.finalize()

m_0 = 100.                   # background value
m_blk = 40.                  # block value
xc, yc, zc = -20., 0., -20.  # center of block
dx, dy, dz = 25., 40., 30.   # dimensions in x,y,z

# Define surface topography
[xx, yy] = np.meshgrid(mesh.vectorNx, mesh.vectorNy)
zz = -3*np.exp((xx**2 + yy**2) / 60**2) + 45.
topo = np.c_[Utils.mkvc(xx), Utils.mkvc(yy), Utils.mkvc(zz)]

# Set active cells and define unit values
m_air = 0.
actv = Utils.surface2ind_topo(mesh, topo)
actvMap = Maps.InjectActiveCells(mesh, actv, m_air)

# Define the model on subsurface cells
mod = np.r_[m_0, m_blk, xc, dx, yc, dy, zc, dz]
paramMap = Maps.ParametricBlock(mesh, indActive=actv, epsilon=1e-10, p=5.)

# Define a single mapping from model to mesh
modMap = Maps.ComboMap([actvMap, paramMap])

# Plot
Fig = plt.figure(figsize=(5, 5))
Ax = Fig.add_subplot(111)
mesh.plotSlice(modMap*mod, normal='Y', ax=Ax,
               ind=int(mesh.hy.size/2), grid=True)
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
mesh = refineTopography(mesh)
mesh = refineBox(mesh)
mesh.finalize()

sig_back = np.log(100.)
sig_sph = np.log(70.)
sig_dyke = np.log(40.)
mu_back = 1.
mu_sph = 1.25

# Define surface topography
[xx, yy] = np.meshgrid(mesh.vectorNx, mesh.vectorNy)
zz = -3*np.exp((xx**2 + yy**2) / 60**2) + 45.
topo = np.c_[Utils.mkvc(xx), Utils.mkvc(yy), Utils.mkvc(zz)]

# Set active cells
m_air = 0.
actv = Utils.surface2ind_topo(mesh, topo)
actvMap = Maps.InjectActiveCells(mesh, actv, m_air)

# Define model for cells under the surface topography
N = int(actv.sum())
mod = np.kron(np.ones((N, 1)), np.c_[sig_back, mu_back])

# Add a conductive and permeable sphere
sphInd = Utils.ModelBuilder.getIndicesSphere(np.r_[-20., 0., -15.],
                                             20., mesh.gridCC)
sphInd = sphInd[actv]  # So same size and order as model
mod[sphInd, :] = np.c_[sig_sph, mu_sph]

# Add a conductive and non-permeable dyke
xp = np.kron(np.ones((2)), [-10., 10., 55., 35.])
yp = np.kron([-1000., 1000.], np.ones((4)))
zp = np.kron(np.ones((2)), [-120., -120., 45., 45.])
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
mesh.plotSlice(sigMap*mod, normal='Y', ax=Ax,
               ind=int(mesh.hy.size/2), grid=True)
Ax.set_title('Model slice at y = -2.5 m')
plt.show()
