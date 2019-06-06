

"""
Cylindrical Meshes
==================

Cylindrical meshes are useful when the geological problem demonstrates
rotational symmetry. In this case, we need only define how the model changes
as a funcion of the radial distance and elevation; thus limiting the number
of model parameters. Here we demonstrate various ways that models can be
defined and mapped to cylindrical meshes. Some things we consider are:

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

#############################################
# Defining the mesh
# -----------------
#
# Here, we create the tensor mesh that will be used for all examples.
#


def makeExampleMesh():

    ncr = 20     # number of mesh cells in r
    ncz = 20     # number of mesh cells in z
    dh = 5.      # cell width

    hr = [(dh, ncr), (dh, 5, 1.3)]
    hz = [(dh, 5, -1.3), (dh, ncz), (dh, 5, 1.3)]

    # Use flag of 1 to denote perfect rotational symmetry
    mesh = Mesh.CylMesh([hr, 1, hz], '0CC')

    return mesh


#############################################
# Vertical Pipe and a 2 Layered Earth
# -----------------------------------
#
# In this example we create a model containing a vertical pipe and a layered
# Earth. We will see that we need only define the model as a function
# of r and z. Models of this type are plotted from the center of the mesh to
# the total radial distance of the mesh. That is why pipes and rings look like
# blocks.
#

mesh = makeExampleMesh()

m_back = 100.
m_layer = 70.
m_pipe = 40.

# Find cells below topography and define mapping
m_air = 0.
actv = mesh.gridCC[:, 2] < 0.
modMap = Maps.InjectActiveCells(mesh, actv, m_air)

# Define the model
mod = m_back*np.ones(actv.sum())
k_layer = ((mesh.gridCC[actv, 2] > -20.) & (mesh.gridCC[actv, 2] < -0))
mod[k_layer] = m_layer
k_pipe = ((mesh.gridCC[actv, 0] < 10.) &
          (mesh.gridCC[actv, 2] > -50.) & (mesh.gridCC[actv, 2] < 0.)
          )
mod[k_pipe] = m_pipe


# Plotting
Fig = plt.figure(figsize=(5, 5))
Ax = Fig.add_subplot(111)
mesh.plotImage(modMap*mod, ax=Ax, grid=True)
Ax.set_title('Cylindrically Symmetric Model')


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
m_layer = np.log(1./70.)
m_pipe = np.log(1./40.)


# Find cells below topography and define mapping
m_air = 0.
actv = mesh.gridCC[:, 2] < 0.
actvMap = Maps.InjectActiveCells(mesh, actv, m_air)

# Define the model
mod = m_back*np.ones(actv.sum())
k_layer = ((mesh.gridCC[actv, 2] > -20.) & (mesh.gridCC[actv, 2] < -0))
mod[k_layer] = m_layer
k_pipe = ((mesh.gridCC[actv, 0] < 10.) &
          (mesh.gridCC[actv, 2] > -50.) & (mesh.gridCC[actv, 2] < 0.)
          )
mod[k_pipe] = m_pipe

# Define a single mapping from model to mesh
expMap = Maps.ExpMap()
recMap = Maps.ReciprocalMap()
modMap = Maps.ComboMap([actvMap, recMap, expMap])

# Plotting
Fig = plt.figure(figsize=(5, 5))
Ax = Fig.add_subplot(111)
mesh.plotImage(modMap*mod, ax=Ax, grid=True)
Ax.set_title('Cylindrically Symmetric Model')


#############################################
# Parameterized pipe model
# ------------------------
#
# Instead of defining a model value for each sub-surface cell, we can define
# the model in terms of a small number of parameters. Here we parameterize the
# model as a block in a half-space. We then create a mapping which projects
# this model onto the mesh.
#

mesh = makeExampleMesh()

m_0 = 100.                   # background value
m_pipe = 40.                 # pipe value
rc, zc = 0., -25.            # center of pipe
dr, dz = 20., 50.            # dimensions in r, z

# Find cells below topography and define mapping
m_air = 0.
actv = mesh.gridCC[:, 2] < 0.
actvMap = Maps.InjectActiveCells(mesh, actv, m_air)

# Define the model on subsurface cells
mod = np.r_[m_0, m_pipe, rc, dr, 0., 1., zc, dz]  # add dummy values for phi
paramMap = Maps.ParametricBlock(mesh, indActive=actv, epsilon=1e-10, p=8.)

# Define a single mapping from model to mesh
modMap = Maps.ComboMap([actvMap, paramMap])

# Plotting
Fig = plt.figure(figsize=(5, 5))
Ax = Fig.add_subplot(111)
mesh.plotImage(modMap*mod, ax=Ax, grid=True)
Ax.set_title('Cylindrically Symmetric Model')


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
sig_layer = np.log(70.)
sig_pipe = np.log(40.)
mu_back = 1.
mu_pipe = 5.

# Find cells below topography and define mapping
m_air = 0.
actv = mesh.gridCC[:, 2] < 0.
actvMap = Maps.InjectActiveCells(mesh, actv, m_air)

# Define model for cells under the surface topography
N = int(actv.sum())
mod = np.kron(np.ones((N, 1)), np.c_[sig_back, mu_back])

# Add a conductive and non-permeable layer
k_layer = ((mesh.gridCC[actv, 2] > -20.) & (mesh.gridCC[actv, 2] < -0))
mod[k_layer, 0] = sig_layer

# Add a conductive and permeable pipe
k_pipe = ((mesh.gridCC[actv, 0] < 10.) &
          (mesh.gridCC[actv, 2] > -50.) & (mesh.gridCC[actv, 2] < 0.)
          )
mod[k_pipe] = np.c_[sig_pipe, mu_pipe]

# Create model vector and wires
mod = Utils.mkvc(mod)
wireMap = Maps.Wires(('logsig', N), ('mu', N))

# Use combo maps to map from model to mesh
sigMap = Maps.ComboMap([actvMap, Maps.ExpMap(), wireMap.logsig])
muMap = Maps.ComboMap([actvMap, wireMap.mu])

# Plotting
Fig = plt.figure(figsize=(5, 5))
Ax = Fig.add_subplot(111)
mesh.plotImage(sigMap*mod, ax=Ax, grid=True)
Ax.set_title('Cylindrically Symmetric Model')
