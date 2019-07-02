

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

from discretize import CylMesh
from SimPEG.Utils import mkvc
from SimPEG import Maps
import numpy as np
import matplotlib.pyplot as plt

#############################################
# Defining the mesh
# -----------------
#
# Here, we create the tensor mesh that will be used for all examples.
#


def make_example_mesh():

    ncr = 20     # number of mesh cells in r
    ncz = 20     # number of mesh cells in z
    dh = 5.      # cell width

    hr = [(dh, ncr), (dh, 5, 1.3)]
    hz = [(dh, 5, -1.3), (dh, ncz), (dh, 5, 1.3)]

    # Use flag of 1 to denote perfect rotational symmetry
    mesh = CylMesh([hr, 1, hz], '0CC')

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

mesh = make_example_mesh()

background_val = 100.
layer_val = 70.
pipe_val = 40.

# Find cells below topography and define mapping
air_val = 0.
actv = mesh.gridCC[:, 2] < 0.
mod_map = Maps.InjectActiveCells(mesh, actv, air_val)

# Define the model
mod = background_val*np.ones(actv.sum())
k_layer = ((mesh.gridCC[actv, 2] > -20.) & (mesh.gridCC[actv, 2] < -0))
mod[k_layer] = layer_val
k_pipe = (
    (mesh.gridCC[actv, 0] < 10.) &
    (mesh.gridCC[actv, 2] > -50.) &
    (mesh.gridCC[actv, 2] < 0.)
)
mod[k_pipe] = pipe_val


# Plotting
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111)
mesh.plotImage(mod_map*mod, ax=ax, grid=True)
ax.set_title('Cylindrically Symmetric Model')


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

mesh = make_example_mesh()

background_val = np.log(1./100.)
layer_val = np.log(1./70.)
pipe_val = np.log(1./40.)


# Find cells below topography and define mapping
air_val = 0.
actv = mesh.gridCC[:, 2] < 0.
actv_map = Maps.InjectActiveCells(mesh, actv, air_val)

# Define the model
mod = background_val*np.ones(actv.sum())
k_layer = ((mesh.gridCC[actv, 2] > -20.) & (mesh.gridCC[actv, 2] < -0))
mod[k_layer] = layer_val
k_pipe = (
    (mesh.gridCC[actv, 0] < 10.) &
    (mesh.gridCC[actv, 2] > -50.) & (mesh.gridCC[actv, 2] < 0.)
)
mod[k_pipe] = pipe_val

# Define a single mapping from model to mesh
exp_map = Maps.ExpMap()
rec_map = Maps.ReciprocalMap()
mod_map = Maps.ComboMap([actv_map, rec_map, exp_map])

# Plotting
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111)
mesh.plotImage(mod_map*mod, ax=ax, grid=True)
ax.set_title('Cylindrically Symmetric Model')


#############################################
# Parameterized pipe model
# ------------------------
#
# Instead of defining a model value for each sub-surface cell, we can define
# the model in terms of a small number of parameters. Here we parameterize the
# model as a block in a half-space. We then create a mapping which projects
# this model onto the mesh.
#

mesh = make_example_mesh()

background_val = 100.        # background value
pipe_val = 40.               # pipe value
rc, zc = 0., -25.            # center of pipe
dr, dz = 20., 50.            # dimensions in r, z

# Find cells below topography and define mapping
air_val = 0.
actv = mesh.gridCC[:, 2] < 0.
actv_map = Maps.InjectActiveCells(mesh, actv, air_val)

# Define the model on subsurface cells
mod = np.r_[background_val, pipe_val, rc, dr, 0., 1., zc, dz]  # add dummy values for phi
param_map = Maps.ParametricBlock(mesh, indActive=actv, epsilon=1e-10, p=8.)

# Define a single mapping from model to mesh
mod_map = Maps.ComboMap([actv_map, param_map])

# Plotting
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111)
mesh.plotImage(mod_map*mod, ax=ax, grid=True)
ax.set_title('Cylindrically Symmetric Model')


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

mesh = make_example_mesh()

sig_back = np.log(100.)
sig_layer = np.log(70.)
sig_pipe = np.log(40.)
mu_back = 1.
mu_pipe = 5.

# Find cells below topography and define mapping
air_val = 0.
actv = mesh.gridCC[:, 2] < 0.
actv_map = Maps.InjectActiveCells(mesh, actv, air_val)

# Define model for cells under the surface topography
N = int(actv.sum())
mod = np.kron(np.ones((N, 1)), np.c_[sig_back, mu_back])

# Add a conductive and non-permeable layer
k_layer = ((mesh.gridCC[actv, 2] > -20.) & (mesh.gridCC[actv, 2] < -0))
mod[k_layer, 0] = sig_layer

# Add a conductive and permeable pipe
k_pipe = (
    (mesh.gridCC[actv, 0] < 10.) &
    (mesh.gridCC[actv, 2] > -50.) &
    (mesh.gridCC[actv, 2] < 0.)
)
mod[k_pipe] = np.c_[sig_pipe, mu_pipe]

# Create model vector and wires
mod = mkvc(mod)
wire_map = Maps.Wires(('logsig', N), ('mu', N))

# Use combo maps to map from model to mesh
sig_map = Maps.ComboMap([actv_map, Maps.ExpMap(), wire_map.logsig])
mu_map = Maps.ComboMap([actv_map, wire_map.mu])

# Plotting
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111)
mesh.plotImage(sig_map*mod, ax=ax, grid=True)
ax.set_title('Cylindrically Symmetric Model')
