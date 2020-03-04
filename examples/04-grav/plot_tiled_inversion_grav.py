"""
PF: Gravity: Tiled Inversion Linear
===================================

Invert data in tiles.

"""
import numpy as np
import matplotlib.pyplot as plt

from discretize import TensorMesh
from SimPEG.potential_fields import gravity
from SimPEG import (
    maps, data, data_misfit, regularization, optimization, inverse_problem,
    directives, inversion
    )
from discretize.utils import mesh_builder_xyz, refine_tree_xyz

try:
    from SimPEG import utils
    from SimPEG.utils import plot2Ddata
except:
    from SimPEG import Utils as utils
    from SimPEG.Utils.Plotutils import plot2Ddata

import shutil

###############################################################################
# Setup
# -----
#
# Define the survey and model parameters
#
# Create an a global survey and mesh and simulate some data
#
#


# Create and array of observation points
xr = np.linspace(-30., 30., 20)
yr = np.linspace(-30., 30., 20)
X, Y = np.meshgrid(xr, yr)

# Move the observation points 5m above the topo
Z = -np.exp((X**2 + Y**2) / 75**2)

# Create a topo array
topo = np.c_[utils.mkvc(X.T), utils.mkvc(Y.T), utils.mkvc(Z.T)]

# Create station locations drapped 0.1 m above topo
rxLoc = np.c_[utils.mkvc(X.T), utils.mkvc(Y.T), utils.mkvc(Z.T)+0.1]

##########################################################################
# Divided and Conquer
# -------------------
#
# Split the data set in two and create sub-problems
#
#

local_surveys = []
local_meshes = []
local_indices = []

h = [5, 5, 5]
padDist = np.ones((3, 2)) * 100

# First tile
local_indices.append(rxLoc[:, 0] <= 0)
receivers = gravity.receivers.point_receiver(rxLoc[local_indices[0], :])
srcField = gravity.sources.SourceField([receivers])
local_surveys.append(gravity.survey.GravitySurvey(srcField))

# Create a local mesh that covers all points, but refined on the local survey
local_meshes.append(
    mesh_builder_xyz(
        topo, h, padding_distance=padDist, depth_core=100, mesh_type='tree'
    )
)
local_meshes[0] = refine_tree_xyz(
    local_meshes[0], local_surveys[0].receiver_locations,
    method='surface', octree_levels=[8, 4], finalize=True
)

# Second tile
local_indices.append(rxLoc[:, 0] > 0)
receivers = gravity.receivers.point_receiver(rxLoc[local_indices[1], :])
srcField = gravity.sources.SourceField([receivers])
local_surveys.append(gravity.survey.GravitySurvey(srcField))

# Create a local mesh that covers all points, but refined on the local survey
local_meshes.append(
    mesh_builder_xyz(
        topo, h, padding_distance=padDist, depth_core=100, mesh_type='tree'
    )
)
local_meshes[1] = refine_tree_xyz(
    local_meshes[1], local_surveys[1].receiver_locations,
    method='surface', octree_levels=[8, 4], finalize=True
)

fig = plt.figure(figsize=(12, 6))
for ii, local_mesh in enumerate(local_meshes):

    activeCells = utils.surface2ind_topo(local_mesh, topo)

    ax = plt.subplot(2, 2, ii+1)
    local_mesh.plotSlice(activeCells, normal='Y', ax=ax, grid=True)
    ax.set_aspect('equal')
    ax.set_title(f"Mesh {ii+1}. Active cells {activeCells.sum()}")

###############################################################################
# Global Mesh
# ------------
#
# Create a global mesh survey for simulation
#
#

mesh = mesh_builder_xyz(topo, h, padding_distance=padDist, depth_core=100, mesh_type='tree')

# This garantees that the local meshes are always coarser or equal
for local_mesh in local_meshes:
    mesh.insert_cells(
        local_mesh.gridCC,
        local_mesh.cell_levels_by_index(np.arange(local_mesh.nC)),
        finalize=False
    )
mesh.finalize()

# Define an active cells from topo
activeCells = utils.surface2ind_topo(mesh, topo)
nC = int(activeCells.sum())

ax = plt.subplot(2, 1, 2)
mesh.plotSlice(activeCells, normal='Y', ax=ax, grid=True)
ax.set_title(f"Global Mesh. Active cells {activeCells.sum()}")
ax.set_aspect('equal')
plt.show()

# We can now create a density model and generate data
# Here a simple block in half-space
# Get the indicies of the magnetized block
model = np.zeros(mesh.nC)
ind = utils.ModelBuilder.getIndicesBlock(
    np.r_[-10, -10, -30], np.r_[10, 10, -10],
    mesh.gridCC,
)[0]

# Assign magnetization values
model[ind] = 0.3

# Remove air cells
model = model[activeCells]

# Create reduced identity map
idenMap = maps.IdentityMap(nP=nC)

# Create a global survey just for simulation of data
receivers = gravity.receivers.point_receiver(rxLoc)
srcField = gravity.sources.SourceField([receivers])
survey = gravity.survey.GravitySurvey(srcField)

# Create the forward simulation for the global dataset
simulation = gravity.simulation.IntegralSimulation(
    survey=survey, mesh=mesh, rhoMap=idenMap, actInd=activeCells
)

# Compute linear forward operator and compute some data
d = simulation.fields(model)

# Add noise and uncertainties
# We add some random Gaussian noise (1nT)
synthetic_data = d + np.random.randn(len(d))*1e-3
wd = np.ones(len(synthetic_data))*1e-3  # Assign flat uncertainties

###############################################################
# Tiled misfits
#
#
#
#
local_misfits = []
for ii, local_survey in enumerate(local_surveys):

    tile_map = maps.TileMap(
        mesh, activeCells, local_meshes[ii]
    )

    local_actives = tile_map.local_active

    # Create the forward simulation
    simulation = gravity.simulation.IntegralSimulation(
        survey=local_survey, mesh=local_meshes[ii], rhoMap=tile_map,
        actInd=local_actives,
        sensitivity_path=f"Inversion\Tile{ii}.zarr"
    )

    data_object = data.Data(
        local_survey,
        dobs=synthetic_data[local_indices[ii]],
        uncertainty=wd[local_indices[ii]]
    )

    local_misfits.append(
        data_misfit.L2DataMisfit(data=data_object, simulation=simulation)
    )


# Our global misfit
global_misfit = local_misfits[0] + local_misfits[1]

# Create reduced identity map
idenMap = maps.IdentityMap(nP=nC)

# Create a regularization
reg = regularization.Sparse(mesh, indActive=activeCells, mapping=idenMap)

m0 = np.ones(nC)*1e-4  # Starting model

# Add directives to the inversion
opt = optimization.ProjectedGNCG(
    maxIter=100, lower=-1., upper=1.,
    maxIterLS=20, maxIterCG=10, tolCG=1e-3
)
invProb = inverse_problem.BaseInvProblem(global_misfit, reg, opt)
betaest = directives.BetaEstimate_ByEig(beta0_ratio=1e-1)

# Here is where the norms are applied
# Use pick a threshold parameter empirically based on the distribution of
# model parameters
update_IRLS = directives.Update_IRLS(
    f_min_change=1e-4, max_irls_iterations=0,
    coolEpsFact=1.5, beta_tol=1e-2,
)
saveDict = directives.SaveOutputEveryIteration(save_txt=False)
update_Jacobi = directives.UpdatePreconditioner()
sensitivity_weights = directives.UpdateSensitivityWeights(everyIter=False)
inv = inversion.BaseInversion(
    invProb,
    directiveList=[
        update_IRLS, sensitivity_weights, betaest, update_Jacobi, saveDict
    ]
)

# Run the inversion
mrec = inv.run(m0)

# Create active map to go from reduce set to full
activeCellsMap = maps.InjectActiveCells(mesh, activeCells, np.nan)

# Plot the result
ax = plt.subplot(1, 2, 1)
mesh.plotSlice(activeCellsMap*model, normal='Y', ax=ax, grid=True)
ax.set_title("True")
ax.set_aspect('equal')

ax = plt.subplot(1, 2, 2)
mesh.plotSlice(activeCellsMap*mrec, normal='Y', ax=ax, grid=True)
ax.set_title("Recovered")
ax.set_aspect('equal')
plt.show()
