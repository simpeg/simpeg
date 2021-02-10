"""
Magnetic inversion on a TreeMesh
================================

Invert data in tiles

"""

import numpy as np
import gc
import matplotlib.pyplot as plt
from discretize import TreeMesh
from discretize.utils import mesh_builder_xyz, refine_tree_xyz, active_from_xyz
from dask.distributed import Client, LocalCluster
from SimPEG import dask
import dask
from dask import config
from SimPEG.utils.drivers import create_tile_meshes, create_nested_mesh
from SimPEG.potential_fields import magnetics
from SimPEG import (
    maps,
    data,
    regularization,
    optimization,
    inverse_problem,
    directives,
    inversion,
    data_misfit,
    objective_function
)

import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing.pool import ThreadPool
try:
    from SimPEG import utils
    from SimPEG.utils import plot2Ddata, tile_locations
except:
    from SimPEG import Utils as utils
    from SimPEG.Utils.Plotutils import plot2Ddata

import shutil

from time import time
cluster = LocalCluster(processes=False)
client = Client(cluster)

# config.set(scheduler="threads", pool=ThreadPool(12))

max_chunk_size = 256

config.set({"array.chunk-size": str(max_chunk_size) + "MiB"})
workers = None  # Here runs locally


def create_tile(locations, obs, uncert, global_mesh, global_active, tile_id, components=['bxz'], h0=[50000, 0, 0], workers=None):
    max_chunk_size = 256
    receivers = magnetics.receivers.Point(locations, components=components)
    srcField = magnetics.sources.SourceField(receiver_list=[receivers], parameters=h0)
    local_survey = magnetics.survey.Survey(srcField)

    # Create tile map between global and local
    local_mesh = create_nested_mesh(
        receivers.locations, global_mesh,
    )
    local_map = maps.TileMap(global_mesh, global_active, local_mesh, components=3)
    print("[info] tile size: ", local_mesh.nC, local_map.local_active.sum(), local_map.shape)
    # Create the local misfit
    simulation = magnetics.simulation.Simulation3DIntegral(
        survey=local_survey,
        mesh=local_mesh,
        chiMap=maps.IdentityMap(nP=int(local_map.local_active.sum()) * 3),
        actInd=local_map.local_active,
        modelType="vector",
        sensitivity_path=f"./sensitivity/Inversion/Tile{tile_id}.zarr",
        chunk_format="row",
        store_sensitivities="disk",
        max_chunk_size=max_chunk_size,
        workers=workers
    )
    data_object = data.Data(
        local_survey,
        dobs=obs,
        standard_deviation=uncert,
    )
    local_misfit = data_misfit.L2DataMisfit(
        data=data_object, simulation=simulation,
        workers=workers, model_map=local_map
    )

    return local_misfit


# Create a mesh
topography_refinement = [0, 0, 2]
locations_refinement = [5, 5, 5]
max_distance = 25
h = [5, 5, 5]
padDist = np.ones((3, 2)) * 100

###############################################################################
# Setup
# -----
#
# Define the survey and model parameters
#
# First we need to define the direction of the inducing field
# As a simple case, we pick a vertical inducing field of magnitude 50,000 nT.
#
#
sp.random.seed(1)
# We will assume a vertical inducing field
H0 = (50000.0, 90.0, 0.0)
components = ["bxx", "byy", "bzz", "bxz", "bxy", "byz", "tmi"]

# The magnetization is set along a different direction (induced + remanence)
M = np.array([45.0, 90.0])

# Create grid of points for topography
# Lets create a simple Gaussian topo and set the active cells
[xx, yy] = np.meshgrid(np.linspace(-200, 200, 50), np.linspace(-200, 200, 50))
b = 100
A = 50
zz = A * np.exp(-0.5 * ((xx / b) ** 2.0 + (yy / b) ** 2.0))

topo = np.c_[utils.mkvc(xx), utils.mkvc(yy), utils.mkvc(zz)]

# Create and array of observation points
xr = np.linspace(-100.0, 100.0, 10)
yr = np.linspace(-100.0, 100.0, 10)
X, Y = np.meshgrid(xr, yr)
Z = A * np.exp(-0.5 * ((X / b) ** 2.0 + (Y / b) ** 2.0)) 
print(Z.max(), Z.min())
# Z = np.zeros_like(X) - 0.1

###############################################################################
# Inversion Mesh
# --------------
#
# Here, we create a TreeMesh with base cell size of 5 m. We created a small
# utility function to center the mesh around points and to figure out the
# outer most dimension for adequate padding distance.
# The second stage allows to refine the mesh around points or surfaces
# (point assumed to follow some horizontal trend)
# The refinement process is repeated twice to allow for a finer level around
# the survey locations.
#

# Move the observation points 5m above the topo
# Z = np.zeros_like(X) - 0.1 #-np.exp((X ** 2 + Y ** 2) / 75 ** 2)

# Create a topo array
# topo = np.c_[utils.mkvc(X.T), utils.mkvc(Y.T), utils.mkvc(Z.T)]

# Create station locations drapped 0.1 m above topo
rxLoc = np.c_[utils.mkvc(X.T), utils.mkvc(Y.T), utils.mkvc(Z.T) + 5.0]

global_mesh = mesh_builder_xyz(
    rxLoc, h,
    padding_distance=padDist,
    mesh_type='TREE',
    depth_core=200
)
global_mesh = refine_tree_xyz(
    global_mesh, topo,
    method='surface', octree_levels=topography_refinement,
    finalize=False
)

global_mesh = refine_tree_xyz(
    global_mesh, rxLoc,
    method='surface', octree_levels=locations_refinement,
    max_distance=max_distance,
    finalize=True
)
activeCells = active_from_xyz(global_mesh, topo, method='linear')
nC = int(activeCells.sum())

###########################################################################
# Forward modeling data
# ---------------------
#
# We can now create a magnetization model and generate data
# Lets start with a block below topography
#


model = np.zeros((global_mesh.nC, 3))

# Convert the inclination declination to vector in Cartesian
M_xyz = utils.mat_utils.dip_azimuth2cartesian(M[0], M[1])

# Get the indicies of the magnetized block
ind = utils.model_builder.getIndicesBlock(
    np.r_[-20, -20, -10], np.r_[20, 20, 25], global_mesh.gridCC,
)[0]

# Assign magnetization values
model[ind, :] = np.kron(np.ones((ind.shape[0], 1)), M_xyz * 0.05)

# Remove air cells
model = model[activeCells, :]

# Creat reduced identity map
idenMap = maps.IdentityMap(nP=nC * 3)

receivers = magnetics.receivers.Point(rxLoc, components=components)
srcField = magnetics.sources.SourceField(receiver_list=[receivers], parameters=H0)
survey = magnetics.survey.Survey(srcField)

# Create the simulation
simulation = magnetics.simulation.Simulation3DIntegral(
    survey=survey, mesh=global_mesh, chiMap=idenMap, actInd=activeCells, modelType="vector",
    store_sensitivities="forward_only",
    max_chunk_size=max_chunk_size,
)

# Compute some data and add some random noise
d = simulation.fields(utils.mkvc(model))
# d = client.compute(simulation.fields(utils.mkvc(model))).result()
std = 1  # nT

# Add noise and uncertainties
# We add some random Gaussian noise (1nT)
synthetic_data = d + np.random.randn(len(d)) * std
wd = np.ones(len(synthetic_data)) * std  # Assign flat uncertainties

prob_size = activeCells.sum() * 1e-9 * rxLoc.shape[0] * 8.
print(f"Size {activeCells.sum()} x {rxLoc.shape[0]}, Total size {prob_size}")

##########################################################################
# Divided and Conquer
# -------------------
#
# Split the data set in two and create sub-problems
#
#

# Create tiles of data
rxLoc = survey.receiver_locations
indices = tile_locations(rxLoc, 4, method="kmeans")
obs_r = np.reshape(synthetic_data, (rxLoc.shape[0], len(components)))
std_r = np.reshape(wd, (rxLoc.shape[0], len(components)))
ct = time()
local_misfits = []
for ii, local_index in enumerate(indices):
    locations = rxLoc[local_index, :]
    obs_sub = obs_r[local_index, :]
    std_sub = std_r[local_index, :]
    obs_sub = np.reshape(obs_sub, (obs_sub.size, ))
    std_sub = np.reshape(std_sub, (std_sub.size, ))
    local_misfits += [create_tile(
        locations,  obs_sub,
        std_sub, global_mesh, activeCells, ii, components=components, h0=H0, workers=None
    )]
    # local_misfits += [client.compute(delayed_misfit)]

# local_misfits[0].simulation.linear_operator()
# local_misfits = client.gather(local_misfits)
tile_time = time() - ct
print(f"Tile creation {tile_time}")

global_misfit = objective_function.ComboObjectiveFunction(
    local_misfits
)

# Trigger sensitivity calcs
ct = time()
for local in local_misfits:
    local.simulation.Jmatrix
    local.simulation.Jmatrix
    # del local.simulation.mesh

print('[info] global: ', global_mesh.nC, activeCells.sum(), synthetic_data.shape)
# Plot the model on different meshes
# fig = plt.figure(figsize=(12, 6))
# c_code = ['r', 'g', 'b', 'm']
# for ii, local_misfit in enumerate(global_misfit.objfcts):

#     local_mesh = local_misfit.simulation.mesh
#     local_map = local_misfit.model_map
#     print('[info] type: ', type(local_map))
#     inject_local = maps.InjectActiveCells(local_mesh, local_map.local_active, np.nan)

#     # Interpolate values to mesh.gridCC if not 'CC'
#     m = local_map * utils.mkvc(model)
#     # m = np.reshape(m, (local_map.local_active.sum(), 3))
#     nC_t = local_map.local_active.sum()
#     mx = m[:nC_t]
#     my = m[nC_t:2*nC_t]
#     mz = m[2*nC_t:]

#     m = np.c_[mx, my, mz]
#     amp = np.sum(m ** 2.0, axis=1) ** 0.5

#     ax = plt.subplot(2, 3, ii + 1)
#     local_mesh.plot_slice(
#         inject_local * (amp), index=100, normal="Z", ax=ax, grid=True
#     )
#     sensors = local_misfit.simulation.survey.receiver_locations
#     ax.scatter(sensors[:, 0], sensors[:, 1], 10)
#     ax.set_xlim(-200, 200)
#     ax.set_ylim(-200, 200)
#     ax.set_aspect("equal")
#     ax.set_title(f"Mesh {ii+1}. Active cells {local_map.local_active.sum()}")


# Create active map to go from reduce set to full
inject_global = maps.InjectActiveCells(global_mesh, activeCells, 0)

# ax = plt.subplot(2, 3, 6)
mx = model[:, 0]
my = model[:, 1]
mz = model[:, 2]

m_ = np.c_[mx, my, mz]
amp = np.sum(m_ ** 2.0, axis=1) ** 0.5
# global_mesh.plot_slice(inject_global * amp, index=100, normal="Z", ax=ax, grid=True)
# ax.scatter(rxLoc[:, 0], rxLoc[:, 1], 10)
# ax.set_title(f"Global Mesh. Active cells {activeCells.sum()}")
# ax.set_xlim(-200, 200)
# ax.set_ylim(-200, 200)
# ax.set_aspect("equal")
# plt.show()


# ####################################################
# Invert on the global mesh

# This Mapping connects the regularizations for the three-component
# vector model
wires = maps.Wires(("p", nC), ("s", nC), ("t", nC))


m0 = np.ones(3 * nC) * 1e-4  # Starting model
wr = np.zeros_like(m0)
norm = np.tile(global_mesh.cell_volumes[activeCells]**2., 3)
for ii, dmisfit in enumerate(global_misfit.objfcts):
    wr += dmisfit.getJtJdiag(m0) / norm

wr += np.percentile(wr, 40)
wr *= norm
wr **= 0.5
wr = wr / wr.max()

# Create three regularization for the different components
# of magnetization
reg_p = regularization.Sparse(global_mesh, indActive=activeCells, mapping=wires.p)
reg_p.alpha_s = 3
reg_p.cell_weights = (wires.p * wr)
reg_p.mref = np.zeros(3 * nC)

reg_s = regularization.Sparse(global_mesh, indActive=activeCells, mapping=wires.s)
reg_s.alpha_s = 3
reg_s.cell_weights = (wires.s * wr)
reg_s.mref = np.zeros(3 * nC)

reg_t = regularization.Sparse(global_mesh, indActive=activeCells, mapping=wires.t)
reg_t.alpha_s = 3
reg_t.cell_weights = (wires.t * wr)
reg_t.mref = np.zeros(3 * nC)

reg = reg_p + reg_s + reg_t
reg.mref = np.zeros(3 * nC)

# Add directives to the inversion
opt = optimization.ProjectedGNCG(
    maxIter=10, lower=-10, upper=10.0, maxIterLS=20, maxIterCG=20, tolCG=1e-4
)

invProb = inverse_problem.BaseInvProblem(global_misfit, reg, opt)

# A list of directive to control the inverson
betaest = directives.BetaEstimate_ByEig(beta0_ratio=1e1)

# Add sensitivity weights
update_Jacobi = directives.UpdatePreconditioner()
sensitivity_weights = directives.UpdateSensitivityWeights()

# Here is where the norms are applied
# Use pick a threshold parameter empirically based on the distribution of
#  model parameters
IRLS = directives.Update_IRLS(f_min_change=1e-3, max_irls_iterations=2, beta_tol=5e-1)

saveDict = directives.SaveOutputEveryIteration(save_txt=False)
# Pre-conditioner
update_Jacobi = directives.UpdatePreconditioner()

inv = inversion.BaseInversion(
    invProb, directiveList=[IRLS, sensitivity_weights, betaest, update_Jacobi, saveDict],
)

# Run the inversion
mrec_MVIC = inv.run(m0)

vec_xyz = opt.xc

vec_x = vec_xyz[:nC]
vec_y = vec_xyz[nC:2*nC]
vec_z = vec_xyz[2*nC:]

vec = np.c_[vec_x, vec_y, vec_z]

amp_ = np.sum(vec ** 2.0, axis=1) ** 0.5

fig = plt.figure(figsize=(12, 6))
# Plot the result
ax = plt.subplot(3, 2, 1)
global_mesh.plot_slice(inject_global * amp, ind=75, normal="Z", ax=ax, grid=False)
ax.set_title("True")
ax.set_xlim(-100, 100)
ax.set_ylim(-100, 100)
ax.set_aspect("equal")

ax = plt.subplot(3, 2, 2)
global_mesh.plot_slice(inject_global * amp, ind=64, normal="X", ax=ax, grid=False)
ax.set_title("True")
ax.set_xlim(-100, 100)
ax.set_ylim(-100, 100)
ax.set_aspect("equal")

ax = plt.subplot(3, 2, 3)
global_mesh.plot_slice(inject_global * amp_, ind=75, normal="Z", ax=ax, grid=False)
ax.set_title("Recovered")
ax.set_xlim(-100, 100)
ax.set_ylim(-100, 100)
ax.set_aspect("equal")

ax = plt.subplot(3, 2, 4)
global_mesh.plot_slice(inject_global * amp_, ind=64, normal="X", ax=ax, grid=False)
ax.set_title("Recovered")
ax.set_xlim(-100, 100)
ax.set_ylim(-100, 100)

ax = plt.subplot(3, 2, 5)
ax.hist(synthetic_data - np.hstack(invProb.dpred), 100)
ax.set_title("predicted - observed")
plt.show()
