"""
PF: Gravity: Tiled Inversion Linear
===================================

Invert data in tiles.

"""
import numpy as np
import gc
import matplotlib.pyplot as plt
from discretize.utils import mesh_builder_xyz, refine_tree_xyz, active_from_xyz
from dask.distributed import Client, LocalCluster
from SimPEG import dask
import dask
from dask import config
from SimPEG.utils.drivers import create_tile_meshes, create_nested_mesh
from SimPEG.potential_fields import gravity
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
h = [2, 2, 2]
padDist = np.ones((3, 2)) * 100

topography_refinement = [0, 0, 2]
locations_refinement = [5, 5, 5]
max_distance = 25

###############################################################################
# Setup
# -----
#
# Define the survey and model parameters
#
# Create a global survey and mesh and simulate some data
#
#

# @dask.delayed
def create_tile(locations, obs, uncert, global_mesh, global_active, tile_id, workers=None):
    receivers = gravity.receivers.Point(locations)
    srcField = gravity.sources.SourceField([receivers])
    local_survey = gravity.survey.Survey(srcField)

    # Create tile map between global and local
    local_mesh = create_nested_mesh(
        receivers.locations, global_mesh,
    )
    local_map = maps.TileMap(global_mesh, global_active, local_mesh)

    # Create the local misfit
    simulation = gravity.simulation.Simulation3DIntegral(
        survey=local_survey,
        mesh=local_mesh,
        rhoMap=maps.IdentityMap(nP=int(local_map.local_active.sum())),
        actInd=local_map.local_active,
        sensitivity_path=f"Inversion\Tile{tile_id}.zarr",
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

# Create and array of observation points
xr = np.linspace(-100.0, 100.0, 20)
yr = np.linspace(-100.0, 100.0, 20)
X, Y = np.meshgrid(xr, yr)

# Move the observation points 5m above the topo
Z = np.zeros_like(X) - 0.1 #-np.exp((X ** 2 + Y ** 2) / 75 ** 2)

# Create a topo array
topo = np.c_[utils.mkvc(X.T), utils.mkvc(Y.T), utils.mkvc(Z.T)]

# Create station locations drapped 0.1 m above topo
rxLoc = np.c_[utils.mkvc(X.T), utils.mkvc(Y.T), utils.mkvc(Z.T) + 0.1]

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

##########################################################################
# Synthetic data simulation
# -------------------------
#
# We can now create a density model and generate data
# Here a simple block in half-space
# Get the indices of the magnetized block
model = np.zeros(global_mesh.nC)
ind = utils.ModelBuilder.getIndicesBlock(
    np.r_[-10, -10, -30], np.r_[10, 10, -10], global_mesh.gridCC,
)[0]

# Assign density values
model[ind] = 0.3

# Remove air cells
model = model[activeCells]

# Create reduced identity map
idenMap = maps.IdentityMap(nP=int(activeCells.sum()))

# Create a global survey just for simulation of data
receivers = gravity.receivers.Point(rxLoc)
srcField = gravity.sources.SourceField([receivers])
survey = gravity.survey.Survey(srcField)

# Create the forward simulation for the global dataset
simulation = gravity.simulation.Simulation3DIntegral(
    survey=survey, mesh=global_mesh, rhoMap=idenMap, actInd=activeCells,
    store_sensitivities="forward_only",
    max_chunk_size=max_chunk_size,
)

# Compute linear forward operator and compute some data
d = simulation.dpred(model).compute()

# Add noise and uncertainties
# We add some random Gaussian noise (1nT)
synthetic_data = d + np.random.randn(len(d)) * 2e-3
wd = np.ones(len(synthetic_data)) * 2e-3  # Assign flat uncertainties


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
indices = tile_locations(rxLoc, 4, method="kmeans")
ct = time()
local_misfits = []
for ii, local_index in enumerate(indices):
    locations = rxLoc[local_index, :]

    # sc_locs, sc_mesh = client.scatter([locations, global_mesh])

    local_misfits += [
        create_tile(
            locations,  synthetic_data[local_index],
            wd[local_index], global_mesh, activeCells, ii, workers=None
        )
    ]


# local_misfits = client.gather(local_misfits)
# tile_time = time() - ct
# print(f"Tile creation {tile_time}")


global_misfit = objective_function.ComboObjectiveFunction(
        local_misfits
)

# Trigger sensitivities
for local in local_misfits:
    local.simulation.Jmatrix
    local.simulation.Jmatrix

# Plot the model on different meshes
fig = plt.figure(figsize=(12, 6))
c_code = ['r', 'g', 'b', 'm']
for ii, local_misfit in enumerate(global_misfit.objfcts):

    local_mesh = local_misfit.simulation.mesh
    local_map = local_misfit.model_map

    inject_local = maps.InjectActiveCells(local_mesh, local_map.local_active, np.nan)

    ax = plt.subplot(2, 3, ii + 1)
    local_mesh.plot_slice(
        inject_local * (local_map * model), index=200, normal="Z", ax=ax, grid=True
    )
    sensors = local_misfit.simulation.survey.receiver_locations
    ax.scatter(sensors[:, 0], sensors[:, 1], 10, c=local_misfit.data.dobs)
    ax.set_xlim(-200, 200)
    ax.set_ylim(-200, 200)
    ax.set_aspect("equal")
    ax.set_title(f"Mesh {ii+1}. Active cells {local_map.local_active.sum()}")


# Create active map to go from reduce set to full
inject_global = maps.InjectActiveCells(global_mesh, activeCells, 0)

ax = plt.subplot(2, 3, 6)
global_mesh.plot_slice(inject_global * model, index=200, normal="Z", ax=ax, grid=True)
ax.scatter(rxLoc[:, 0], rxLoc[:, 1], 10, c=synthetic_data)
ax.set_title(f"Global Mesh. Active cells {activeCells.sum()}")
ax.set_xlim(-200, 200)
ax.set_ylim(-200, 200)
ax.set_aspect("equal")
plt.show()


#####################################################
# Invert on the global mesh
#
#
#
#
#

# Create a regularization on the global mesh
reg = regularization.Sparse(
    global_mesh, indActive=activeCells, mapping=idenMap
)

m0 = np.ones(int(activeCells.sum())) * 1e-4  # Starting model

# Add directives to the inversion
opt = optimization.ProjectedGNCG(
    maxIter=10, lower=-1.0, upper=1.0, maxIterLS=20, maxIterCG=10, tolCG=1e-16
)
invProb = inverse_problem.BaseInvProblem(global_misfit, reg, opt)
betaest = directives.BetaEstimate_ByEig(beta0_ratio=1e-1, method="ratio")

# Here is where the norms are applied
# Use a threshold parameter empirically based on the distribution of
# model parameters
update_IRLS = directives.Update_IRLS(
    f_min_change=1e-4, max_irls_iterations=0, coolEpsFact=1.5, beta_tol=4.,
)
saveDict = directives.SaveOutputEveryIteration(save_txt=False)
update_Jacobi = directives.UpdatePreconditioner()
sensitivity_weights = directives.UpdateSensitivityWeights(everyIter=False)
inv = inversion.BaseInversion(
    invProb,
    directiveList=[update_IRLS, sensitivity_weights, betaest, update_Jacobi, saveDict],
)


# Run the inversion
mrec = inv.run(m0)
# inv_time = opt.cg_runtime
# global_mesh.write_UBC(
#     f"InvMesh{kk}_{jj}.msh",
#     models={f"inv{kk}_{jj}.mod": inject_global * invProb.model})
fig = plt.figure(figsize=(12, 6))
# Plot the result
ax = plt.subplot(1, 2, 1)
global_mesh.plot_slice(inject_global * model, normal="Y", ax=ax, grid=True)
ax.set_title("True")
ax.set_xlim(-60, 60)
ax.set_ylim(-50, 10)
ax.set_aspect("equal")

ax = plt.subplot(1, 2, 2)
global_mesh.plot_slice(inject_global * mrec, normal="Y", ax=ax, grid=True)
ax.set_title("Recovered")
ax.set_xlim(-60, 60)
ax.set_ylim(-50, 10)
ax.set_aspect("equal")
plt.show()
# f = open(output, "a")
# f.write(
#     "{:1.4e} {:1.4e} {:1.4e} {:1.4e} {:1.4e} {:1.4e} \n".format(
#         prob_size, 2**jj, max_chunk_size, tile_time, sens_time, inv_time
#     )
# )
# f.close()


# np.savetxt('Runtime_CG.txt', np.vstack(inv_time))
# np.savetxt('Runtime_Sens.txt', np.vstack(sens_time))
# np.savetxt('ProblemSize.txt', np.vstack(sens_time))