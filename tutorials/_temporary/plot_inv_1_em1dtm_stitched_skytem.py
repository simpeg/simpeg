"""
Forward Simulation of Stitched SkyTEM Data
==========================================





"""

#####################################################
# Import Modules
# --------------
#

import numpy as np
import tarfile
import matplotlib as mpl
from matplotlib import pyplot as plt
from discretize import TensorMesh
from pymatsolver import PardisoSolver

from SimPEG.utils import mkvc
from SimPEG import (
    maps,
    data,
    data_misfit,
    inverse_problem,
    regularization,
    optimization,
    directives,
    inversion,
    utils,
)

import SimPEG.electromagnetics.time_domain_1d as em1d
from SimPEG.electromagnetics.utils.em1d_utils import (
    get_2d_mesh,
    get_vertical_discretization_time,
)
from SimPEG.electromagnetics.time_domain_1d.known_waveforms import (
    skytem_HM_2015,
    skytem_LM_2015,
)

save_file = True

plt.rcParams.update({"font.size": 16, "lines.linewidth": 2, "lines.markersize": 8})


#############################################
# Define File Names
# -----------------
#
# File paths for assets we are loading. To set up the inversion, we require
# topography and field observations. The true model defined on the whole mesh
# is loaded to compare with the inversion result.
#

# storage bucket where we have the data
data_source = "https://storage.googleapis.com/simpeg/doc-assets/em1dtm_stitched_skytem_data.tar.gz"

# download the data
downloaded_data = utils.download(data_source, overwrite=True)

# unzip the tarfile
tar = tarfile.open(downloaded_data, "r")
tar.extractall()
tar.close()

# filepath to data file
data_filename = downloaded_data.split(".")[0] + ".obs"

#####################################################################
# topography
# -------------
#
#

x = np.linspace(50, 4950, 50)
y = np.zeros_like(x)
z = np.zeros_like(x)
topo = np.c_[x, y, z].astype(float)

n_sounding = len(x)

source_locations = np.c_[x, np.zeros(n_sounding), 30.0 * np.ones(n_sounding)]
source_current = 1.0
source_orientation = "z"
receiver_offset_r = 13.25
receiver_offset_z = 2.0

receiver_locations = np.c_[
    x + receiver_offset_r,
    np.zeros(n_sounding),
    30.0 * np.ones(n_sounding) + receiver_offset_z,
]
receiver_orientation = "z"  # "x", "y" or "z"


#############################################
# Load Data and Plot
# ------------------
#

# Load field data
dobs = np.loadtxt(str(data_filename))
dobs = mkvc(dobs[:, -1])


######################################################
# Create Survey
# -------------
#

wave_HM = skytem_HM_2015()
wave_LM = skytem_LM_2015()
time_HM = wave_HM.time_gate_center[0::2]
time_LM = wave_LM.time_gate_center[0::2]

time_input_currents_HM = wave_HM.current_times[-7:]
input_currents_HM = wave_HM.currents[-7:]
time_input_currents_LM = wave_LM.current_times[-13:]
input_currents_LM = wave_LM.currents[-13:]


source_list = []

for ii in range(0, n_sounding):

    source_location = mkvc(source_locations[ii, :])
    receiver_location = mkvc(receiver_locations[ii, :])
    receiver_list = [
        em1d.receivers.PointReceiver(
            receiver_location,
            times=time_HM,
            times_dual_moment=time_LM,
            orientation=receiver_orientation,
            component="dbdt",
        )
    ]

    #     Sources
    source_list.append(
        em1d.sources.MagneticDipoleSource(
            receiver_list=receiver_list,
            location=source_location,
            moment_amplitude=source_current,
            orientation=source_orientation,
            wave_type="general",
            moment_type="dual",
            time_input_currents=time_input_currents_HM,
            input_currents=input_currents_HM,
            n_pulse=1,
            base_frequency=25.0,
            time_input_currents_dual_moment=time_input_currents_LM,
            input_currents_dual_moment=input_currents_LM,
            base_frequency_dual_moment=210,
        )
    )


# Survey
survey = em1d.survey.EM1DSurveyTD(source_list)


#############################################
# Assign Uncertainties
# --------------------
#
#

uncertainties = 0.1 * np.abs(dobs) * np.ones(np.shape(dobs))


###############################################
# Define Data
# --------------------
#
# Here is where we define the data that are inverted. The data are defined by
# the survey, the observation values and the uncertainties.
#

data_object = data.Data(survey, dobs=dobs, noise_floor=uncertainties)


###############################################
# Defining a Global Mesh
# ----------------------
#

n_layer = 25
thicknesses = get_vertical_discretization_time(
    time_HM, sigma_background=0.1, n_layer=n_layer - 1
)

dx = 100.0
hx = np.ones(n_sounding) * dx
hz = np.r_[thicknesses, thicknesses[-1]]
mesh2D = TensorMesh([hx, np.flipud(hz)], x0="0N")
mesh_soundings = TensorMesh([hz, hx], x0="00")

n_param = n_layer * n_sounding


###############################################
# Defining a Model
# ----------------------
#

conductivity = np.ones(n_param) * 0.1

mapping = maps.ExpMap(nP=n_param)
starting_model = np.log(conductivity)

#######################################################################
# Define the Forward Simulation and Predic Data
# ----------------------------------------------
#


# Simulate response for static conductivity
simulation = em1d.simulation.StitchedEM1DTMSimulation(
    survey=survey,
    thicknesses=thicknesses,
    sigmaMap=mapping,
    topo=topo,
    Solver=PardisoSolver,
)

# simulation = em1d.simulation.StitchedEM1DTMSimulation(
#     survey=survey, thicknesses=thicknesses, sigmaMap=mapping,
#     topo=topo, parallel=True, n_cpu=4, verbose=True, Solver=PardisoSolver
# )


pred = simulation.dpred(starting_model)
fig = plt.figure()
plt.semilogy(-pred, ".", ms=1)
plt.semilogy(-dobs, "x")
plt.show()


########################################################################
# Define Inverse Problem
# ----------------------
#
# The inverse problem is defined by 3 things:
#
#     1) Data Misfit: a measure of how well our recovered model explains the field data
#     2) Regularization: constraints placed on the recovered model and a priori information
#     3) Optimization: the numerical approach used to solve the inverse problem
#
#

# Define the data misfit. Here the data misfit is the L2 norm of the weighted
# residual between the observed data and the data predicted for a given model.
# The weighting is defined by the reciprocal of the uncertainties.
dmis = data_misfit.L2DataMisfit(simulation=simulation, data=data_object)
dmis.W = 1.0 / uncertainties


# Define the regularization (model objective function)
mesh_reg = get_2d_mesh(n_sounding, hz)
reg_map = maps.IdentityMap(mesh_reg)
reg = regularization.LaterallyConstrained(
    mesh_reg,
    mapping=reg_map,
    alpha_s=0.1,
    alpha_x=1.0,
    alpha_y=1.0,
)
xy = utils.ndgrid(x, np.r_[0.0])
reg.get_grad_horizontal(xy, hz, dim=2, use_cell_weights=True)


# reg_map = maps.IdentityMap(nP=mesh_soundings.nC)
# reg = regularization.Sparse(
#     mesh_reg, mapping=reg_map,
# )

ps, px, py = 1, 1, 1
reg.norms = np.c_[ps, px, py, 0]

reg.mref = starting_model
reg.mrefInSmooth = False

# Define how the optimization problem is solved. Here we will use an inexact
# Gauss-Newton approach that employs the conjugate gradient solver.
opt = optimization.InexactGaussNewton(maxIter=40, maxIterCG=20)

# Define the inverse problem
inv_prob = inverse_problem.BaseInvProblem(dmis, reg, opt)


#######################################################################
# Define Inversion Directives
# ---------------------------
#
# Here we define any directiveas that are carried out during the inversion. This
# includes the cooling schedule for the trade-off parameter (beta), stopping
# criteria for the inversion and saving inversion results at each iteration.
#

# Apply and update sensitivity weighting as the model updates
# sensitivity_weights = directives.UpdateSensitivityWeights()

# Reach target misfit for L2 solution, then use IRLS until model stops changing.
# IRLS = directives.Update_IRLS(max_irls_iterations=40, minGNiter=1, f_min_change=1e-5, chifact_start=2)
# IRLS = directives.Update_IRLS(
#    max_irls_iterations=20, minGNiter=1, fix_Jmatrix=True, coolingRate=2,
#    beta_tol=1e-2, f_min_change=1e-5,
#    chifact_start = 1.
# )

# Defining a starting value for the trade-off parameter (beta) between the data
# misfit and the regularization.
starting_beta = directives.BetaEstimate_ByEig(beta0_ratio=10)


beta_schedule = directives.BetaSchedule(coolingFactor=2, coolingRate=2)

# Update the preconditionner
update_Jacobi = directives.UpdatePreconditioner()

# Options for outputting recovered models and predicted data for each beta.
save_iteration = directives.SaveOutputEveryIteration(save_txt=False)


update_IRLS = directives.Update_IRLS(
    max_irls_iterations=20,
    minGNiter=1,
    fix_Jmatrix=True,
    f_min_change=1e-3,
    coolingRate=3,
)

# Updating the preconditionner if it is model dependent.
update_jacobi = directives.UpdatePreconditioner()

# Setting a stopping criteria for the inversion.
target_misfit = directives.TargetMisfit(chifact=1)

# Add sensitivity weights
sensitivity_weights = directives.UpdateSensitivityWeights()

target = directives.TargetMisfit()

# The directives are defined as a list.
directives_list = [
    # sensitivity_weights,
    starting_beta,
    beta_schedule,
    save_iteration,
    # target_misfit,
    update_IRLS,
    # update_jacobi,
]

#####################################################################
# Running the Inversion
# ---------------------
#
# To define the inversion object, we need to define the inversion problem and
# the set of directives. We can then run the inversion.
#

# Here we combine the inverse problem and the set of directives
inv = inversion.BaseInversion(inv_prob, directives_list)

# Run the inversion
recovered_model = inv.run(starting_model)


#######################################################################
# Plotting Results
# -------------------------------------------------
#
#


# True model
from scipy.spatial import Delaunay


def PolygonInd(mesh, pts):
    hull = Delaunay(pts)
    inds = hull.find_simplex(mesh.gridCC) >= 0
    return inds


background_conductivity = 0.1
overburden_conductivity = 0.025
slope_conductivity = 0.4

true_model = np.ones(mesh2D.nC) * background_conductivity

layer_ind = mesh2D.gridCC[:, -1] > -30.0
true_model[layer_ind] = overburden_conductivity


x0 = np.r_[0.0, -30.0]
x1 = np.r_[dx * n_sounding, -30.0]
x2 = np.r_[dx * n_sounding, -130.0]
x3 = np.r_[0.0, -50.0]
pts = np.vstack((x0, x1, x2, x3, x0))
poly_inds = PolygonInd(mesh2D, pts)
true_model[poly_inds] = slope_conductivity

# true_model = true_model.reshape(mesh_soundings.vnC, order='C')
# true_model = np.flipud(true_model)
# true_model = mkvc(true_model)


l2_model = inv_prob.l2model
dpred_l2 = simulation.dpred(l2_model)
l2_model = np.exp(l2_model)
# l2_model = l2_model.reshape((simulation.n_sounding, simulation.n_layer),)
# l2_model = mkvc(l2_model)

dpred = simulation.dpred(recovered_model)
recovered_model = np.exp(recovered_model)
# recovered_model = recovered_model.reshape((simulation.n_sounding, simulation.n_layer))
# recovered_model = mkvc(recovered_model)


mesh_plotting = TensorMesh([hx, np.flipud(hz)], x0="0N")
l2_model = l2_model.reshape(mesh_plotting.vnC, order="C")
l2_model = mkvc(np.fliplr(l2_model))
recovered_model = recovered_model.reshape(mesh_plotting.vnC, order="C")
recovered_model = mkvc(np.fliplr(recovered_model))


models_list = [true_model, l2_model, recovered_model]


for ii, mod in enumerate(models_list):

    fig = plt.figure(figsize=(9, 3))
    ax1 = fig.add_axes([0.1, 0.12, 0.73, 0.78])
    log_mod = np.log10(mod)

    mesh_plotting.plotImage(
        log_mod,
        ax=ax1,
        grid=False,
        clim=(np.log10(true_model.min()), np.log10(true_model.max())),
        #        clim=(np.log10(0.1), np.log10(1)),
        pcolorOpts={"cmap": "viridis"},
    )
    ax1.set_ylim(mesh_plotting.vectorNy.min(), mesh_plotting.vectorNy.max())

    ax1.set_title("Conductivity Model")
    ax1.set_xlabel("x (m)")
    ax1.set_ylabel("depth (m)")

    ax2 = fig.add_axes([0.85, 0.12, 0.05, 0.78])
    norm = mpl.colors.Normalize(
        vmin=np.log10(true_model.min()),
        vmax=np.log10(true_model.max())
        #        vmin=np.log10(0.1), vmax=np.log10(1)
    )
    cbar = mpl.colorbar.ColorbarBase(
        ax2,
        norm=norm,
        cmap=mpl.cm.viridis,
        orientation="vertical",
        format="$10^{%.1f}$",
    )
    cbar.set_label("Conductivity [S/m]", rotation=270, labelpad=15, size=12)


data_list = [dobs, dpred_l2, dpred]
color_list = ["k", "b", "r"]

fig, ax = plt.subplots(1, 1, figsize=(7, 7))
n_time = time_HM.size + time_LM.size
for ii in range(0, len(data_list)):
    d = np.reshape(data_list[ii], (n_sounding, n_time))
    ax.semilogy(x, np.abs(d), color_list[ii], lw=1)

ax.set_xlabel("Times (s)")
ax.set_ylabel("|dBdt| (T/s)")
plt.show()
