"""
Stitched 1D Frequency-Domain Inversion
======================================

Here we use the module *SimPEG.electromangetics.frequency_domain_1d* to perform
a stitched 1D inversion on a 3D FDEM dataset. That is, we recover a local 1D
conductivity model for each sounding. In this tutorial, we focus on the following:

    - Defining receivers, sources and the survey for the stitched 1D case
    - Implementing a regularization that connects nearby 1D models
    - Recovering a models for each sounding that define the vertical conductivity profile
    - Constructing a 2D/3D mesh, then interpolating the set of 1D models onto the mesh

For each sounding, the survey geometry consisted of a vertical magnetic dipole source
located 30 m above the Earth's surface. The receiver was offset
10 m horizontally from the source.


"""

#####################################################
# Import Modules
# --------------
#

import numpy as np
from scipy.spatial import cKDTree, Delaunay
import os, tarfile
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from discretize import TensorMesh
from pymatsolver import PardisoSolver

from SimPEG.utils import mkvc
from SimPEG import (
    maps, data, data_misfit, inverse_problem, regularization, optimization,
    directives, inversion, utils
    )

from SimPEG.utils import mkvc
import SimPEG.electromagnetics.frequency_domain_1d as em1d
from SimPEG.regularization import LaterallyConstrained
from SimPEG.electromagnetics.utils.em1d_utils import get_2d_mesh, plot_layer, get_vertical_discretization_frequency

save_file = True

plt.rcParams.update({'font.size': 16, 'lines.linewidth': 2, 'lines.markersize':8})

# sphinx_gallery_thumbnail_number = 4


#############################################
# Download Test Data File
# -----------------------
#
# Here we provide the file path to the data we plan on inverting.
# The path to the data file is stored as a
# tar-file on our google cloud bucket:
# "https://storage.googleapis.com/simpeg/doc-assets/em1dfm_stitched.tar.gz"
#

# storage bucket where we have the data
data_source = "https://storage.googleapis.com/simpeg/doc-assets/em1dfm_stitched.tar.gz"

# download the data
downloaded_data = utils.download(data_source, overwrite=True)

# unzip the tarfile
tar = tarfile.open(downloaded_data, "r")
tar.extractall()
tar.close()

# path to the directory containing our data
dir_path = downloaded_data.split(".")[0] + os.path.sep

# files to work with
data_filename = dir_path + "em1dfm_stitched_data.txt"

#############################################
# Load Data and Plot
# ------------------
#

# Load field data
#dobs = np.loadtxt(str(data_filename))
dobs = np.loadtxt(str(data_filename), skiprows=1)

source_locations = np.unique(dobs[:, 0:3], axis=0)
n_sounding = np.shape(source_locations)[0]

frequencies = np.unique(dobs[:, 3])
n_freq = len(frequencies)

d_real = np.reshape(dobs[:, 4], (n_sounding, n_freq))
d_imag = np.reshape(dobs[:, 5], (n_sounding, n_freq))

fig, ax = plt.subplots(1,1, figsize = (7, 7))

ax.loglog(frequencies, np.abs(d_real[1, :]), 'k-', lw=2)
ax.loglog(frequencies, np.abs(d_imag[1, :]), 'k--', lw=2)
for ii in range(1, n_sounding):
    ax.loglog(frequencies, np.abs(d_real[ii, :]), '-', lw=2)
    ax.loglog(frequencies, np.abs(d_imag[ii, :]), '--', lw=2)

ax.grid(True)
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("|Hs/Hp| (ppm)")
ax.set_title("Observed Data")
ax.legend(["Real", "Imaginary"])

plt.show()


######################################################
# Define Survey
# -------------
#
# Here we define the receivers, sources and the survey needed to invert the data.
# The survey consisted of a line of equally spaced 1D soundings along the
# Easting direction. For each sounding, the survey geometry consisted of a
# vertical magnetic dipole source located 30 m above the Earth's surface.
# The receiver was offset 10 m horizontally from the source. The data were
# secondary field data in ppm.

moment_amplitude = 1.

receiver_locations = np.c_[source_locations[:, 0]+10., source_locations[:, 1:]]
receiver_orientation = "z"  # "x", "y" or "z"
field_type = "ppm"          # "secondary", "total" or "ppm"

source_list = []

for ii in range(0, n_sounding):

    source_location = mkvc(source_locations[ii, :])
    receiver_location = mkvc(receiver_locations[ii, :])

    receiver_list = []

    receiver_list.append(
        em1d.receivers.PointReceiver(
            receiver_location, frequencies, orientation=receiver_orientation,
            field_type=field_type, component="real"
        )
    )
    receiver_list.append(
        em1d.receivers.PointReceiver(
            receiver_location, frequencies, orientation=receiver_orientation,
            field_type=field_type, component="imag"
        )
    )

    source_list.append(
        em1d.sources.MagneticDipoleSource(
            receiver_list=receiver_list, location=source_location, orientation="z",
            moment_amplitude=moment_amplitude
        )
    )

# Survey
survey = em1d.survey.EM1DSurveyFD(source_list)


###############################################
# Assign Uncertainties and Define Data
# ------------------------------------
#
# Here is where we define the data that are being inverted and their uncertainties.
# A data object is used to define the survey, the observation values and the uncertainties.
#

# Define uncertainties for the real and imaginary components separately
unc_real = 0.1*np.abs(d_real)*np.ones(np.shape(d_real))
unc_imag = 0.1*np.abs(d_imag)*np.ones(np.shape(d_imag))

# Define the observed data and associated uncertainties as a vector. Data
# should be organized by source (sounding), then by receiver, then by frequency.
dobs = mkvc(np.c_[d_real, d_imag].T)
uncertainties = mkvc(np.c_[unc_real, unc_imag].T)

# Define the data object
data_object = data.Data(survey, dobs=dobs, standard_deviation=uncertainties)


#######################################################
# Define Layer Thicknesses Used for All Soundings
# -----------------------------------------------
# 
# Although separate 1D models are recovered for each sounding, the number of
# layers and the thicknesses is the same for each sounding.
# For a background conductivity and a set of frequencies, we can determine the
# the optimum layer thicknesses for a set number of layers. Note that when defining
# the thicknesses, it is the number of layers minus one.
#

n_layer = 30
thicknesses = get_vertical_discretization_frequency(
    frequencies, sigma_background=0.1, n_layer=n_layer-1
)


######################################################
# Define a Mapping and a Starting/Reference Model
# -----------------------------------------------
# 
# When defining a starting or reference model, it is important to realize that
# the total number of conductivity required is the number of layers times the
# number of soundings. To keep the tutorial simple, we will invert for the
# log-conductivity. Where *mi* is a 1D array  representing the 1D conductivity
# model for sounding *i*, the 1D array containing all 1D conductivity models is
# organized as [m1,m2,m3,...].
# 

n_param = n_layer*n_sounding  # Number of model parameters

# Define the conductivities for all layers for all soundings into a 1D array.
conductivity = np.ones(n_param) * 0.1

# Define the mapping between the model and the conductivitys
mapping = maps.ExpMap(nP=n_param)

# Define the starting model
starting_model = np.log(conductivity)

#######################################################################
# Define the Forward Problem Using the Simulation Class
# -----------------------------------------------------
#

simulation = em1d.simulation.StitchedEM1DFMSimulation(
    survey=survey, thicknesses=thicknesses, sigmaMap=mapping,
    Solver=PardisoSolver
)


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
dmis.W = 1./uncertainties

# Define the regularization (model objective function). Here we use a laterally
# constrained regularization. This regularization connects the 1D models of
# nearby soundings and ensure lateral changes in electrical conductivity are
# sufficiently smooth.
hz = np.r_[thicknesses, thicknesses[-1]]  # We need to include a thickness for bottom layer
mesh_reg = get_2d_mesh(n_sounding, hz)    # Define a modified mesh for stitched 1D regularization
reg_map = maps.IdentityMap(nP=n_param)    # Mapping between the model and regularization
reg = LaterallyConstrained(
    mesh_reg, mapping=reg_map,
    alpha_s = 0.1,
    alpha_x = 1.,
    alpha_y = 1.,
)

# We must set the gradient for the smoothness portion of the regularization
xy = source_locations[:, [0, 1]]
reg.get_grad_horizontal(xy, hz, dim=2, use_cell_weights=True)

# The laterally constrained case does support sparse norm inversion
ps, px, py = 1, 1, 1
reg.norms = np.c_[ps, px, py, 0]

# Define starting model
reg.mref = starting_model

# Define how the optimization problem is solved. Here we will use an inexact
# Gauss-Newton approach that employs the conjugate gradient solver.
opt = optimization.InexactGaussNewton(maxIter = 40, maxIterCG=20)

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

# Defining a starting value for the trade-off parameter (beta) between the data
# misfit and the regularization.
starting_beta = directives.BetaEstimate_ByEig(beta0_ratio=10)

# Update the preconditionner
update_jacobi = directives.UpdatePreconditioner()

# Options for outputting recovered models and predicted data for each beta.
save_iteration = directives.SaveOutputEveryIteration(save_txt=False)

# Directives of the IRLS
update_IRLS = directives.Update_IRLS(
    max_irls_iterations=20, minGNiter=1,
    fix_Jmatrix=True,
    f_min_change = 1e-3,
    coolingRate=3
)

# The directives are defined as a list.
directives_list = [
    starting_beta,
    save_iteration,
    update_IRLS,
    update_jacobi
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

# Define the least-squares model (if relevant)
l2_model = inv_prob.l2model

#######################################################################
# Interpolate Local Models to 2D/3D Mesh
# --------------------------------------
#
# Now that we have recovered a vector containing the 1D log-conductivities for
# each sounding, we would like to plot recovered models on a 2D or 3D mesh.
# Here, we demonstrate how to interpolate the local 1D models to a global mesh.
#

# Define the global 2D or 3D mesh on which you would like to interpolate your model.
x = source_locations[:, 0]
dx = 50.                                      # horizontal cell width
ncx = int(np.ceil((np.max(x)-np.min(x))/dx))  # number of horizontal cells
hx = np.ones(ncx) * dx                        # horizontal cell widths
hz = 10*np.ones(40)                           # vertical cell widths
mesh2D = TensorMesh([hx, hz], x0='0N')

# Define the locations which correspond to the model values in the recovered model.
# Recall that the array containing these values is organized by sounding, then
# layer from the top layer down. 
z = np.r_[thicknesses, thicknesses[-1]]
z = -(np.cumsum(z) - z/2.)
x, z = np.meshgrid(x, z)
xz = np.c_[mkvc(x), mkvc(z)]

# Use nearest neighbour to interpolate from vector of 1D soundings to the
# global mesh.
tree = cKDTree(xz)
_, ind = tree.query(mesh2D.cell_centers)

model2d_L2 = l2_model[ind]
model2d_sparse = recovered_model[ind]

# Convert from log-conductivity to conductivity
conductivity_L2 = np.exp(model2d_L2)
conductivity_sparse = np.exp(model2d_sparse)

#######################################################################
# Plot the True and Recovered Models
# ----------------------------------
#

# Defining the true model
def PolygonInd(mesh, pts):
    hull = Delaunay(pts)
    inds = hull.find_simplex(mesh.cell_centers)>=0
    return inds

background_conductivity = 0.1
overburden_conductivity = 0.025
slope_conductivity = 0.4

conductivity_true = np.ones(mesh2D.nC) * background_conductivity

layer_ind = mesh2D.cell_centers[:, -1] > -30.
conductivity_true[layer_ind] = overburden_conductivity

x0 = np.r_[0., -30.]
x1 = np.r_[np.max(source_locations[:, 0]), -30.]
x2 = np.r_[np.max(source_locations[:, 0]), -130.]
x3 = np.r_[0., -50.]
pts = np.vstack((x0, x1, x2, x3, x0))
poly_inds = PolygonInd(mesh2D, pts)
conductivity_true[poly_inds] = slope_conductivity

# Plot
title_list = ['True', 'L2', 'Sparse']
models_list = [conductivity_true, conductivity_L2, conductivity_sparse]
for ii, mod in enumerate(models_list):

    fig = plt.figure(figsize=(9, 3))
    ax1 = fig.add_axes([0.15, 0.12, 0.65, 0.78])
    log_mod = np.log10(mod)

    mesh2D.plotImage(
        log_mod, ax=ax1, grid=False,
        clim=(np.log10(conductivity_true.min()), np.log10(conductivity_true.max())),
        pcolorOpts={"cmap": "viridis"},
    )
    ax1.set_ylim(mesh2D.vectorNy.min(), mesh2D.vectorNy.max())

    ax1.set_title("{} Conductivity Model".format(title_list[ii]))
    ax1.set_xlabel("x (m)")
    ax1.set_ylabel("depth (m)")

    ax2 = fig.add_axes([0.85, 0.12, 0.05, 0.78])
    norm = mpl.colors.Normalize(
        vmin=np.log10(conductivity_true.min()), vmax=np.log10(conductivity_true.max())
    )
    cbar = mpl.colorbar.ColorbarBase(
        ax2, norm=norm, cmap=mpl.cm.viridis, orientation="vertical", format="$10^{%.1f}$"
    )
    cbar.set_label("Conductivity [S/m]", rotation=270, labelpad=15, size=12)

#############################################################
# Plot Predicted vs. Observed Data
# --------------------------------
#

x = source_locations[:, 0]

dpred_l2 = simulation.dpred(l2_model)
dpred = simulation.dpred(recovered_model)

data_list = [dobs, dpred_l2, dpred]
color_list = ['k', 'b', 'r']

fig = plt.figure(figsize = (12, 6))
ax1 = fig.add_axes([0.1, 0.1, 0.37, 0.8])
ax2 = fig.add_axes([0.6, 0.1, 0.37, 0.8])

for ii in range(0, len(data_list)):
    
    d_plotting = np.reshape(data_list[ii], (2*n_sounding, len(frequencies)))

    d_real = d_plotting[0::2, :]
    d_imag = d_plotting[1::2, :]
    
    ax1.semilogy(x, np.abs(d_real), color_list[ii], lw=1)
    ax2.semilogy(x, np.abs(d_imag), color_list[ii], lw=1)

ax1.set_xlabel("Sounding Location (m)")
ax1.set_ylabel("Re[H] (ppm)")
ax1.set_title("Real Component")
ax1.legend(["True Model", "L2 Model", "Sparse Model"])
leg = ax1.get_legend()
for ii in range(0, 3):
    leg.legendHandles[ii].set_color(color_list[ii])

ax2.set_xlabel("Sounding Location (m)")
ax2.set_ylabel("Im[H] (ppm)")
ax2.set_title("Imaginary Component")

