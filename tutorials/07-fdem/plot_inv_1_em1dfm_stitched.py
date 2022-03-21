"""
Stitched 1D Frequency-Domain Inversion
======================================

Here we use the module *SimPEG.electromagnetics.frequency_domain* to perform
a stitched 1D inversion on a 3D FDEM (frequency-domain electromangetic)dataset. That is, we recover a local 1D
conductivity model for each sounding. In this tutorial, we focus on the following:

    - Defining receivers, sources and the survey for the stitched 1D case
    - Implementing a regularization that connects nearby 1D vertical conductivity profiles
    - Recovering a stitched model composed of a 1D vertical conductivity profile at each sounding location

For each sounding, the survey geometry consisted of a vertical magnetic dipole source
located 30 m above the Earth's surface. The receiver was offset
10 m horizontally from the source.


"""

#####################################################
# Import Modules
# --------------
#

import numpy as np
import pandas as pd
from scipy.spatial import Delaunay
import os
import tarfile
import matplotlib as mpl
from matplotlib import pyplot as plt
from discretize import TensorMesh, SimplexMesh
from pymatsolver import PardisoSolver

from SimPEG.utils import mkvc
from SimPEG import (
    maps, data, data_misfit, inverse_problem, regularization, optimization,
    directives, inversion, utils
    )

import SimPEG.electromagnetics.frequency_domain as fdem
from SimPEG.electromagnetics.utils.em1d_utils import set_mesh_1d, get_vertical_discretization, Stitched1DModel

save_file = False

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
data_source = "https://storage.googleapis.com/simpeg/doc-assets/em1dfm_stitched_fwd.tar.gz"

# download the data
downloaded_data = utils.download(data_source, overwrite=True)

# path to the directory containing our data
dir_path = downloaded_data.split(".")[0] + os.path.sep

if not os.path.exists(dir_path):
    os.mkdir(dir_path)

# unzip the tarfile
tar = tarfile.open(downloaded_data, "r")
tar.extractall(dir_path)
tar.close()

# files to work with
data_filename = dir_path + "em1dfm_stitched_data.csv"

#############################################
# Load Data and Plot
# ------------------
#

# Load field data
df = pd.read_csv(data_filename)

source_locations = df[['X', 'Y', 'Z']].values
n_sounding = np.shape(source_locations)[0]

data_header = np.array(list(key for key in df.columns[6:]))
data_header_real = data_header[::2]
data_header_imag = data_header[1::2]

frequencies = np.array([25, 100, 382, 1822, 7970, 35920])
n_freq = len(frequencies)
fdem_data = df[data_header].values
n_sounding = df.shape[0]

group_line = df.groupby('LINENO')

uniq_line = list(group_line.groups.keys())
data_real_tmp = group_line.get_group(uniq_line[0])[data_header_real].values
data_imag_tmp = group_line.get_group(uniq_line[0])[data_header_imag].values
fig, ax = plt.subplots(1,1,figsize=(10, 5))
x = group_line.get_group(uniq_line[0])['X'].values
_ = ax.semilogy(x, data_real_tmp, 'k')
_ = ax.semilogy(x, data_imag_tmp, 'r')
ax.set_xlabel("Easting (m)")
_ = ax.semilogy(x, data_real_tmp[:,0], 'k', label='real')
_ = ax.semilogy(x, data_imag_tmp[:,0], 'r', label='imag')
ax.legend()
ax.set_title("Observed data")
plt.tight_layout()

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

moment = 1.

receiver_locations = np.c_[source_locations[:, 0]+10., source_locations[:, 1:]]
receiver_orientation = "z"  # "x", "y" or "z"
data_type = "ppm"           # "secondary", "total" or "ppm"

source_list = []

for ii in range(0, n_sounding):

    source_location = mkvc(source_locations[ii, :])
    receiver_location = mkvc(receiver_locations[ii, :])

    receiver_list = []

    receiver_list.append(
        fdem.receivers.PointMagneticFieldSecondary(
            receiver_location, orientation=receiver_orientation,
            data_type=data_type, component="real"
        )
    )
    receiver_list.append(
        fdem.receivers.PointMagneticFieldSecondary(
            receiver_location, orientation=receiver_orientation,
            data_type=data_type, component="imag"
        )
    )

    for freq in frequencies:
        source_list.append(
            fdem.sources.MagDipole(
                receiver_list=receiver_list, frequency=freq, location=source_location,
                orientation="z", moment=moment, i_sounding=ii
            )
        )

# Survey
survey = fdem.Survey(source_list)


###############################################
# Assign Uncertainties and Define Data
# ------------------------------------
#
# Here is where we define the data that are being inverted and their uncertainties.
# A data object is used to define the survey, the observation values and the uncertainties.
#


# Define the observed data and associated uncertainties as a vector. Data
# should be organized by source (sounding), then by frequency, then by receiver.
dobs = fdem_data.reshape((n_sounding, n_freq, 2)).flatten()
uncertainties = 0.05*abs(dobs)

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
thicknesses = get_vertical_discretization(n_layer-1, 3, 1.07)


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

simulation = fdem.Simulation1DLayeredStitched(
    survey=survey, thicknesses=thicknesses, sigmaMap=mapping,
    solver=PardisoSolver
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

tri = Delaunay(source_locations[:,:2])
mesh_radial = SimplexMesh(tri.points, tri.simplices)
mesh_vertical = set_mesh_1d(hz)
mesh_reg = [mesh_radial, mesh_vertical]
n_param = int(mesh_radial.n_nodes * mesh_vertical.nC)
reg_map = maps.IdentityMap(nP=n_param)    # Mapping between the model and regularization
reg = regularization.LaterallyConstrained(
    mesh_reg, mapping=reg_map,
    alpha_s = 1e-10,
    alpha_r = 1.,
    alpha_z = 1.,
)

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

# Defining how to cool the trade-off parameter, beta, through out the inversion
# coolingFactor=2, coolingRate=1 indicate the beta value is decreased with a factor of 2
# for every iteration.

beta_schedule = directives.BetaSchedule(coolingFactor=2, coolingRate=1)

target =directives.TargetMisfit()

# The directives are defined as a list.
directives_list = [
    starting_beta,
    beta_schedule,
    target
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

# Generate a Stitched1DModel object for plotting
line = df['LINENO'].values
topography = df[['X', 'Y', 'ELEVATION']].values
time_stamp = df['SOUNDINGNUMBER'].values
model_plot = Stitched1DModel(
    hz=hz,
    line=line,
    time_stamp=time_stamp,
    topography=topography,
    physical_property=1./np.exp(recovered_model),
)

fig, ax = plt.subplots(1,1, figsize=(10, 8))
_, ax, cb = model_plot.plot_section(i_line=0, aspect=1, dx=20, cmap='turbo', clim=(8, 100), ax=ax)
cb.set_label("Resistivity ($\Omega$m)")
