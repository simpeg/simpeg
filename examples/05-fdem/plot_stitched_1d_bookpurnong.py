
"""
Inversion of RESOLVE data acquired at Bookpurnong, Austrailia
=============================================================

XXX

"""

import numpy as np
import os
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from discretize import TensorMesh
from pymatsolver import PardisoSolver

import SimPEG
from SimPEG.utils import mkvc
from SimPEG import (
    maps, data, data_misfit, inverse_problem, regularization, optimization,
    directives, inversion, utils
    )

from SimPEG.utils import mkvc
import SimPEG.electromagnetics.frequency_domain_1d as em1d
from SimPEG.regularization import LaterallyConstrained
from SimPEG.electromagnetics.utils.em1d_utils import get_2d_mesh, plot_layer, get_vertical_discretization_frequency

#####################################################################
# Load data
# -------------
#
#

import h5py
import tarfile
import os
import shutil
def download_and_unzip_data(
    url = "https://storage.googleapis.com/simpeg/bookpurnong/bookpurnong_inversion.tar.gz"
):
    """
    Download the data from the storage bucket, unzip the tar file, return
    the directory where the data are
    """
    # download the data
    downloads = utils.download(url)

    # directory where the downloaded files are
    directory = downloads.split(".")[0]

    # unzip the tarfile
    tar = tarfile.open(downloads, "r")
    tar.extractall()
    tar.close()

    return downloads, directory



# download the data
downloads, directory = download_and_unzip_data()

# Load resolve data
resolve = h5py.File(
    os.path.sep.join([directory, "booky_resolve.hdf5"]), "r"
)
river_path = resolve["river_path"]    # River path
nskip = 1
nSounding = resolve["data"][::nskip, :].shape[0]    # the # of soundings

# Bird height from surface
b_height_resolve = (resolve["src_elevation"])[::nskip]

# fetch the frequencies we are considering
cpi_inds = [0, 2, 6, 8, 10]  # Indices for HCP in-phase
cpq_inds = [1, 3, 7, 9, 11]  # Indices for HCP quadrature
frequency_cp = resolve["frequency_cp"]
xy = (resolve["xy"])[::nskip, :]
line = resolve['line'][::nskip]


data_cpi = resolve["data"][::nskip, cpi_inds].astype(float)
data_cpq = resolve["data"][::nskip, cpq_inds].astype(float)


#####################################################################
# Create Survey
# -------------
#
#

from scipy.constants import mu_0
frequencies = np.array([382, 1822, 7970, 35920, 130100], dtype=float)
n_frequency = frequencies.size
# thicknesses = get_vertical_discretization_frequency(frequencies, sigma_background=1./50)
thicknesses = 1 * 1.1**np.arange(19)
n_layer = thicknesses.size + 1
# survey parameters
rxOffset = 7.86  # tx-rx separation
n_sounding = xy.shape[0]
b_height_resolve = resolve["src_elevation"][::nskip]
topo_resolve = resolve["ground_elevation"][::nskip]
uniq_line = np.unique(line)
x = xy[:,0]
y = xy[:,1]
z = topo_resolve + b_height_resolve
receiver_locations = np.c_[x+rxOffset, y, z]
source_locations = np.c_[x, y, z]
topo = np.c_[x, y, topo_resolve].astype(float)


receiver_orientation = 'z'
field_type = 'ppm'
source_list = []
moment_amplitude = 1.
for ii in range(0, n_sounding):
    source_location = mkvc(source_locations[ii, :])
    receiver_location = mkvc(receiver_locations[ii, :])
    rx  = em1d.receivers.PointReceiver(
        receiver_location, frequencies,
        orientation=receiver_orientation,
        field_type=field_type,
        component="both"
    )
    receiver_list = [rx]
    source_list.append(
        em1d.sources.MagneticDipoleSource(
            receiver_list=receiver_list, location=source_location, orientation="z",
            moment_amplitude=moment_amplitude
        )
    )


# Survey
survey = em1d.survey.EM1DSurveyFD(source_list)

#######################################################################
# Define the Forward Simulation
# ----------------------------------------------

mapping = maps.ExpMap(nP=int(n_sounding*n_layer))

simulation = em1d.simulation.StitchedEM1DFMSimulation(
    survey=survey, thicknesses=thicknesses, sigmaMap=mapping, topo=topo,
    verbose=True, Solver=PardisoSolver, parallel=False
)


m0 = np.ones(mapping.nP) * np.log(1./50)

DOBS = np.empty((simulation.n_sounding, 2, n_frequency))
for i_freq in range(frequency_cp.size):
    DOBS[:,0,i_freq] = data_cpi[:, i_freq]
    DOBS[:,1,i_freq] = data_cpq[:, i_freq]


i_line = 3
ind_line = line == uniq_line[i_line]
fig = plt.figure(figsize=(7, 7))
ax = plt.subplot(111)
out = utils.plot2Ddata(xy, DOBS[:,0,0], scale='linear', contourOpts={'cmap':'jet', 'alpha':1}, ncontour=40, ax=ax)
# ax.plot(xy[:,0], xy[:,1], 'k.', ms=1)
ax.plot(river_path[:,0], river_path[:,1], 'k-')
cb = plt.colorbar(out[0], format="%.1e", ax=ax, fraction=0.05)
cb.set_label("Bz (ppm)")
ax.plot(xy[ind_line,0], xy[ind_line,1], 'k.', ms=3)


i_line = 4
ind_line = line == uniq_line[i_line]
fig = plt.figure(figsize=(10, 5))
for i_freq in range(n_frequency):
    plt.semilogy(xy[ind_line,0], DOBS[ind_line,0,i_freq], 'k')
for i_freq in range(n_frequency):
    plt.semilogy(xy[ind_line,0], DOBS[ind_line,1,i_freq], 'r')


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

dobs = DOBS.flatten()
std = 0.1
floor = 20.
uncertainties = std*abs(dobs)+floor
data_object = data.Data(survey, dobs=dobs, standard_deviation=uncertainties)

hz = np.r_[thicknesses, thicknesses[-1]]
mesh_reg = get_2d_mesh(n_sounding, hz)
# Now we can create the regularization using the 2D mesh
reg = LaterallyConstrained(mesh_reg, mapping=maps.IdentityMap(nP=mesh_reg.nC))


tri = reg.get_grad_horizontal(xy, hz)
fig = plt.figure(figsize=(10, 10))
plt.triplot(xy[:,0], xy[:,1], tri.simplices)
plt.plot(xy[:,0], xy[:,1], '.')
# plt.show()

m0 = np.ones(mesh_reg.nC) * np.log(1./100.)
dmis = data_misfit.L2DataMisfit(simulation=simulation, data=data_object)
dmis.W = 1./uncertainties
regmap = maps.IdentityMap(mesh_reg)

reg = LaterallyConstrained(
    mesh_reg, mapping=maps.IdentityMap(nP=mesh_reg.nC),
    alpha_s = 1e-3,
    alpha_x = 1.,
    alpha_y = 1.,
)
tri = reg.get_grad_horizontal(xy, hz)
opt = optimization.InexactGaussNewton(maxIter = 10)
invProb = inverse_problem.BaseInvProblem(dmis, reg, opt)
beta = directives.BetaSchedule(coolingFactor=2, coolingRate=1)
betaest = directives.BetaEstimate_ByEig(beta0_ratio=1.)
target = directives.TargetMisfit()
inv = inversion.BaseInversion(invProb, directiveList=[beta,betaest,target])
simulation.counter = opt.counter = utils.Counter()
opt.LSshorten = 0.5
opt.remember('xc')
mopt = inv.run(m0)

sigma = mapping * mopt

#######################################################################
# Plotting Results
# -------------------------------------------------
#
#


PRED = invProb.dpred.reshape((simulation.n_sounding, 2, n_frequency))

i_line = 14
ind_line = line == uniq_line[i_line]
fig = plt.figure(figsize=(10, 5))
for i_freq in range(n_frequency):
    plt.semilogy(xy[ind_line,0], DOBS[ind_line,0,i_freq], 'k')
    plt.semilogy(xy[ind_line,0], PRED[ind_line,0,i_freq], 'k.')
for i_freq in range(n_frequency):
    plt.semilogy(xy[ind_line,0], DOBS[ind_line,1,i_freq], 'b')
    plt.semilogy(xy[ind_line,0], PRED[ind_line,1,i_freq], 'r.')
plt.ylabel("Hz (ppm)")


from simpegEM1D import ModelIO
IO = ModelIO(
    hz = hz,
    topography=topo,
    line=line,
    physical_property=1./sigma
)

i_line = 23
IO.plot_section(
    plot_type='pcolor', aspect=10, i_line=i_line, clim=(0.3, 50) ,
    scale='log', cmap='Spectral', dx=50
)




