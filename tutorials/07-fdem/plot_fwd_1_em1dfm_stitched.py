"""
Stitched 1D Forward Simulation
==============================

Here we use the module *SimPEG.electromagnetics.frequency_domain* to predict
frequency domain data for a set of "stitched" 1D soundings. That is, the data
for each source is predicted for a separate, user-defined 1D model.
In this tutorial, we focus on the following:

    - Defining receivers, sources and the survey for the stitched 1D case
    - Constructing a stitched conductivity model composed of many 1D vertical conductivity profiles
    - Plotting the stitched model
    - Simulating the frequency-domain electromagnetic (FDEM) data

For each sounding, we compute predicted data for a vertical magnetic dipole source
located 30 m above the Earth's surface. The receiver is offset
10 m horizontally from the source.


"""

#####################################################
# Import Modules
# --------------
#

import numpy as np
from scipy.spatial import Delaunay
import os
import matplotlib as mpl
from matplotlib import pyplot as plt
from discretize import TensorMesh
from pymatsolver import PardisoSolver

from SimPEG import maps
from SimPEG.utils import mkvc, ndgrid
import SimPEG.electromagnetics.frequency_domain as fdem
from SimPEG.electromagnetics.utils.em1d_utils import (
    get_vertical_discretization, set_mesh_1d, Stitched1DModel
)

plt.rcParams.update({'font.size': 16})
write_output = False

# sphinx_gallery_thumbnail_number = 1

#####################################################################
# Create Survey
# -------------
#
# Here we demonstrate a general way to define receivers, sources and the survey.
# For this tutorial, we define three lines of 1D soundings along the
# Easting direction. However, there is no restriction on the spacing and position
# of each sounding.
#

nx = 11
ny = 3
x = np.arange(nx)*50
y = np.arange(ny)*100
z = np.array([30.])

xyz = ndgrid(x, y, z)
np.random.seed(1)
xyz[:,1] += np.random.randn(nx*ny) * 5
n_sounding = xyz.shape[0]
source_locations = xyz  # xyz locations for the centre of the loop
moment = 1

receiver_locations = xyz.copy()         # xyz locations for the receivers
source_receiver_offset = 10.
receiver_locations[:,0] = xyz[:,0] + source_receiver_offset
receiver_orientation = "z"              # "x", "y" or "z"
data_type = "ppm"                       # "secondary", "total" or "ppm"


frequencies = np.array([25., 100., 382, 1822, 7970, 35920], dtype=float)

# For each sounding, we define the source and the associated receivers.
source_list = []
for ii in range(0, n_sounding):

    # Source and receiver locations
    source_location = mkvc(source_locations[ii, :])
    receiver_location = mkvc(receiver_locations[ii, :])

    # Define receiver list for source ii
    receiver_list = []
    receiver_list.append(
        fdem.receivers.PointMagneticFieldSecondary(
            receiver_location, orientation=receiver_orientation,
            data_type=data_type, component="both"
        )
    )

    # Define source ii at frequency jj
    for freq in frequencies:
        source_list.append(
            fdem.sources.MagDipole(
                receiver_list=receiver_list, frequency=freq, location=source_location,
                orientation="z", moment=moment, i_sounding=ii
            )
        )

fig, ax = plt.subplots(1,1, figsize=(5,5))
ax.plot(xyz[:,0], xyz[:,1], '.')
ax.set_aspect(1)
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_title("Sounding locations")

# Survey
survey = fdem.survey.Survey(source_list)

###############################################
# Constructing a stitched model
# --------------------------------
#
# A stitched model is composed of many 1D conductivity vertical profiles.
# Here, we create each 1D conductivity profile then stack all of them to generate
# the stitched model.


# line number
line = (np.arange(ny).repeat(nx)).astype(float)
# time stamp
time_stamp = np.arange(n_sounding).astype(float)
# topography
topography = np.c_[xyz[:,:2], np.zeros(n_sounding)]
# vertical cell widths
hz = 10*np.ones(40)

# A function for generating a wedge layer
def get_y(x):
    y = 30/500 * x + 70.
    return y
# Conductivity values for each unit
background_conductivity = 1./50.
layer_conductivity = 1./10

# Define a 1D vertical mesh
mesh_1d = set_mesh_1d(hz)
# Generate a stitched 1D model
n_layer = hz.size
conductivity = np.zeros((n_sounding, n_layer), dtype=float)

for i_sounding in range(n_sounding):
    y = get_y(xyz[i_sounding, 0])
    layer_ind = np.logical_and(mesh_1d.vectorCCx>50., mesh_1d.vectorCCx<y)
    conductivity_1d = np.ones(n_layer, dtype=float) * background_conductivity
    conductivity_1d[layer_ind] = layer_conductivity
    conductivity[i_sounding,:]=conductivity_1d

# Note: oder of the conductivity model is vertical first then lateral
stitched_conductivity_model = conductivity.flatten()

# Generate a Stitched1DModel object for plotting
model_plot = Stitched1DModel(
    hz=hz,
    line=line,
    time_stamp=time_stamp,
    topography=topography,
    physical_property=1./stitched_conductivity_model
)

_, ax, cb = model_plot.plot_section(cmap='turbo', aspect=0.5, dx=20, i_line=2, clim=(8, 100))
cb.set_label("Resistivity ($\Omega$m)")


#######################################################################
# Define the Mapping, Forward Simulation and Predict Data
# -------------------------------------------------------
#
# Here we define the simulation and predict the FDEM data.
# The simulation requires the user define the survey, the layer thicknesses
# and a mapping from the model to the conductivities.
#
# When using the *SimPEG.electromagnetics.frequency_domain_1d* module, predicted
# data are organized by source (sounding), then by receiver, then by frequency.
#

# Model and mapping. Here the model is defined by the log-conductivity.
stitched_model = np.log(stitched_conductivity_model)
mapping = maps.ExpMap(nP=len(stitched_model))

# Thicknesses of the layers from the surface to the top of the last layer
# The last one is infinite, so no need to define the thickness
thicknesses = hz[:-1]

# Define the simulation
simulation = fdem.Simulation1DLayeredStitched(
    survey=survey, thicknesses=thicknesses, sigmaMap=mapping,
    parallel=False, solver=PardisoSolver
)

# Predict data
dpred = simulation.dpred(stitched_model)


#######################################################################
# Plotting Results
# -------------------------------------------------
#

d_real = dpred[0::2]
d_imag = dpred[1::2]

d_real = np.reshape(d_real, (n_sounding, len(frequencies))).T
d_imag = np.reshape(d_imag, (n_sounding, len(frequencies))).T

fig, ax = plt.subplots(1,1, figsize = (7, 7))

ax.loglog(frequencies, np.abs(d_real[:, 0]), 'k-', lw=2)
ax.loglog(frequencies, np.abs(d_imag[0:len(frequencies):, 0]), 'k--', lw=2)
for ii in range(1, n_sounding):
    ax.loglog(frequencies, np.abs(d_real[:, ii]), '-', lw=2)
    ax.loglog(frequencies, np.abs(d_imag[0:len(frequencies):, ii]), '--', lw=2)

ax.grid(True)
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("|Hs/Hp| (ppm)")
ax.set_title("Secondary Magnetic Field")
ax.legend(["Real", "Imaginary"])

#######################################################################
# Write Output (Optional)
# -----------------------
#

if write_output:
    import pandas as pd
    import tarfile
    import os.path
    dir_path = os.path.dirname(__file__).split(os.path.sep)
    dir_path.extend(["outputs"])
    dir_path = os.path.sep.join(dir_path) + os.path.sep

    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

    np.random.seed(1)
    noise = 0.03*np.abs(dpred)*np.random.rand(len(dpred))
    dpred += noise
    fname = dir_path + "em1dfm_stitched_data.csv"

    DPRED = np.reshape(dpred, (n_sounding, len(frequencies), 2))
    data_header = []
    data_table = np.zeros((n_sounding, int(len(frequencies)*2)), dtype=float)
    i_count = 0
    for i_freq, freq in enumerate(frequencies):
        for i_comp, comp in enumerate(['R', 'I']):
            data_table[:,i_count] = DPRED[:,i_freq,i_comp]
            header = 'Hz{}{}'.format(int(freq), comp)
            data_header.append(header)
            i_count += 1
    sounding_number = np.arange(n_sounding)
    data = np.c_[sounding_number, line, source_locations, topography[:,2], data_table]
    header = ['SOUNDINGNUMBER', 'LINENO', 'X', 'Y', 'Z', 'ELEVATION'] + data_header
    df = pd.DataFrame(data=data, columns=header)
    df.to_csv(fname, index=False)
    def make_tarfile(output_filename, source_dir):
        with tarfile.open(output_filename, "w:gz") as tar:
            tar.add(source_dir, arcname=os.path.basename(source_dir))
    fname_zip = dir_path + 'em1dfm_stitched_fwd.tar.gz'
    make_tarfile(fname_zip, dir_path)    