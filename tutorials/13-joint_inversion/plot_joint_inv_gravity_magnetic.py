#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Cross-gradient Joint Inversion of Gravity and Magnetic Anomaly Data
===============================================

Here we simultaneously invert gravity and magentic data using cross-gradient 
constraint. The recovered density and susceptibility models are supposed to have
structural similarity. For this tutorial, we focus on the following:
    
    - Defining the survey from xyz formatted data
    - Generating a mesh based on survey geometry
    - Including surface topography
    - Defining the inverse problem via combmaps (2 data misfit terms, 
        3 regularization terms including the coupling term, optimization)
    - Specifying directives for the inversion
    - Plotting the recovered model and data misfit


Although we consider gravity and magnetic anomaly data in this tutorial, 
the same approach can be used to invert gradiometry and other types of geophysical data.



"""

#########################################################################
# Import modules
# --------------
#

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import tarfile

from discretize import TensorMesh

from SimPEG.utils import plot2Ddata, surface2ind_topo
from SimPEG.potential_fields import gravity
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


#############################################
# Define File Names
# -----------------
#
# File paths for assets we are loading. To set up the inversion, we require
# topography and field observations. The true model defined on the whole mesh
# is loaded to compare with the inversion result. These files are stored as a
# tar-file on our google cloud bucket:
# "https://storage.googleapis.com/simpeg/doc-assets/gravity.tar.gz"

# storage bucket where we have the data
data_source = "https://storage.googleapis.com/simpeg/doc-assets/gravity.tar.gz"

# download the data
downloaded_data = utils.download(data_source, overwrite=True)

# unzip the tarfile
tar = tarfile.open(downloaded_data, "r")
tar.extractall()
tar.close()

# path to the directory containing our data
dir_path = downloaded_data.split(".")[0] + os.path.sep

# files to work with
topo_filename = dir_path + "gravity_topo.txt"
data_filename = dir_path + "gravity_data.obs"
model_filename = dir_path + "true_model.txt"


#############################################
# Load Data and Plot
# ------------------
#
# Here we load and plot synthetic gravity anomaly data. Topography is generally
# defined as an (N, 3) array. Gravity data is generally defined with 4 columns:
# x, y, z and data.
#
 


# Load topography
xyz_topo = np.loadtxt(str(topo_filename))

# Load field data
dobs = np.loadtxt(str(data_filename))

# Define receiver locations and observed data
receiver_locations = dobs[:, 0:3]
dobs = dobs[:, -1]

# Plot
mpl.rcParams.update({"font.size": 12})
fig = plt.figure(figsize=(7, 5))

ax1 = fig.add_axes([0.1, 0.1, 0.73, 0.85])
plot2Ddata(receiver_locations, dobs, ax=ax1, contourOpts={"cmap": "bwr"})
ax1.set_title("Gravity Anomaly")
ax1.set_xlabel("x (m)")
ax1.set_ylabel("y (m)")

ax2 = fig.add_axes([0.8, 0.1, 0.03, 0.85])
norm = mpl.colors.Normalize(vmin=-np.max(np.abs(dobs)), vmax=np.max(np.abs(dobs)))
cbar = mpl.colorbar.ColorbarBase(
    ax2, norm=norm, orientation="vertical", cmap=mpl.cm.bwr, format="%.1e"
)
cbar.set_label("$mgal$", rotation=270, labelpad=15, size=12)

plt.show()


























