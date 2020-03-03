"""
Reading and Plotting data with DC.IO class
==========================================

The DC.IO class is a convenient way to handle DC data and
carry inversions within a same class. It also has several plotting utils
such as pseudosections. We show here an example of plotting DC data based
on a demonstration dataset.
"""

import numpy as np
import pandas as pd
import shutil
import os
import matplotlib.pyplot as plt
from SimPEG.electromagnetics.static import resistivity as DC
from SimPEG import Report
from SimPEG.utils.io_utils import download

###############################################################################
# Download an example DC data csv file
# ------------------------------------
#

# file origina and name
url = "https://storage.googleapis.com/simpeg/examples/dc_data.csv"
fname = download(url, folder='./test_url', overwrite=True)

# read csv using pandas
df = pd.read_csv(fname)
# header for ABMN locations
header_loc = ['Spa.'+str(i+1) for i in range(4)]
# Apparent resistivity
header_apprho = df.keys()[6]

###############################################################################
#
# Convert file to DC.IO object
# ----------------------------
#

# Number of the data
ndata = df[header_loc[0]].values.size
# ABMN locations
a = np.c_[df[header_loc[0]].values, np.zeros(ndata)]
b = np.c_[df[header_loc[1]].values, np.zeros(ndata)]
m = np.c_[df[header_loc[2]].values, np.zeros(ndata)]
n = np.c_[df[header_loc[3]].values, np.zeros(ndata)]
# Apparent resistivity
apprho = df[header_apprho].values

# Create DC.IO survey Object object
IO = DC.IO()
# Generate DC survey using IO object
dc_survey = IO.from_ambn_locations_to_survey(
    a, b, m, n,
    survey_type='dipole-dipole',
    data_dc=apprho,
    data_dc_type='apparent_resistivity'
)

###############################################################################
#
# Plot
# ----
#
fig, ax = plt.subplots(1, 1, figsize=(10, 3))
IO.plotPseudoSection(
    data_type='apparent_resistivity',
    scale='linear',
    clim=(0, 1000),
    ncontour=3,
    ax=ax
)
plt.show()

# clean up
shutil.rmtree(os.path.expanduser('./test_url'))

###############################################################################
# Print the version of SimPEG and dependencies
# --------------------------------------------
#

Report()

###############################################################################
# Moving Forward
# --------------
#
# If you have suggestions for improving this example, please create a `pull request on the example in SimPEG <https://github.com/simpeg/simpeg/blob/master/examples/06-dc/read_plot_DC_data_with_IO_class.py>`_
#
# You might try:
#    - changing the contour levels
#    - try with you own dataset
#    - create a mask for negative apparent resistivities
#    - ...
