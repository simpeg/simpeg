"""
1D Forward Simulation with Primary and Dual Waveforms
=====================================================

Some TDEM systems use separate waveforms to model the response at early times and
at late times. In the tutorial, we show the user how to define sources
and receivers when there is a 'primary' and a 'dual' waveform.


"""

#####################################################
# Import Modules
# --------------
#

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
mpl.rcParams.update({'font.size':16})

from SimPEG import maps
import SimPEG.electromagnetics.time_domain_1d as em1d
from SimPEG.electromagnetics.time_domain_1d.supporting_functions.waveform_functions import (
    skytem_2015_LM_time_channels, skytem_2015_HM_time_channels
)


#####################################################################
# Define Waveforms
# ----------------
# 
# Here, we define the waveform object(s) required for our forward simulation.
# The 'primary' and 'dual' waveforms are contained within the same waveform object. 
# Here, we show the user a completely general way to define a waveform object
# that has primary and dual waveforms. We also define a SkyTEM 2015 waveform
# object using our known set of waveforms.
#

# General waveform (quarter-sine ramp off)
def custom_waveform(t, t_start, t_max):
    out = np.sin(0.5*np.pi*(t-t_start)/(t_max-t_start))
    out[t>=t_max] = 1 + (t[t>=t_max] - t_max)/t_max
    return out

# Define the on-times for the primary and the dual waveforms. Note that our
# discretization in time uses a log scale and is much smaller right before the
# off-time. This is to ensure the early times of the response are simulated accurately.
waveform_times_primary = np.r_[np.linspace(-0.005, -0.0011, 10), -np.logspace(-3, -6, 41), 0.]
waveform_times_dual = np.r_[np.linspace(-0.02, -0.011, 10), -np.logspace(-2, -5, 41), 0.]

# Define primary and dual current during on-time
waveform_current_primary = custom_waveform(waveform_times_primary, -0.005, -0.0005)
waveform_current_dual = custom_waveform(waveform_times_dual, -0.02, -0.002)

# Define the waveform object using the DualWaveform class
general_waveform = em1d.waveforms.DualWaveform(
    waveform_times=waveform_times_primary, waveform_current=waveform_current_primary,
    dual_waveform_times=waveform_times_dual, dual_waveform_current=waveform_current_dual
)

# SkyTEM 2015 waveform. By default, the primary and dual waveforms associated
# with some systems do not have an off-time starting at 0 or a peak current
# amplitude of 1. For comparison's sake, keyword arguments can be used to
# set the off time and scale the peak current amplitude. Here we normalize
# the waveform to have a peak amplitude of 1. 
maximum_current_amplitude = 1.
dual_maximum_current_amplitude = 1.
skytem_waveform = em1d.waveforms.Skytem2015Waveform(
    peak_current_amplitude=1., dual_peak_current_amplitude=1.
)

# Plotting the waveforms
fig = plt.figure(figsize=(8, 6))
ax = fig.add_axes([0.15, 0.15, 0.8, 0.75])

ax.plot(general_waveform.waveform_times, general_waveform.waveform_current, 'b-', lw=3)
ax.plot(general_waveform.dual_waveform_times, general_waveform.dual_waveform_current, 'b--', lw=3)
ax.plot(skytem_waveform.waveform_times, skytem_waveform.waveform_current, 'r-', lw=3)
ax.plot(skytem_waveform.dual_waveform_times, skytem_waveform.dual_waveform_current, 'r--', lw=3)

ax.set_xlabel("Time (s)")
ax.set_ylabel("Current (A)")
ax.set_title("Waveforms")
ax.legend(["General (primary)", "General (dual)", "SkyTEM (primary)", "SkyTEM (dual)"])


#####################################################################
# Create Survey
# -------------
# 
# Here we define the sources and receivers for the simulation. The waveform
# is a property of the source. When defining the receiver, we must define
# time channels for both the primary and dual waveforms.
# 

# Receiver location and geometry
receiver_location = np.array([0., 0., 0.])
receiver_orientation = "z"                    # "x", "y" or "z"
field_type = "secondary"                      # "secondary", "total" or "ppm"

# Source geometry
source_location = np.array([0., 0., 0.])  
source_radius = 10.      # loop radius
current_amplitude = 1.   # can scale by peak amplitude if waveform was normalized

# Define empty list to append sources
source_list = []

# Define source and receiver for our general waveform object. The time channels
# for the dual waveform are added using a keyword argument.
primary_times_general = np.logspace(-5, -3, 13)  # time channels for primary waveform
dual_times_general = np.logspace(-4, -2, 13)     # time channels for dual waveform

receiver_list = [
    em1d.receivers.PointReceiver(
        receiver_location, primary_times_general, orientation=receiver_orientation,
        component="dbdt", dual_times=dual_times_general
    )
]

source_list.append(
    em1d.sources.HorizontalLoopSource(
        receiver_list=receiver_list, location=source_location,
        waveform=general_waveform, radius=source_radius, current_amplitude=current_amplitude
    )
)

# Skytem 2015 source and receiver.
primary_times_skytem = skytem_2015_LM_time_channels()
dual_times_skytem = skytem_2015_HM_time_channels()

receiver_list = [
    em1d.receivers.PointReceiver(
        receiver_location, primary_times_skytem, orientation=receiver_orientation,
        component="dbdt", dual_times=dual_times_skytem
    )
]

source_list.append(
    em1d.sources.HorizontalLoopSource(
        receiver_list=receiver_list, location=source_location,
        waveform=skytem_waveform, radius=source_radius, current_amplitude=current_amplitude
    )
)

# Define the survey
survey = em1d.survey.EM1DSurveyTD(source_list)

###############################################
# Defining a 1D Layered Earth Model
# ---------------------------------
#
# Here, we define the layer thicknesses and electrical conductivities for our
# 1D simulation. If we have N layers, we define N electrical conductivity
# values and N-1 layer thicknesses. The lowest layer is assumed to extend to
# infinity.
#

# Layer thicknesses
thicknesses = np.array([40., 40.])
n_layer = len(thicknesses) + 1

# half-space physical properties
sigma = 1e-2
eta = 0.5
tau = 0.01
c = 0.5
chi = 0.

# physical property models
sigma_model = sigma * np.ones(n_layer)
eta_model = eta * np.ones(n_layer)
tau_model =  tau * np.ones(n_layer)
c_model = c * np.ones(n_layer)
chi_model = chi * np.ones(n_layer)

# Define a mapping for conductivities
model_mapping = maps.IdentityMap(nP=n_layer)

#######################################################################
# Define the Forward Simulation and Predict Data
# ----------------------------------------------
#

# Define the simulation
simulation = em1d.simulation.EM1DTMSimulation(
    survey=survey, thicknesses=thicknesses, sigmaMap=model_mapping,
    chi=chi_model
)

# Predict data for a given model
dpred = simulation.dpred(sigma_model)

#######################################################################
# Plotting Results
# -------------------------------------------------
#

k1 = len(primary_times_general)
k2 = len(dual_times_general)
k3 = len(primary_times_skytem) 

fig = plt.figure(figsize = (8, 8))
ax = fig.add_axes([0.15, 0.1, 0.8, 0.85])

ax.loglog(primary_times_general, np.abs(dpred[0:k1]), 'b-', lw=3)
ax.loglog(dual_times_general, np.abs(dpred[k1:k1+k2]), 'b--', lw=3)
ax.loglog(primary_times_skytem, np.abs(dpred[k1+k2:k1+k2+k3]), 'r-', lw=3)
ax.loglog(dual_times_skytem, np.abs(dpred[k1+k2+k3:]), 'r--', lw=3)

ax.legend(
    ["General (primary)", "General (dual)", "SkyTEM (primary)", "SkyTEM (dual)"]
)
ax.set_xlabel("Times (s)")
ax.set_ylabel("|dB/dt| (T/s)")
ax.set_title("TEM Response")
