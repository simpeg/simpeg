"""
Forward Simulation for a 1D Sounding: Dual Waveforms
====================================================

Some TDEM systems use separate waveforms to model the response at early times and
at late times. In the tutorial, we show the user how to define waveforms, sources
and receivers in this case. In addition to defining the early time and late
time waveforms, the user must also define which time channels in the receiver
are early time and which are late time. 

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
# Here, we define wave 
# 



# General waveform
def custom_waveform(t, t_start, t_max):
    out = np.sin(0.5*np.pi*(t-t_start)/(t_max-t_start))
    out[t>=t_max] = 1 + (t[t>=t_max] - t_max)/t_max
    return out

waveform_times_early = np.r_[np.linspace(-0.005, -0.0011, 10), -np.logspace(-3, -6, 41), 0.]
waveform_times_late = np.r_[np.linspace(-0.02, -0.011, 10), -np.logspace(-2, -5, 41), 0.]

waveform_current_early = custom_waveform(waveform_times_early, -0.005, -0.0005)
waveform_current_late = custom_waveform(waveform_times_late, -0.02, -0.002)

general_waveform = em1d.waveforms.DualWaveform(
    waveform_times=waveform_times_early, waveform_current=waveform_current_early,
    dual_waveform_times=waveform_times_late, dual_waveform_current=waveform_current_late
)

# SkyTEM waveform
off_time = 0.
maximum_current_amplitude = 1.
dual_maximum_current_amplitude = 1.
skytem_waveform = em1d.waveforms.Skytem2015Waveform(peak_current_amplitude=1., dual_peak_current_amplitude=1.)

###############################################
# Plot the Waveforms
# ------------------
#
# Here, we plot the set of waveforms that are used in the simulation.
#

fig = plt.figure(figsize=(8, 6))
ax = fig.add_axes([0.1, 0.1, 0.85, 0.8])

ax.plot(general_waveform.waveform_times, general_waveform.waveform_current, 'b-', lw=2)
ax.plot(general_waveform.dual_waveform_times, general_waveform.dual_waveform_current, 'b--', lw=2)
ax.plot(skytem_waveform.waveform_times, skytem_waveform.waveform_current, 'r-', lw=2)
ax.plot(skytem_waveform.dual_waveform_times, skytem_waveform.dual_waveform_current, 'r--', lw=2)


ax.set_xlabel("Time (s)")
ax.set_ylabel("Current (A)")
ax.set_title("Waveforms")
ax.legend(["General (low-moment)", "General (high-moment)", "SkyTEM (low-moment)", "SkyTEM (high-moment)"])


#####################################################################
# Create Survey
# -------------
# 
# The waveform is a property of the source. So for each waveform, we will need
# to define a separate source object. For simplicity, all sources will be
# horizontal loops with a radius of 10 m.
# 

# Receiver location and geometry
receiver_location = np.array([0., 0., 0.])
receiver_orientation = "z"                    # "x", "y" or "z"
field_type = "secondary"                      # "secondary", "total" or "ppm"

# Source geometry
source_location = np.array([0., 0., 0.])  
source_radius = 10.
current_amplitude = 1.


source_list = []

# Define general source and receiver
early_times_general = np.logspace(-4, -2, 13)
late_times_general = np.logspace(-3, -1, 13)

receiver_list = [
    em1d.receivers.PointReceiver(
        receiver_location, early_times_general, orientation=receiver_orientation,
        component="dbdt", dual_times=late_times_general
    )
]

source_list.append(
    em1d.sources.HorizontalLoopSource(
        receiver_list=receiver_list, location=source_location,
        waveform=general_waveform, radius=source_radius, current_amplitude=current_amplitude
    )
)

# Skytem
early_times_skytem = skytem_2015_LM_time_channels()
late_times_skytem = skytem_2015_HM_time_channels()

receiver_list = [
    em1d.receivers.PointReceiver(
        receiver_location, early_times_skytem, orientation=receiver_orientation,
        component="dbdt", dual_times=late_times_skytem
    )
]

source_list.append(
    em1d.sources.HorizontalLoopSource(
        receiver_list=receiver_list, location=source_location,
        waveform=skytem_waveform, radius=source_radius, current_amplitude=current_amplitude
    )
)
    

# Survey
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

k1 = len(early_times_general)
k2 = len(late_times_general)
k3 = len(early_times_skytem) 

fig = plt.figure(figsize = (8, 8))
ax = fig.add_axes([0.1, 0.1, 0.8, 0.85])

ax.loglog(early_times_general, np.abs(dpred[0:k1]), 'b-', lw=2)
ax.loglog(late_times_general, np.abs(dpred[k1:k1+k2]), 'b--', lw=2)
ax.loglog(early_times_skytem, np.abs(dpred[k1+k2:k1+k2+k3]), 'r-', lw=2)
ax.loglog(late_times_skytem, np.abs(dpred[k1+k2+k3:]), 'r--', lw=2)

ax.legend(["General (low-moment)", "General (high-moment)", "SkyTEM (low-moment)", "SkyTEM (high-moment)"])
ax.set_xlabel("Times (s)")
ax.set_ylabel("|dB/dt| (T/s)")
ax.set_title("TEM Response")
