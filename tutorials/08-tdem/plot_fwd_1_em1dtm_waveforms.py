"""
1D Forward Simulation with User-Defined Waveforms
=================================================

For time-domain electromagnetic problems, the response depends strongly on the
souce waveforms. In this tutorial, we construct a set of waveforms of different
types and simulate the response for a halfspace. Many types of waveforms can
be constructed within *SimPEG.electromagnetics.time_domain_1d*. These include:
    
    - the unit step off waveform
    - a set of basic waveforms: rectangular, triangular, quarter sine, etc...
    - a set of system-specific waveforms: SkyTEM, VTEM, GeoTEM, etc...
    - fully customized waveforms


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


#####################################################################
# Define Waveforms
# ----------------
# 
# Here, we define the set of waveforms that will be used to simulated the
# TEM response.
# 

# Unit stepoff waveform can be defined directly
stepoff_waveform = em1d.waveforms.StepoffWaveform()

# Rectangular waveform. The user may customize the waveform by setting the start
# time, end time and on time amplitude for the current waveform.
waveform_times = np.r_[np.linspace(-0.02, -0.011, 10), -np.logspace(-2, -6, 61), 0.]
start_time = -0.004
end_time = 0.
peak_current_amplitude = 1.
rectangular_waveform = em1d.waveforms.RectangularWaveform(
        waveform_times, start_time, end_time, peak_current_amplitude
)

# Triangular waveform. The user may customize the waveform by setting the start
# time, peak time, end time and peak amplitude for the current waveform.
waveform_times = np.r_[np.linspace(-0.02, -0.011, 10), -np.logspace(-2, -6, 61), 0.]
start_time = -0.02
peak_time = -0.01
end_time = 0.
peak_current_amplitude = 1.
triangle_waveform = em1d.waveforms.TriangleWaveform(
        waveform_times, start_time, peak_time, end_time, peak_current_amplitude
)

# VTEM plus 2015 waveform is part of small library of system-specific waveforms.
# The discretization of the waveform is fixed, however the user may specify
# the beginning of the off-time and scale the amplitude of the waveform.
off_time = 0.
peak_current_amplitude = 1.
vtem_waveform = em1d.waveforms.VTEMPlusWaveform(
    off_time=off_time, peak_current_amplitude=peak_current_amplitude
)

# General waveform. This is a fully general way to define the waveform.
# The use simply provides times and the current.
def custom_waveform(t, tmax):
    out = np.cos(0.5*np.pi*(t-tmax)/(tmax+0.02))
    out[t>=tmax] = 1 + (t[t>=tmax] - tmax)/tmax
    return out

waveform_times = np.r_[np.linspace(-0.02, -0.011, 10), -np.logspace(-2, -6, 61), 0.]
waveform_current = custom_waveform(waveform_times, -0.0055)
general_waveform = em1d.waveforms.GeneralWaveform(
        waveform_times=waveform_times, waveform_current=waveform_current
)

###############################################
# Plot the Waveforms
# ------------------
#
# Here, we plot the set of waveforms that are used in the simulation.
#

fig = plt.figure(figsize=(8, 6))
ax = fig.add_axes([0.1, 0.1, 0.85, 0.8])

ax.plot(np.r_[-2e-2, 0., 1e-10, 1e-3], np.r_[1., 1., 0., 0.], 'k', lw=3)
ax.plot(rectangular_waveform.waveform_times, rectangular_waveform.waveform_current, 'r', lw=2)
ax.plot(triangle_waveform.waveform_times, triangle_waveform.waveform_current, 'b', lw=2)
ax.plot(vtem_waveform.waveform_times, vtem_waveform.waveform_current, 'g', lw=2)
ax.plot(general_waveform.waveform_times, general_waveform.waveform_current, 'm', lw=2)

ax.set_xlabel("Time (s)")
ax.set_ylabel("Current (A)")
ax.set_title("Waveforms")
ax.legend(["Step-off", "Rectangular", "Triangle", "VTEM Plus", "General"])


#####################################################################
# Create Survey
# -------------
# 
# The waveform is a property of the source. So for each waveform, we will need
# to define a separate source object. For simplicity, all sources will be
# horizontal loops with a radius of 10 m.
# 

# Define a receiver list. In this case, we measure the vertical component of
# db/dt. Thus we only have a single receiver in the list.
receiver_location = np.array([0., 0., 0.])
receiver_orientation = "z"                    # "x", "y" or "z"
field_type = "secondary"                      # "secondary", "total" or "ppm"
times = np.logspace(-4, -1, 41)               # time channels

receiver_list = [
    em1d.receivers.PointReceiver(
        receiver_location, times, orientation=receiver_orientation,
        component="dbdt"
    )
]

# Source properties. If you defined the true waveform (not normalized), the current amplitude
# should be set to 1. Otherwise you will be accounting for the maximum current
# amplitude twice!!! 
source_location = np.array([0., 0., 0.])  
source_radius = 10.
current_amplitude = 1.

source_list = []

# Stepoff Waveform
source_list.append(
    em1d.sources.HorizontalLoopSource(
        receiver_list=receiver_list, location=source_location,
        waveform=stepoff_waveform, radius=source_radius, current_amplitude=current_amplitude
    )
)
    
# Rectangular Waveform
source_list.append(
    em1d.sources.HorizontalLoopSource(
        receiver_list=receiver_list, location=source_location,
        waveform=rectangular_waveform, radius=source_radius, current_amplitude=current_amplitude
    )
)

# Triangle Waveform
source_list.append(
    em1d.sources.HorizontalLoopSource(
        receiver_list=receiver_list, location=source_location,
        waveform=triangle_waveform, radius=source_radius, current_amplitude=current_amplitude
    )
)
    
# VTEM Plus Waveform
source_list.append(
    em1d.sources.HorizontalLoopSource(
        receiver_list=receiver_list, location=source_location,
        waveform=vtem_waveform, radius=source_radius, current_amplitude=current_amplitude
    )
)
    
# General Waveform
source_list.append(
    em1d.sources.HorizontalLoopSource(
        receiver_list=receiver_list, location=source_location,
        waveform=general_waveform, radius=source_radius, current_amplitude=current_amplitude
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

fig = plt.figure(figsize = (8, 8))
d = np.reshape(dpred, (len(source_list), len(times))).T
ax = fig.add_axes([0.1, 0.1, 0.8, 0.85])
colorlist = ['k', 'b', 'r', 'g', 'm']
for ii, k in enumerate(colorlist):
    ax.loglog(times, np.abs(d[:, ii]), k, lw=2)

ax.legend(["Step-off", "Rectangular", "Triangle", "VTEM Plus", "General"])
ax.set_xlabel("Times (s)")
ax.set_ylabel("|dB/dt| (T/s)")
ax.set_title("TEM Response")
