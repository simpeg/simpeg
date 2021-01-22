"""
Forward Simulation for a Single 1D Sounding for Different Waveforms
===================================================================





"""

#####################################################
# Import Modules
# --------------
#

import numpy as np
from matplotlib import pyplot as plt

from SimPEG import maps
import SimPEG.electromagnetics.time_domain_1d as em1d
from SimPEG.electromagnetics.utils.em1d_utils import ColeCole
from SimPEG.electromagnetics.time_domain_1d.waveforms import TriangleFun, VTEMFun



#####################################################################
# Create Survey
# -------------
#
#

source_location = np.array([0., 0., 0.])  
source_orientation = "z"  # "x", "y" or "z"
source_current = 1.
source_radius = 10.
moment_amplitude=1.

receiver_location = np.array([10., 0., 0.])
receiver_orientation = "z"  # "x", "y" or "z"
field_type = "secondary"  # "secondary", "total" or "ppm"

times = np.logspace(-5, -2, 41)

# Receiver list
receiver_list = [
    em1d.receivers.PointReceiver(
        receiver_location, times, orientation=receiver_orientation,
        component="dbdt"
    )
]

# Sources
source_list = []

# Step off
source_list.append(
    em1d.sources.MagneticDipoleSource(
        receiver_list=receiver_list, location=source_location,
        orientation=source_orientation, moment_amplitude=moment_amplitude,
        wave_type="stepoff"
    )
)

# Triangle Waveform
time_input_currents = np.r_[-np.logspace(-2, -5, 31), 0.]
input_currents = TriangleFun(time_input_currents+0.01, 5e-3, 0.01)
source_list.append(
    em1d.sources.MagneticDipoleSource(
        receiver_list=receiver_list, location=source_location,
        moment_amplitude=moment_amplitude,
        orientation=source_orientation,
        wave_type="general",
        time_input_currents=time_input_currents,
        input_currents=input_currents,
        n_pulse = 1,
        base_frequency = 25.,
        use_lowpass_filter=False,
        high_cut_frequency=210*1e3
    )
)

# VTEM waveform
time_input_currents_2 = np.r_[-np.logspace(-2, -5, 31), 0.]
input_currents_2 = VTEMFun(time_input_currents+0.01, 8e-3, 0.01, 1)
source_list.append(
    em1d.sources.MagneticDipoleSource(
        receiver_list=receiver_list, location=source_location,
        moment_amplitude=moment_amplitude,
        orientation=source_orientation, 
        wave_type="general",
        time_input_currents=time_input_currents_2,
        input_currents=input_currents_2,
        n_pulse = 1,
        base_frequency = 25.,
        use_lowpass_filter=False,
        high_cut_frequency=210*1e3
    )
)


# Survey
survey = em1d.survey.EM1DSurveyTD(source_list)

###############################################
# Plot the Waveforms
# ------------------
#
#

fig = plt.figure(figsize=(6, 4))
ax = fig.add_axes([0.1, 0.1, 0.85, 0.8])
ax.plot(np.r_[-1e-2, 0., 1e-10, 1e-3], np.r_[1., 1., 0., 0.], 'k', lw=2)
ax.plot(time_input_currents, input_currents, 'b', lw=2)
ax.plot(time_input_currents_2, input_currents_2, 'r', lw=2)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Normalized Current (A)")
ax.legend(["Step-off", "Triangular", "VTEM"])





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

# Compute and plot complex conductivity at all frequencies
frequencies = np.logspace(-3, 6, 91)
#sigma_complex = ColeCole(frequencies, sigma, eta, tau, c)
#
#fig = plt.figure()
#ax = fig.add_subplot(111)
#ax.semilogx(frequencies, sigma*np.ones(len(frequencies)), "b", lw=2)
#ax.semilogx(frequencies, np.real(sigma_complex), "r", lw=2)
#ax.semilogx(frequencies, np.imag(sigma_complex), "r--", lw=2)
#ax.set_xlim(np.min(frequencies), np.max(frequencies))
#ax.set_ylim(0., 1.1*sigma)
#ax.set_xlabel("Frequency (Hz)")
#ax.set_ylabel("Conductivity")
#ax.legend(
#    ["$\sigma_{DC}$", "$Re[\sigma (\omega)]$", "$Im[\sigma (\omega)]$"],
#    loc="center right"
#)

#######################################################################
# Define the Forward Simulation and Predict Data
# ----------------------------------------------
#


# Simulate response for static conductivity
simulation = em1d.simulation.EM1DTMSimulation(
    survey=survey, thicknesses=thicknesses, sigmaMap=model_mapping,
    chi=chi_model
)

dpred = simulation.dpred(sigma_model)

#######################################################################
# Plotting Results
# -------------------------------------------------
#
#


fig = plt.figure(figsize = (6, 5))
d = np.reshape(dpred, (len(source_list), len(times))).T
ax = fig.add_axes([0.1, 0.1, 0.8, 0.85])
colorlist = ['k', 'b', 'r']
for ii, k in enumerate(colorlist):
    ax.loglog(times, np.abs(d[:, ii]), k, lw=2)

ax.legend(["Step-off", "Triangular", "VTEM"])
ax.set_xlabel("Times (s)")
ax.set_ylabel("|dB/dt| (T/s)")


























