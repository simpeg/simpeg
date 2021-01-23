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


#####################################################################
# Define Waveforms
# ----------------
#

stepoff_waveform = em1d.waveforms.StepoffWaveform()

waveform_times = np.r_[-0.01, -np.logspace(-2.01, -5, 31), 0.]
triangle_waveform = em1d.waveforms.TriangleWaveform(
        -0.01, -0.005, -1e-10, 1, waveform_times=waveform_times,
        n_pulse=1, base_frequency=25
)

skytem_waveform_lm = em1d.waveforms.SkytemLM2015Waveform(
        peak_time=-9.4274e-006, peak_current_amplitude=1.
)

skytem_waveform_hm = em1d.waveforms.SkytemHM2015Waveform(
        -1.96368E-04, 1.
)

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

# Stepoff Waveform
source_list.append(
    em1d.sources.MagneticDipoleSource(
        receiver_list=receiver_list, location=source_location, waveform=stepoff_waveform,
        orientation=source_orientation, moment_amplitude=moment_amplitude
    )
)

# Triangle Waveform
source_list.append(
    em1d.sources.MagneticDipoleSource(
        receiver_list=receiver_list, location=source_location, waveform=triangle_waveform,
        orientation=source_orientation, moment_amplitude=moment_amplitude
    )
)

# Skytem Waveform LM
source_list.append(
    em1d.sources.MagneticDipoleSource(
        receiver_list=receiver_list, location=source_location, waveform=skytem_waveform_lm,
        orientation=source_orientation, moment_amplitude=moment_amplitude
    )
)
    
    # Skytem Waveform HM
source_list.append(
    em1d.sources.MagneticDipoleSource(
        receiver_list=receiver_list, location=source_location, waveform=skytem_waveform_hm,
        orientation=source_orientation, moment_amplitude=moment_amplitude
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
ax.plot(waveform_times, triangle_waveform.waveform_currents, 'b', lw=2)
ax.plot(skytem_waveform_lm.waveform_times, skytem_waveform_lm.waveform_currents, 'r', lw=2)
ax.plot(skytem_waveform_hm.waveform_times, skytem_waveform_hm.waveform_currents, 'g', lw=2)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Normalized Current (A)")
ax.legend(["Step-off", "Triangle", "SkyTEM LM", "SkyTEM HM"])





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
colorlist = ['k', 'b', 'r', 'g']
for ii, k in enumerate(colorlist):
    ax.loglog(times, np.abs(d[:, ii]), k, lw=2)

ax.legend(["Step-off", "Triangle", "SkyTEM LM", "SkyTEM HM"])
ax.set_xlabel("Times (s)")
ax.set_ylabel("|dB/dt| (T/s)")


























