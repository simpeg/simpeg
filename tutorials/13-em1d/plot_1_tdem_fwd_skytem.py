"""
Forward Simulation with Different Waveforms
===========================================





"""

#####################################################
# Import Modules
# --------------
#

import numpy as np
from matplotlib import pyplot as plt

from SimPEG import maps
import simpegEM1D as em1d
from simpegEM1D import skytem_HM_2015, skytem_LM_2015


#####################################################################
# Create Survey
# -------------
#
#

wave_HM = skytem_HM_2015()
wave_LM = skytem_LM_2015()
time_HM = wave_HM.time_gate_center[0::2]
time_LM = wave_LM.time_gate_center[0::2]


source_location = np.array([0., 0., 40.])
source_orientation = "z"  # "x", "y" or "z"
source_current = 1.
moment_amplitude=1.
receiver_offset_r = 13.25
receiver_offset_z = 2.
receiver_location = np.array([receiver_offset_r, 0., 40.+receiver_offset_z ])
receiver_orientation = "z"  # "x", "y" or "z"
field_type = "secondary"  # "secondary", "total" or "ppm"

# Receiver list
rx = em1d.receivers.TimeDomainPointReceiver(
        receiver_location,
        times=time_HM,
        times_dual_moment=time_LM,
        orientation=receiver_orientation,
        component="dbdt"
)
receiver_list = [rx]

# Sources

time_input_currents_HM = wave_HM.current_times[-7:]
input_currents_HM = wave_HM.currents[-7:]
time_input_currents_LM = wave_LM.current_times[-13:]
input_currents_LM = wave_LM.currents[-13:]

src = em1d.sources.TimeDomainMagneticDipoleSource(
    receiver_list=receiver_list,
    location=source_location,
    moment_amplitude=moment_amplitude,
    orientation=source_orientation,
    wave_type="general",
    moment_type='dual',
    time_input_currents=time_input_currents_HM,
    input_currents=input_currents_HM,
    n_pulse = 1,
    base_frequency = 25.,
    time_input_currents_dual_moment = time_input_currents_LM,
    input_currents_dual_moment = input_currents_LM,
    base_frequency_dual_moment = 210
)
source_list = [src]

# Survey
survey = em1d.survey.EM1DSurveyTD(source_list)


###############################################
# Plot the Waveforms
# ------------------
#
#

fig = plt.figure(figsize=(6, 4))
ax = fig.add_axes([0.1, 0.1, 0.85, 0.8])
ax.plot(time_input_currents_HM, input_currents_HM, 'b', lw=2, label='HM')
ax.plot(time_input_currents_LM, input_currents_LM, 'r', lw=2, label='LM')
ax.set_xlabel("Time (s)")
ax.set_ylabel("Normalized Current (A)")
ax.legend()






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
# physical property models
sigma_model = sigma * np.ones(n_layer)

# Define a mapping for conductivities
model_mapping = maps.IdentityMap(nP=n_layer)


#######################################################################
# Define the Forward Simulation and Predict Data
# ----------------------------------------------
#


# Simulate response for static conductivity
simulation = em1d.simulation.EM1DTMSimulation(
    survey=survey, thicknesses=thicknesses, sigmaMap=model_mapping,
)

dpred = simulation.dpred(sigma_model)


#######################################################################
# Plotting Results
# -------------------------------------------------
#
#


fig = plt.figure(figsize = (6, 5))
ax = fig.add_axes([0.1, 0.1, 0.8, 0.85])
ax.loglog(rx.times, -dpred[:rx.times.size], 'b', lw=2, label='HM')
ax.loglog(rx.times_dual_moment, -dpred[rx.times.size:], 'r', lw=2, label='LM')

ax.legend()
ax.set_xlabel("Times (s)")
ax.set_ylabel("|dB/dt| (T/s)")
plt.show()

















