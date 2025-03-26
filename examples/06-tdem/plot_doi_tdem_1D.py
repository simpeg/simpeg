##
#
# SkyTEM data
import numpy as np
import matplotlib.pyplot as plt

import simpeg.electromagnetics.time_domain as tdem
from simpeg import maps

from simpeg.utils import refine_1d_layer, doi_1d_layer_CA2012

##
# Set up the conductivity model structure
#
depths = np.atleast_1d([0, 2, 6, 19, 55, 180])
t = np.diff(depths)
rho = np.atleast_1d([50, 500, 10, 45, 500, 5])
sigma = 1 / rho

##
# Define survey parameters.
#
system = "312"

times_HM = np.r_[
    4.3641500e-04,
    4.5891500e-04,
    4.8691500e-04,
    5.2791500e-04,
    5.7891500e-04,
    6.2991500e-04,
    7.0591500e-04,
    8.0691500e-04,
    9.0791500e-04,
    1.0344150e-03,
    1.2114150e-03,
    1.4389150e-03,
    1.7424150e-03,
    2.1214150e-03,
    2.5759150e-03,
    3.1824150e-03,
    3.9404150e-03,
    4.8494150e-03,
    5.9604150e-03,
    7.2744150e-03,
    8.8924150e-03,
]

skytem_HM = tdem.sources.TrapezoidWaveform(
    ramp_on=[-5.0000e-03, -4.7242e-03], ramp_off=[0.0, 2.9805e-04]
)

receiver_orientation = "z"
# specs that are the same for the 306 and 312 systems
rx_area = 325
tx_area = 342
source_radius = np.sqrt(tx_area / np.pi)
source_altitude = 50

source_location = np.array([0.0, 0.0, source_altitude])
receiver_locations = np.array([0, 0.0, source_altitude + 2])

receiver_list_HM = [
    tdem.receivers.PointMagneticFluxTimeDerivative(
        receiver_locations, times_HM, orientation=receiver_orientation
    )
]

source_list_HM = [
    tdem.sources.CircularLoop(
        receiver_list=receiver_list_HM,
        location=source_location,
        waveform=skytem_HM,
        current=1,
        radius=source_radius,
        n_turns=1,
    )
]

survey_HM = tdem.Survey(source_list_HM)

##
# Simulate data in order to create a 'fake'
# data error array.
sigma_map = maps.ExpMap(nP=len(sigma))
simulation_HM = tdem.Simulation1DLayered(
    survey=survey_HM,
    thicknesses=t,
    sigmaMap=sigma_map,
)

m0 = np.log(sigma)

# Compute the predicted dB/dt data for the current model.
dBdT_pred_HM = simulation_HM.dpred(sigma)

# For the normalization, calculate the standard deviation of the data.
# std_data_HM = np.std(dBdT_pred_HM)
std_data_HM = 0.05 * np.abs(dBdT_pred_HM)

##
# Setup the sensitivity matrix for computing DOI.
t_star, m_star = refine_1d_layer(t, m0, 100)
sigma_map_J = maps.ExpMap(nP=len(m_star))
simulation_HM_J = tdem.Simulation1DLayered(
    survey=survey_HM,
    thicknesses=t_star,
    sigmaMap=sigma_map_J,
)

J = simulation_HM_J.getJ(m_star).copy()
J = J["ds"]

threshold = 0.8
doi, Sj_star = doi_1d_layer_CA2012(J, t_star, std_data_HM, threshold)

S = np.flip(np.cumsum(Sj_star[::-1]))

# print("Normalized aggregated sensitivity (per layer):")
print("Depth of Investigation (DOI): {:.2f} m".format(doi))

##
# Plot
fig, axs = plt.subplots(1, 3, figsize=(8, 6))
depths = np.r_[0, np.cumsum(t_star)]

y_min = 0.5
y_max = 500

##
# Resistivity model
axs[0].step(
    1.0 / m_star,
    depths,
    "k--",
    label="Resistivity",
)

axs[0].set_xscale("log")
axs[0].set_yscale("log")
axs[0].set_xlim(1, 1000)
axs[0].set_ylim(y_min, y_max)
axs[0].invert_yaxis()
axs[0].set_ylabel("Depth (m)")
axs[0].set_xlabel("Resistivity")
axs[0].grid(True)

##
# Sensitivity
axs[1].plot(
    Sj_star,
    depths[:-1],
    "o-",
    c="k",
    markersize=3,
    label="Sensitivity",
)

axs[1].set_xlabel("Sensitivity")
axs[1].set_xscale("log")
axs[1].set_yscale("log")
# axs[1].set_xlim(0.01, 1000)
axs[1].set_ylim(y_min, y_max)
axs[1].invert_yaxis()

##
# Cummulated sensitivity
axs[2].plot(
    S,
    depths[:-1],
    "o-",
    c="k",
    markersize=3,
    label="Cumulative sensitivity",
)

axs[2].axhline(
    doi,
    color="green",
    linestyle="--",
    label="DOI_HM ({:.1f}) m".format(doi),
)

axs[2].axvline(
    threshold,
    color="red",
    linestyle="--",
    label="Threshold ({:.8})".format(threshold),
)

axs[2].set_xlabel("Cumulative Sensitivity")
axs[2].set_xscale("log")
axs[2].set_yscale("log")
# axs[2].set_xlim(0.1, 100)
axs[2].set_ylim(y_min, y_max)
axs[2].invert_yaxis()

plt.suptitle("DOI")

plt.legend()
plt.grid(True)

plt.show()
