##
#
# FDEM DOI example.
import numpy as np
import matplotlib.pyplot as plt

import simpeg.electromagnetics.frequency_domain as fdem
from simpeg import maps

from simpeg.electromagnetics.frequency_domain.doi import doi_fdem_1d_layer_CA2012

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

# Frequencies being observed in Hz
frequencies = np.array([382, 1822, 7970, 35920, 130100], dtype=float)

# Define a list of receivers. The real and imaginary components are defined
# as separate receivers.
receiver_location = np.array([10.0, 0.0, 30.0])
receiver_orientation = "z"  # "x", "y" or "z"
data_type = "ppm"  # "secondary", "total" or "ppm"

receiver_list = []
receiver_list.append(
    fdem.receivers.PointMagneticFieldSecondary(
        receiver_location,
        orientation=receiver_orientation,
        data_type=data_type,
        component="real",
    )
)
receiver_list.append(
    fdem.receivers.PointMagneticFieldSecondary(
        receiver_location,
        orientation=receiver_orientation,
        data_type=data_type,
        component="imag",
    )
)

# Define the source list. A source must be defined for each frequency.
source_location = np.array([0.0, 0.0, 30.0])
source_orientation = "z"  # "x", "y" or "z"
moment = 1.0  # dipole moment

source_list = []
for freq in frequencies:
    source_list.append(
        fdem.sources.MagDipole(
            receiver_list=receiver_list,
            frequency=freq,
            location=source_location,
            orientation=source_orientation,
            moment=moment,
        )
    )

# Define a 1D FDEM survey
survey = fdem.survey.Survey(source_list)

##
# Simulate data in order to create a 'fake'
# data error array.
sigma_map = maps.IdentityMap(nP=len(sigma))
simulation = fdem.Simulation1DLayered(
    survey=survey,
    thicknesses=t,
    sigmaMap=sigma_map,
)

# Compute the predicted dB/dt data for the current model.
dpred = simulation.dpred(sigma)

# For the normalization, calculate the standard deviation of the data.
std_data = 0.05 * np.abs(dpred)

threshold = 0.8
doi, t_star, m_star, Sj_star, S = doi_fdem_1d_layer_CA2012(
    t, sigma, survey, std_data, threshold
)

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
