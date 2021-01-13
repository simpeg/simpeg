"""
Forward Simulation for Dispersive Physical Properties
=====================================================





"""

#####################################################
# Import Modules
# --------------
#

import numpy as np
from matplotlib import pyplot as plt

from SimPEG import maps
import simpegEM1D as em1d
from simpegEM1D.analytics import ColeCole


#####################################################################
# Create Survey
# -------------
#
#

source_location = np.array([0., 0., 0.])
source_orientation = "z"  # "x", "y" or "z"
source_current = 1.
moment_amplitude = 1.

receiver_location = np.array([8., 0., 0.])
receiver_orientation = "z"  # "x", "y" or "z"
field_type = "secondary"  # "secondary", "total" or "ppm"

frequencies = np.logspace(-1, 8, 51)

# Receiver list
receiver_list = []
receiver_list.append(
    em1d.receivers.HarmonicPointReceiver(
        receiver_location, frequencies, orientation=receiver_orientation,
        field_type=field_type, component="real"
    )
)

receiver_list.append(
    em1d.receivers.HarmonicPointReceiver(
        receiver_location, frequencies, orientation=receiver_orientation,
        field_type="secondary", component="imag"
    )
)

# Sources
source_list = [
    em1d.sources.HarmonicMagneticDipoleSource(
        receiver_list=receiver_list, location=source_location, orientation=source_orientation,
        moment_amplitude=moment_amplitude
    )
]

# Survey
survey = em1d.survey.EM1DSurveyFD(source_list)


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
thicknesses = np.array([20., 20.])
n_layer = len(thicknesses) + 1

# half-space physical properties
sigma = 1e-2
eta = 0.5
tau = 0.001
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
sigma_complex = ColeCole(frequencies, sigma, eta, tau, c)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.semilogx(frequencies, sigma*np.ones(len(frequencies)), "b", lw=3)
ax.semilogx(frequencies, np.real(sigma_complex), "r", lw=3)
ax.semilogx(frequencies, np.imag(sigma_complex), "r--", lw=3)
ax.set_xlim(np.min(frequencies), np.max(frequencies))
ax.set_ylim(0., 1.1*sigma)
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Conductivity")
ax.legend(
    ["$\sigma_{\infty}$", "$Re[\sigma (\omega)]$", "$Im[\sigma (\omega)]$"],
    loc="center right"
)
plt.show()

#######################################################################
# Define the Forward Simulation and Predict Data
# -----------------------------------------------
#


# Simulate response for static conductivity
simulation = em1d.simulation.EM1DFMSimulation(
    survey=survey, thicknesses=thicknesses, sigmaMap=model_mapping,
    chi=chi_model
)

dpred = simulation.dpred(sigma_model)

# Simulate response for complex conductivity
simulation_colecole = em1d.simulation.EM1DFMSimulation(
    survey=survey, thicknesses=thicknesses, sigmaMap=model_mapping,
    eta=eta, tau=tau, c=c, chi=chi_model
)

dpred_colecole = simulation_colecole.dpred(sigma_model)


#######################################################################
# Plotting Results
# -------------------------------------------------
#
#


fig, ax = plt.subplots(1,1, figsize = (7, 7))
ax.loglog(frequencies, np.abs(dpred[0:len(frequencies)]), 'b-', lw=2)
ax.loglog(frequencies, np.abs(dpred[len(frequencies):]), 'b--', lw=2)
ax.loglog(frequencies, np.abs(dpred_colecole[0:len(frequencies)]), 'r-', lw=2)
ax.loglog(frequencies, np.abs(dpred_colecole[len(frequencies):]), 'r--', lw=2)
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("|H| (A/m)")
ax.set_title("Magnetic Field as a Function of Frequency")
ax.legend((
    'Real (non-chargeable)', 'Imaginary (non-chargeable)',
    'Real (chargeable)', 'Imaginary (chargeable)'
))

























