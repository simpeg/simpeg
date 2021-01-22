"""
Forward Simulation for a Single 1D Sounding for a Susceptible and Chargeable Earth
==================================================================================

Here we use the module *SimPEG.electromangetics.frequency_domain_1d* to compare
predicted frequency domain data for a single sounding when the Earth is
purely conductive, conductive and magnetically susceptible, and when it is chargeable.
In this tutorial, we focus on:

    - Defining receivers, sources and the survey
    - Defining physical properties when the Earth is chargeable and/or magnetically susceptibility
    - Setting physical property values as constant in the simulation

Our survey geometry consists of a vertical magnetic dipole source
located 30 m above the Earth's surface. The receiver is offset
10 m horizontally from the source.


"""

#####################################################
# Import Modules
# --------------
#

import numpy as np
from matplotlib import pyplot as plt

from SimPEG import maps
import SimPEG.electromagnetics.frequency_domain_1d as em1d
from SimPEG.electromagnetics.utils.em1d_utils import ColeCole

plt.rcParams.update({'font.size': 16})

# sphinx_gallery_thumbnail_number = 2

#####################################################################
# Create Survey
# -------------
#
# Here we demonstrate a general way to define the receivers, sources and survey.
# For this tutorial, we define a single vertical magnetic dipole source as well
# as receivers which measure real and imaginary ppm data for a set of frequencies.
# 

# Frequencies being observed in Hz
frequencies = np.logspace(0, 8, 41)

# Define a list of receivers. The real and imaginary components are defined
# as separate receivers.
receiver_location = np.array([10., 0., 10.])
receiver_orientation = "z"                   # "x", "y" or "z"
field_type = "secondary"                     # "secondary", "total" or "ppm"

receiver_list = []
receiver_list.append(
    em1d.receivers.PointReceiver(
        receiver_location, frequencies, orientation=receiver_orientation,
        field_type=field_type, component="real"
    )
)
receiver_list.append(
    em1d.receivers.PointReceiver(
        receiver_location, frequencies, orientation=receiver_orientation,
        field_type=field_type, component="imag"
    )
)

# Define a source list. For each list of receivers, we define a source.
# In this case, we define a single source.
source_location = np.array([0., 0., 10.])
source_orientation = 'z'                      # "x", "y" or "z"
moment_amplitude = 1.                         # dipole moment amplitude

source_list = [
    em1d.sources.MagneticDipoleSource(
        receiver_list=receiver_list, location=source_location,
        orientation=source_orientation, moment_amplitude=moment_amplitude
    )
]

# Define a 1D FDEM survey
survey = em1d.survey.EM1DSurveyFD(source_list)


###############################################
# Defining a Layered Earth Model
# ------------------------------
#
# Here, we define the layer thicknesses and physical properties for our
# 1D simulation. If we have N layers, parameters for the physical properties
# must be defined for each layer and we must provide N-1 layer thicknesses.
# The lowest layer is assumed to extend to infinity.
#
# For this tutorial, we predict the response for a halfspace model, however
# the script has been generalized to work for an arbitrary number of layers.
# If the Earth is a halfspace, the thicknesses could instead be defined by
# an empty array, and each physical property value by an array of length 1.
#

# Layer thicknesses
thicknesses = np.array([20, 40])
n_layer = len(thicknesses) + 1

# In SimPEG, the Cole-Cole model is used to define a frequency-dependent
# electrical conductivity when the Earth is chargeable.
sigma = 1e-2        # infinite conductivity in S/m
eta = 0.8           # intrinsice chargeability [0, 1]
tau = 0.0001        # central time-relaxation constant in seconds
c = 0.8             # phase constant [0, 1]

# Magnetic susceptibility in SI
chi = 0.2

# For each physical property, the parameters must be defined for each layer.
# In this case, we must define all parameters for the Cole-Cole conductivity
# as well as the magnetic susceptibility.
sigma_model = sigma * np.ones(n_layer)
eta_model = eta * np.ones(n_layer)
tau_model =  tau * np.ones(n_layer)
c_model = c * np.ones(n_layer)
chi_model = chi * np.ones(n_layer)

# Here, we let the infinite conductivity be the model. As a result, we only
# need to define the mapping for this parameter. All other parameters used
# to define physical properties will be fixed when creating the simulation.
model_mapping = maps.IdentityMap(nP=n_layer)

# Plot complex conductivity at all frequencies
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
# Here we predict the FDEM sounding for several halfspace models
# (conductive, susceptible, chargeable). Since the physical properties defining 
# the Earth are different, it requires a separate simulation object be created
# for each case. Each simulation requires the user
# define the survey, the layer thicknesses and a mapping.
# 
# A universal mapping was created by letting sigma be the model. All other
# parameters used to define the physical properties are permanently set when
# defining the simulation.
#
# When using the *SimPEG.electromagnetics.frequency_domain_1d* module, note that
# predicted data are organized by source, then by receiver, then by frequency.
#
#

# Response for conductive Earth
simulation = em1d.simulation.EM1DFMSimulation(
    survey=survey, thicknesses=thicknesses, sigmaMap=model_mapping
)

dpred = simulation.dpred(sigma_model)

# Simulate response for a conductive and susceptible Earth
simulation_susceptible = em1d.simulation.EM1DFMSimulation(
    survey=survey, thicknesses=thicknesses, sigmaMap=model_mapping,
    chi=chi_model
)

dpred_susceptible = simulation_susceptible.dpred(sigma_model)

# Simulate response for a chargeable Earth
simulation_chargeable = em1d.simulation.EM1DFMSimulation(
    survey=survey, thicknesses=thicknesses, sigmaMap=model_mapping,
    eta=eta, tau=tau, c=c
)

dpred_chargeable = simulation_chargeable.dpred(sigma_model)


#######################################################################
# Plotting Results
# -------------------------------------------------
#

fig = plt.figure(figsize=(7, 7))
ax = fig.add_axes([0.15, 0.1, 0.8, 0.8])
ax.semilogx(frequencies, dpred[0:len(frequencies)], 'b-', lw=3)
ax.semilogx(frequencies, dpred[len(frequencies):], 'b--', lw=3)
ax.semilogx(frequencies, dpred_susceptible[0:len(frequencies)], 'r-', lw=3)
ax.semilogx(frequencies, dpred_susceptible[len(frequencies):], 'r--', lw=3)
ax.semilogx(frequencies, dpred_chargeable[0:len(frequencies)], 'g-', lw=3)
ax.semilogx(frequencies, dpred_chargeable[len(frequencies):], 'g--', lw=3)
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("|H| (A/m)")
ax.set_title("Secondary Magnetic Field")
ax.legend((
    'Real (conductive)', 'Imaginary (conductive)',
    'Real (susceptible)', 'Imaginary (susceptible)',
    'Real (chargeable)', 'Imaginary (chargeable)'
))

