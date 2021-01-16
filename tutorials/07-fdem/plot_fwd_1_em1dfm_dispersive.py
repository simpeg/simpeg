"""
Forward Simulation for a Single 1D Sounding for a Susceptible and Chargeable Earth
==================================================================================

Here we use the module *SimPEG.electromangetics.frequency_domain_1d* to predict
frequency domain data for a single sounding when the Earth is
chargeable and/or susceptible In this tutorial, we focus on the following:

    - General definition of sources and receivers
    - How to define the survey
    - Defining the model and all necessary mappings
    - How to predict total field, secondary field or ppm data
    - Defining models and mapping for dispersive conductivity and magnetic susceptibility

Our survey geometry consists of a vertical magnetic dipole source
located 1 m above the Earth's surface. The receiver is offset
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
# Here we demonstrate a general way to define sources and receivers.
# 

# Frequencies being observed
frequencies = np.logspace(-1, 8, 51)

# Define a list of receivers for each source. In this case we only have
# one source so we will only make one list.
receiver_location = np.array([10., 0., 1.])
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

# Define the source list.
source_location = np.array([0., 0., 1.])
source_orientation = 'z'                      # "x", "y" or "z"
moment_amplitude = 1.

source_list = [
    em1d.sources.MagneticDipoleSource(
        receiver_list=receiver_list, location=source_location,
        orientation=source_orientation, moment_amplitude=moment_amplitude
    )
]

# Survey
survey = em1d.survey.EM1DSurveyFD(source_list)


###############################################
# Defining a Layered Earth Model
# ------------------------------
#
# Here, we define the layer thicknesses and electrical resistivities for our
# 1D simulation. If we have N layers, we define N electrical resistivity
# values and N-1 layer thicknesses. The lowest layer is assumed to extend to
# infinity. In the case of a halfspace, the layer thicknesses would be
# an empty array.
#

# Layer thicknesses
thicknesses = np.array([20., 20.])
n_layer = len(thicknesses) + 1

# In SimPEG, the Cole-Cole model is used to define a frequency-dependent
# electrical conductivity when the Earth is chargeable. 
sigma = 1e-2
eta = 0.5
tau = 0.001
c = 0.5

# Magnetic susceptibility
chi = 0.2

# physical property models
sigma_model = sigma * np.ones(n_layer)
eta_model = eta * np.ones(n_layer)
tau_model =  tau * np.ones(n_layer)
c_model = c * np.ones(n_layer)
chi_model = chi * np.ones(n_layer)

# Define a mapping for conductivities
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
# Here we predict the FDEM sounding data. The simulation requires the user
# define the survey, the layer thicknesses and a mapping from the model
# to the conductivities of the layers.
# 
# For now, only the static conductivity and static susceptibility are
# invertible properties. Because of this, all other parameters defining
# dispersive physical properties are permanently set when defining the
# simulation.
#

# Simulate response for static conductivity
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
    eta=eta, tau=tau, c=c, chi=chi_model
)

dpred_chargeable = simulation_chargeable.dpred(sigma_model)


#######################################################################
# Plotting Results
# -------------------------------------------------
#

fig, ax = plt.subplots(1,1, figsize = (7, 7))
ax.loglog(frequencies, dpred[0:len(frequencies)], 'b-', lw=2)
ax.loglog(frequencies, dpred[len(frequencies):], 'b--', lw=2)
ax.loglog(frequencies, dpred_susceptible[0:len(frequencies)], 'r-', lw=2)
ax.loglog(frequencies, dpred_susceptible[len(frequencies):], 'r--', lw=2)
ax.loglog(frequencies, dpred_chargeable[0:len(frequencies)], 'g-', lw=2)
ax.loglog(frequencies, dpred_chargeable[len(frequencies):], 'g--', lw=2)
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("|H| (A/m)")
ax.set_title("Magnetic Field as a Function of Frequency")
ax.legend((
    'Real (conductive)', 'Imaginary (conductive)',
    'Real (susceptible)', 'Imaginary (susceptible)',
    'Real (chargeable)', 'Imaginary (chargeable)'
))

























