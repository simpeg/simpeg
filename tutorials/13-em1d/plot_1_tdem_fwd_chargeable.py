"""
Forward Simulation over a Chargeable Earth
==========================================





"""

#####################################################
# Import Modules
# --------------
#

import numpy as np
from matplotlib import pyplot as plt

from SimPEG import maps
import SimPEG.electromagnetics.time_domain_1d as em1d
from SimPEG.electromagnetics.utils.em1d_utils import ColeCole, LogUniform
from discretize.utils import mkvc

from scipy.special import expi

#####################################################################
# Create Survey
# -------------
#
#

source_location = np.array([0., 0., 0.5])  
source_orientation = "z"  # "x", "y" or "z"
source_current = 1.
source_radius = 10.

receiver_location = np.array([0., 0., 0.5])
receiver_orientation = "z"  # "x", "y" or "z"
field_type = "secondary"  # "secondary", "total" or "ppm"

times = np.logspace(-6, -1, 51)

# Receiver list
receiver_list = []
receiver_list.append(
    em1d.receivers.PointReceiver(
        receiver_location, times, orientation=receiver_orientation,
        component="b"
    )
)
receiver_list.append(
    em1d.receivers.PointReceiver(
        receiver_location, times, orientation=receiver_orientation,
        component="dbdt"
    )
)

# Sources
source_list = [
    em1d.sources.HorizontalLoopSource(
        receiver_list=receiver_list, location=source_location,
        I=source_current, a=source_radius
    )
]


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

# half-space conductivity properties
sigma = 1e-1
eta = 0.5
tau = 0.01
c = 0.75

# half-space magnetic viscosity properties
chi = 0.001
dchi = 0.001
tau1 = 1e-7
tau2 = 1.


# physical property models
sigma_model = sigma * np.ones(n_layer)
eta_model = eta * np.ones(n_layer)
tau_model =  tau * np.ones(n_layer)
c_model = c * np.ones(n_layer)
chi_model = chi * np.ones(n_layer)
dchi_model = dchi * np.ones(n_layer)
tau1_model = tau1 * np.ones(n_layer)
tau2_model = tau2 * np.ones(n_layer)

# Define a mapping for conductivities
model_mapping = maps.IdentityMap(nP=n_layer)

# Compute and plot complex conductivity at all frequencies
frequencies = np.logspace(-3, 6, 91)
sigma_complex = ColeCole(frequencies, sigma, eta, tau, c)
chi_complex = LogUniform(frequencies, chi, dchi, tau1, tau2)

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
    ["$\sigma_{DC}$", "$Re[\sigma (\omega)]$", "$Im[\sigma (\omega)]$"],
    loc="center right"
)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.semilogx(frequencies, chi*np.ones(len(frequencies)), "b", lw=3)
ax.semilogx(frequencies, np.real(chi_complex), "r", lw=3)
ax.semilogx(frequencies, np.imag(chi_complex), "r--", lw=3)
ax.set_xlim(np.min(frequencies), np.max(frequencies))
ax.set_ylim(-1.1*chi, 1.1*(chi+dchi))
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Susceptibility")
ax.legend(
    ["$\chi_{DC}$", "$Re[\chi (\omega)]$", "$Im[\chi (\omega)]$"],
    loc="center right"
)

#######################################################################
# Define the Forward Simulation and Predict Data
# ----------------------------------------------
#


# Simulate response for static conductivity
simulation_1 = em1d.simulation.EM1DTMSimulation(
    survey=survey, thicknesses=thicknesses, sigmaMap=model_mapping,
    chi=chi_model
)

dpred_1 = simulation_1.dpred(sigma_model)

# Simulate response for complex conductivity
simulation_2 = em1d.simulation.EM1DTMSimulation(
    survey=survey, thicknesses=thicknesses, sigmaMap=model_mapping,
    eta=eta, tau=tau, c=c, chi=chi_model
)

dpred_2 = simulation_2.dpred(sigma_model)

# Simulate response for viscous remanent magnetization
simulation_3 = em1d.simulation.EM1DTMSimulation(
    survey=survey, sigmaMap=maps.IdentityMap(nP=1),
    chi=chi, dchi=dchi, tau1=tau1, tau2=tau2,
)

# m = mkvc(np.array(sigma))
m = mkvc(np.array(1e-6))
dpred_3 = simulation_3.dpred(m)


############################################
# ANALYTIC


F = (1/np.log(tau2/tau1)) * (expi(times/tau2) + expi(-times/tau1))

dFdt = (1/np.log(tau2/tau1)) * (np.exp(-times/tau1) - np.exp(-times/tau2)) / times


mu0 = 4*np.pi*1e-7
a = source_radius
z = 0.5
h = 0.5
B0 = (0.5*mu0*a**2) * (dchi/(2 + dchi)) * ((z + h)**2 + a**2)**-1.5


Banal = B0*F
dBdtanal = B0*dFdt







############################################


fig = plt.figure(figsize = (6, 5))
ax = fig.add_axes([0.1, 0.1, 0.8, 0.85])
ax.loglog(times, np.abs(dpred_1[0:len(times)]), 'k', lw=3)
ax.loglog(times, np.abs(dpred_2[0:len(times)]), 'r', lw=3)
ax.loglog(times, np.abs(dpred_3[0:len(times)]), 'b', lw=3)
ax.loglog(times, np.abs(Banal), 'b*')
ax.legend(["Purely Inductive", "Chargeable", "Viscous Remanent Mag."])
ax.set_xlabel("Times (s)")
ax.set_ylabel("|B| (T)")
ax.set_title("Magnetic Flux")

fig = plt.figure(figsize = (6, 5))
ax = fig.add_axes([0.1, 0.1, 0.8, 0.85])
ax.loglog(times, np.abs(dpred_1[len(times):]), 'k', lw=3)
ax.loglog(times, np.abs(dpred_2[len(times):]), 'r', lw=3)
ax.loglog(times, np.abs(dpred_3[len(times):]), 'b', lw=3)
ax.loglog(times, np.abs(dBdtanal), 'b*')
ax.legend(["Purely Inductive", "Chargeable", "Viscous Remanent Mag."])
ax.set_xlabel("Times (s)")
ax.set_ylabel("|dB/dt| (T/s)")
ax.set_title("Time-Derivative of Magnetic Flux")

##################################################

























