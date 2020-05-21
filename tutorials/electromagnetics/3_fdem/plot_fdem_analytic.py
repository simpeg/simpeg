"""
Simulation with Analytic FDEM Solutions
=======================================

Here, the module *SimPEG.electromagnetics.analytics.FDEM* is used to simulate
electric and magnetic field for basic analytic solutions.


"""

#########################################################################
# Import modules
# --------------
#

import numpy as np
from SimPEG import utils
from SimPEG.electromagnetics.analytics.FDEM import (
    ElectricDipoleWholeSpace, MagneticDipoleWholeSpace)

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


#####################################################################
# Electric Field for a Harmonic Electric Dipole Source
# ----------------------------------------------------
#
#

# Defining electric dipole location and frequency 
source_location = np.r_[0, 0, 0]
frequency = 1e3

# Defining observation locations (avoid placing observation at source)
x = np.arange(-100.5, 100.5, step=1.)
y = np.r_[0]
z = x
observation_locations = utils.ndgrid(x, y, z)

# Define wholespace physical properties
sig = 1e-2
mu = 4*np.pi*1e-7

# Predict the fields
Ex, Ey, Ez = ElectricDipoleWholeSpace(
    observation_locations, source_location, sig, frequency,
    orientation='Z', mu=mu
)

absE = np.sqrt(Ex*Ex.conj()+Ey*Ey.conj()+Ez*Ez.conj()).real

fig, ax = plt.subplots(1, 1, figsize=(6, 5))
explt = Ex.reshape(x.size, z.size)
ezplt = Ez.reshape(x.size, z.size)
pc = ax.pcolor(x, z, absE.reshape(x.size, z.size), norm=LogNorm())
ax.streamplot(x, z, explt.real, ezplt.real, color='k', density=1)
ax.set_xlim([x.min(), x.max()])
ax.set_ylim([z.min(), z.max()])
ax.set_title('Electric Field from Electric Dipole Source')
ax.set_xlabel('x')
ax.set_ylabel('z')
cb = plt.colorbar(pc, ax=ax)
cb.set_label('|E| (V/m)')


#####################################################################
# Magnetic Fields for a Magnetic Dipole Source
# --------------------------------------------
#
#

# Defining electric dipole location and frequency 
source_location = np.r_[0, 0, 0]
frequency = 1e3

# Defining observation locations (avoid placing observation at source)
x = np.arange(-100.5, 100.5, step=1.)
y = np.r_[0]
z = x
observation_locations = utils.ndgrid(x, y, z)

# Define wholespace physical properties
sig = 1e-2
mu = 4*np.pi*1e-7

# Predict the fields
Bx, By, Bz = MagneticDipoleWholeSpace(
    observation_locations, source_location, sig, frequency,
    orientation='Z', mu=mu
)

absB = np.sqrt(Bx*Bx.conj()+By*By.conj()+Bz*Bz.conj()).real

fig, ax = plt.subplots(1, 1, figsize=(6, 5))
bxplt = Bx.reshape(x.size, z.size)
bzplt = Bz.reshape(x.size, z.size)
pc = ax.pcolor(x, z, absB.reshape(x.size, z.size), norm=LogNorm())
ax.streamplot(x, z, bxplt.real, bzplt.real, color='k', density=1)
ax.set_xlim([x.min(), x.max()])
ax.set_ylim([z.min(), z.max()])
ax.set_xlabel('x')
ax.set_ylabel('z')
ax.set_title('Magnetic Field from Magnetic Dipole Source')
cb = plt.colorbar(pc, ax=ax)
cb.set_label('|B| (T)')


