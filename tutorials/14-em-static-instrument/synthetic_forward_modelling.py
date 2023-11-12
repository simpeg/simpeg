#!/usr/bin/env python
# coding: utf-8

"""Forward modelling of time domain EM synthetic data using the static instrument framework
=============================================================================================

This tutorial shows how to generate simple synthetic time domain EM
data using the static instrument framework for forward modelling.

The static instrument framework assumes that the instrument setup does
not change between sounding positions, and can therefore be described
separately from the data.

The framework uses a python class to describe the system, which is
then instantiated with the generated data to perform a forward modelling.

"""


#########################################################################
# Import modules
# --------------

import matplotlib.pyplot as plt
import copy
import libaarhusxyz
import SimPEG.electromagnetics.utils.static_instrument


#########################################################################
# Make a synthetic model
# -----------------------
#
# Make a synthetic model and plot it. The data is in a right handed
# coordinate system: x is forward, y to the right, z down

xyz = SimPEG.electromagnetics.utils.static_instrument.make_2layer(
    xdist=np.arange(300)*50,
    dtb=np.linspace(10, 200, 300),
    layers=np.concatenate(([0], np.logspace(0, 2.5, 25), [np.inf])))

xyz.flightlines["elevation"] = np.zeros(len(xyz.flightlines))
xyz.flightlines["alt"] = 30


fig = plt.figure()
ax = plt.gca()
xyz.plot_line(0, ax=ax)
ax.plot(xyz.flightlines.xdist, xyz.flightlines.elevation - xyz.flightlines.interface_depth)
plt.show()


#########################################################################
# Define the instrument
# ----------------------
#
# Here we only override a few default parameters and provide the gate
# times we want to model, but any part of the system description could
# be overridden, including the construction of the `Survey` and
# `Simulation` objects, or the transmitter or receiver objects. For
# details, see the output of
# `help(SimPEG.electromagnetics.utils.static_instrument.SingleMomentTEMXYZSystem)`.


class MySystem(SimPEG.electromagnetics.utils.static_instrument.SingleMomentTEMXYZSystem):
    area=340
    i_max=1
    times_full = [np.logspace(-5.5, -2, 16)]


#########################################################################
# Do the forward modelling
# ----------------------
#
# Here we combine the system description with the generated model and
# perform a forward modelling.

fwd = MySystem(xyz)
xyzresp = fwd.forward()


#########################################################################
# Make some realistic data with noise and uncertainties
# ----------------------
#
# We add 5 percent noise to the modelled data, and set the
# uncertainties to match that.

xyzmeasured = copy.deepcopy(xyzresp)
SimPEG.electromagnetics.utils.static_instrument.add_noise(xyzmeasured, 0.05)
SimPEG.electromagnetics.utils.static_instrument.add_uncertainty(xyzmeasured, rel_uncertainty=0.05)


#########################################################################
# Plot the forward modelled data
# -------------------------------
# 
# Plot the forward modelled data, with and without the synthetic noise
# we added.

plt.figure(figsize=(12, 8))
xyzresp.plot_line(0, plt.gca(), label="FWDdata %(gate)i @ %(time).2e")
xyzmeasured.plot_line(0, plt.gca(), label="FWDdata + noise %(gate)i @ %(time).2e")
plt.legend()
plt.show()

#########################################################################
# Write data to files
# --------------------
#
# Write the data with noise to file so it can be read by
# `synthetic_inversion.py` to perform inversion on.

xyzmeasured.dump("em1d_data.xyz")
