#!/usr/bin/env python
# coding: utf-8

"""1d stitched inversion of time domain EM synthetic data using the static instrument framework
=============================================================================================


This tutorial shows how to use the static instrument framework for
inversion and forward modelling of synthetic time domain EM data.

The static instrument framework assumes that the instrument setup does
not change between sounding positions, and can therefore be described
separately from the data.

The framework uses a python class to describe the system, which is
then instantiated with the loaded data to perform an inversion, or
with a model to perform forward modelling. Note that the same system
description is used for both use cases!
"""


#########################################################################
# Import modules
# --------------

import matplotlib.pyplot as plt
import libaarhusxyz
import SimPEG.electromagnetics.utils.static_instrument
import SimPEG


#########################################################################
# Load the data from disk
# ------------------------
#
# Load the data from disk and plot it. The data is in a right handed
# coordinate system: x is forward, y to the right, z down

xyz = libaarhusxyz.XYZ("em1d_data.xyz")

plt.figure(figsize=(12, 8))
xyz.plot_line(0, ax=plt.gca())
plt.show()


#########################################################################
# Define the instrument
# ----------------------
#
# Here we only override a few default parameters, but any part of the
# system description could be overridden, including the construction
# of the `Survey` and `Simulation` objects, or the transmitter or
# receiver objects. For details, see the output of
# `help(SimPEG.electromagnetics.utils.static_instrument.SingleMomentTEMXYZSystem)`.

class MySystem(SimPEG.electromagnetics.utils.static_instrument.SingleMomentTEMXYZSystem):
    area=340
    i_max=1
    alpha_s = 1e-10
    alpha_r = 100.
    alpha_z = 1.


#########################################################################
# Do the inversion
# -----------------
#
# Here we combine the system description with the loaded data and
# perform an inversion.

inv = MySystem(xyz)
xyzsparse, xyzl2 = inv.invert()


#########################################################################
# Plot the results
# -----------------

fig = plt.figure(figsize=(12, 8))
ax=plt.gca()
xyzsparse.plot_line(0, ax=ax, cmap="jet")
fig.colorbar(mappable=ax.collections[0])
ax.set_ylim(-200, 0)

#########################################################################
# Forward modelling and comparison to original data
# -----------------
#
# Forward modelling is similar to inversion in that we combine a
# resistivity model with the same system description to form an object
# we can run the forward operator on.
#
# However, we also need to provide the gate times we want to model. To
# be able to compare the forward modelled data fromt this to the
# original data loaded above, we use the same times extracted from it.
#
# For now, xyzl2 is None due to IRLS being disabled by default. You
# can override `make_directives` in `MySystem` to change this.

#xyzl2fwd = MySystem(xyzl2, times=inv.times).forward()
xyzsparsefwd = MySystem(xyzsparse, times=inv.times).forward()


fig = plt.figure(figsize=(12, 8))
ax=plt.gca()
xyz.plot_line(0, ax=ax, c="red", label="measured %(gate)i @ %(time).2e")
#xyzl2fwd.plot_line(0, ax=ax, c="green", label="fwd %(gate)i @ %(time).2e")
xyzsparsefwd.plot_line(0, ax=ax, c="blue", label="fwd %(gate)i @ %(time).2e")
ax.legend([matplotlib.lines.Line2D([0], [0], color="red", lw=4),
           #matplotlib.lines.Line2D([0], [0], color="green", lw=4),
           matplotlib.lines.Line2D([0], [0], color="blue", lw=4)],
          ['Measured',
           #'L2',
           "Sparse"])
plt.show()
