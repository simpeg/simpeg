#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import libaarhusxyz
import SimPEG.electromagnetics.utils.static_instrument
import SimPEG


# Right handed coordinate system. x is forward, y to the right, z down

#### Load the data from disk ####

xyz = libaarhusxyz.XYZ("em1d_data.xyz")

plt.figure(figsize=(12, 8))
xyz.plot_line(0, ax=plt.gca())
plt.show()


#### Define the instrument ####

class MySystem(SimPEG.electromagnetics.utils.static_instrument.SingleMomentTEMXYZSystem):
    area=340
    i_max=1
    alpha_s = 1e-10
    alpha_r = 100.
    alpha_z = 1.


#### Do the inversion ####

inv = MySystem(xyz)
xyzsparse, xyzl2 = inv.invert()


#### Plot the results ####


fig = plt.figure(figsize=(12, 8))
ax=plt.gca()
xyzsparse.plot_line(0, ax=ax, cmap="jet")
fig.colorbar(mappable=ax.collections[0])
ax.set_ylim(-200, 0)


#### Forward modelling and comparison to original data ####

#xyzl2fwd = MySystem(xyzl2, times=inv.times).forward()
xyzsparsefwd = MySystem(xyzsparse, times=invtimes).forward()


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
