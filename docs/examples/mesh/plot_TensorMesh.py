import numpy as np
import matplotlib.pyplot as plt
from SimPEG.mesh import TensorMesh

pad = 7
padfactor = 1.4
xpad = (np.ones(pad)*padfactor)**np.arange(pad)
ypad = (np.ones(pad)*padfactor)**np.arange(pad)

core = 15
xcore = np.ones(core)
ycore = np.ones(core)

h1 = np.r_[xpad[::-1],xcore,xpad]
h2 = np.r_[ypad[::-1],ycore,ypad]

mesh = TensorMesh([h1, h2])
mesh.plotGrid()
plt.axis('tight')

plt.show()
