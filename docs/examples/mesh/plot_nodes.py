import numpy as np
import matplotlib.pyplot as plt
from SimPEG.mesh import TensorMesh

x0 = np.zeros(2)
h1 = np.linspace(.1,.5,3)
h2 = np.linspace(.1,.5,5)
M = TensorMesh([h1,h2],x0)
M.plotGrid()
plt.hold()
plt.plot(M.gridN[:,0], M.gridN[:,1], 'ks', markersize=10)
plt.show()

