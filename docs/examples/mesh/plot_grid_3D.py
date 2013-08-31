import numpy as np
import matplotlib.pyplot as plt
from SimPEG import TensorMesh

h1 = np.linspace(.1,.5,3)
h2 = np.linspace(.1,.5,5)
h3 = np.linspace(.1,.5,3)
mesh = TensorMesh([h1,h2,h3])
mesh.plotGrid()

plt.show()

