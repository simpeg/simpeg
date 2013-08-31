import numpy as np
import matplotlib.pyplot as plt
from SimPEG import TensorMesh

h1 = np.linspace(.1,.5,3)
h2 = np.linspace(.1,.5,5)
mesh = TensorMesh([h1, h2])
mesh.plotGrid()

plt.show()

