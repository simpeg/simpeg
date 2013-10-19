import numpy as np
import matplotlib.pyplot as plt
from SimPEG.mesh import TensorMesh

n = 20
h = np.ones(n)/n
M = TensorMesh([h,h,h])

I = np.sin(M.gridCC[:,0]*2*np.pi)*np.sin(M.gridCC[:,1]*2*np.pi)*np.sin(M.gridCC[:,2]*2*np.pi)
M.plotImage(I, annotationColor='k')

plt.show()
