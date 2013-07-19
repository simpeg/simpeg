import numpy as np

import sys
sys.path.append('../')
from TensorMesh import TensorMesh


err=0.
print '>> Test Curl operator'
for i in range(4):
    icount=i+1
    nc = 2**icount
    # Define the mesh
    h1 = np.ones(nc)/nc
    h2 = np.ones(nc)/nc
    h3 = np.ones(nc)/nc
    h = [h1, h2, h3]
    M = TensorMesh(h)
    #n = M.plotGrid()

    # Generate DIV matrix
    CURL = M.edgeCurl
    #Test function
    fun = lambda x: np.cos(x)  # i (cos(y)) + j (cos(z)) + k (cos(x))
    sol = lambda x: np.sin(x)  # i (sin(z)) + j (sin(x)) + k (sin(y))

    Ex = fun(M.gridEx[:,1])
    Ey = fun(M.gridEy[:,2])
    Ez = fun(M.gridEz[:,0])
    E = np.concatenate((Ex,Ey,Ez))

    Fx = sol(M.gridFx[:,2])
    Fy = sol(M.gridFy[:,0])
    Fz = sol(M.gridFz[:,1])
    curlE_anal = np.concatenate((Fx,Fy,Fz))

    curlE = CURL*E
    err = np.linalg.norm((curlE-curlE_anal), np.inf)

    if icount == 1:
        print 'h       |   inf norm   | error ratio'
        print '---------------------------------------'
        print '%6.4f  |  %8.2e  |'% (h1[0], err)
    else:
        print '%6.4f  |  %8.2e  |  %6.4f' % (h1[0], err, err_old/err)
    err_old = err