import numpy as np

import sys
sys.path.append('../')
from TensorMesh import TensorMesh


err=0.
print '>> Test face Divergence operator'
for i in range(4):
    icount=i+1
    nc = 2**icount
    # Define the mesh
    h1 = np.ones((1,nc))/nc
    h2 = np.ones((1,nc))/nc
    h3 = np.ones((1,nc))/nc
    h = [h1, h2, h3]
    x0 = np.zeros((3, 1))
    M = TensorMesh(h, x0)
    #n = M.plotGrid()

    # Generate DIV matrix
    DIV = M.faceDiv

    #Test function
    fun = lambda x: np.sin(x)
    Fx = fun(M.gridFx[:,0])
    Fy = fun(M.gridFy[:,1])
    Fz = fun(M.gridFz[:,2])

    F = np.concatenate((Fx,Fy,Fz))
    divF = DIV*F
    sol = lambda x, y, z: (np.cos(x)+np.cos(y)+np.cos(z))
    divF_anal = sol(M.gridCC[:,0], M.gridCC[:,1], M.gridCC[:,2])

    #err = np.linalg.norm((divF-divF_anal)*np.sqrt(vol), 2)
    err = np.linalg.norm((divF-divF_anal), np.inf)

    if icount == 1:
        print 'h       |   inf norm   | error ratio'
        print '---------------------------------------'
        print '%6.4f  |  %8.2e  |'% (h1[0,0], err)
    else:
        print '%6.4f  |  %8.2e  |  %6.4f' % (h1[0,0], err, err_old/err)
    err_old = err
