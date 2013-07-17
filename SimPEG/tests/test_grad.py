import numpy as np

import sys
sys.path.append('../')
from TensorMesh import TensorMesh
from getDiffop import getGradMatrix


err=0.
print '>> Test nodal Gradient operator'    
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
    GRAD = getGradMatrix(h)
    #Test function
    fun = lambda x, y, z: (np.cos(x)+np.cos(y)+np.cos(z))    
    sol = lambda x: -np.sin(x) # i (sin(x)) + j (sin(y)) + k (sin(z))
    
    phi = fun(M.gridN[:,0], M.gridN[:,1], M.gridN[:,2])
    gradE = GRAD*phi

    Ex = sol(M.gridEx[:,0])
    Ey = sol(M.gridEy[:,1])
    Ez = sol(M.gridEz[:,2])

    gradE_anal = np.concatenate((Ex,Ey,Ez))     
    err = np.linalg.norm((gradE-gradE_anal), np.inf)

    if icount == 1:
        print 'h       |   inf norm   | error ratio'        
        print '---------------------------------------'                
        print '%6.4f  |  %8.2e  |'% (h1[0,0], err)
    else:
        print '%6.4f  |  %8.2e  |  %6.4f' % (h1[0,0], err, err_old/err)
    err_old = err    
        
