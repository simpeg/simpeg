import numpy as np

import sys
sys.path.append('../')
from TensorMesh import TensorMesh
from getDIV import getDivMatrix, getarea, getvol

# Define the mesh
err=0.
for i in range(4):
    icount=i+1;
    nc = 2*icount;
    h1 = np.pi/nc*np.ones((1,nc))
    h2 = np.pi/nc*np.ones((1,nc))
    h3 = np.pi/nc*np.ones((1,nc))
    h = [h1, h2, h3]
    x0 = -np.pi/2*np.ones((3, 1))
    M = TensorMesh(h, x0)
    #n = M.plotGrid()

    # Generate DIV matrix
    DIV = getDivMatrix(h)
    
    #Test function
    fun = lambda x: np.sin(x)
    Fx = fun(M.gridFx[:,0])
    Fy = fun(M.gridFy[:,1])
    Fz = fun(M.gridFz[:,2])
    
    F = np.concatenate((Fx,Fy,Fz))
    divF = DIV*F
    sol = lambda x, y, z: (np.cos(x)+np.cos(y)+np.cos(z))
    divF_anal = sol(M.gridCC[:,0], M.gridCC[:,1], M.gridCC[:,2])
     
    area = getarea(h)
    vol = getvol(h)
    err = np.linalg.norm((divF-divF_anal)*np.sqrt(vol), 2)
    if icount == 1:
        err1 = err
        print 'h       |   2 norm   | error ratio'        
        print '---------------------------------------'                
        print '%6.4f  |  %8.2e  |'% (h1[0,0], err)
    else:
        print '%6.4f  |  %8.2e  |  %6.4f' % (h1[0,0], err, err1/err)
        
