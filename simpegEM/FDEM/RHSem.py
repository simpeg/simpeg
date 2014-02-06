import numpy as np

def ismember(a, b):
    tf = np.array([i in b for i in a])
    return tf

def path2edgeModel(mesh, pts):
    edm_x = np.zeros(np.prod(mesh.nEx))
    edm_y = np.zeros(np.prod(mesh.nEy))
    edm_z = np.zeros(np.prod(mesh.nEz))
    
    for ii in range (pts.shape[0]-1):
        pt1 = pts[ii,:]
        pt2 = pts[ii+1,:]
        delta = pt2 - pt1
        deltaDim = np.argwhere(delta)
        #assert(np.size(deltaDim)==1), "Path must be orthoginal to mesh"
        if deltaDim == 0:
            xLoc = mesh.vectorCCx[(min(pt1[0],pt2[0]) < mesh.vectorCCx ) & (mesh.vectorCCx < max(pt1[0],pt2[0]))]
            yLoc = pts[ii,1]
            zLoc = pts[ii,2]
            delDir = np.sign(pt2[0]-pt1[0])
            xyz = np.c_[xLoc, np.ones(np.size(xLoc))*yLoc, np.ones(np.size(xLoc))*zLoc]
            edgeInd=ismember(map(tuple,mesh.gridEx),map(tuple,xyz))
            edm_x[edgeInd] = delDir
            # print '>> x-direction', ii
            # print mesh.gridEx[edgeInd]
        if deltaDim == 1:    
            xLoc = pts[ii,0]
            yLoc = mesh.vectorCCy[(min(pt1[1],pt2[1]) < mesh.vectorCCy ) & (mesh.vectorCCy < max(pt1[1],pt2[1]))]
            zLoc = pts[ii,2]
            delDir = np.sign(pt2[1]-pt1[1])
            xyz = np.c_[np.ones(np.size(yLoc))*xLoc, yLoc, np.ones(np.size(yLoc))*zLoc]
            edgeInd=ismember(map(tuple,mesh.gridEy),map(tuple,xyz))
            edm_y[edgeInd] = delDir
            # print '>> y-direction', ii
            # print mesh.gridEy[edgeInd]
        if deltaDim == 2:    
            xLoc = pts[ii,0]
            yLoc = pts[ii,1]
            zLoc = mesh.vectorCCz[(min(pt1[2],pt2[2]) < mesh.vectorCCz ) & (mesh.vectorCCz < max(pt1[2],pt2[2]))]
            delDir = np.sign(pt2[2]-pt1[2])
            xyz = np.c_[np.ones(np.size(zLoc))*xLoc, np.ones(np.size(zLoc))*yLoc, zLoc]
            edgeInd=ismember(map(tuple,mesh.gridEz),map(tuple,xyz))
            edm_z[edgeInd] = delDir
            # print '>> z-direction', ii
            # print mesh.gridEz[edgeInd]
    
            
    edgeModel = np.r_[edm_x, edm_y, edm_z]
    return edgeModel

def rho(x1, y1, x, y):
    r = np.sqrt((x-x1)**2+(y-y1)**2)
    return r

def MMRhalf(loc1, loc2, x, y):
    """ Anaytic function for MMR response (B^{1D})
            - loc1=(x1,y1): x, y location for (+) charge
            - loc2=(x2,y2): x, y1
            - x : observation points in x-direction
            - y : observation points in y-direction
    """
    x1=loc1[0]
    x2=loc2[0]
    y1=loc1[1]
    y2=loc2[1]
    mu0 = 4*np.pi*1e-7
    I = 1
    By =mu0*I/(4*np.pi)*np.array((x-x1)/rho(x1,y1,x,y)**2-(x-x2)/rho(x2,y2,x,y)**2)  
    Bx =mu0*I/(4*np.pi)*np.array(-(y-y1)/rho(x1,y1,x,y)**2+(y-y2)/rho(x2,y2,x,y)**2)  
    
    return Bx, By