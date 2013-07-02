#-------------------------------------------------------------------------------
# Packages
#-------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#-------------------------------------------------------------------------------
# Class definition
#-------------------------------------------------------------------------------
class TensorMesh:
    """ 
    Define nodal, cell-centered and staggered tensor meshes  for 1, 2 and 3 
    dimensions.    
    """
    
    # ----------------------  Properties --------------------------------------
    h   = None          # Array        Spacing or cell-sizes in each direction
    x0  = None          # Array        Origin  (x1,x2,x3)
    dim = None          # Int          Dimension   
    n   = None          # Array        Number of cells in each direction
    nC  = None          # Int          Total number of cells   
    nE  = None          # Int          Total number of edges
    nF  = None          # Int          Total number of faces
    
    # -----------------------  Methods  ----------------------------------------
    def __init__(self,h,x0): 
        """Compute number of edges,faces and cell-centers """
                        
        # Assign values to properties
        self.h   = h
        self.x0  = x0
        
        
        #Compute derived properties
        self.dim = np.size(x0)
        
        #Compute the num of cells in each direction
        self.n = np.zeros((self.dim,1))
        
        for d in range(self.dim):
            self.n[d] = np.size(h[d])
        
        # Compute the number of cell-centers    
        self.nC = np.prod(self.n)
        
        # Compute the number of edges  (makes sense only for 3D)
        # Equivalent to:  
        # nEdges             =   n[0]    * (n[1]+1) * (n[2]+1)
        #                     + (n[0]+1) *  ny[1]   * (nz[2]+1)
        #                     + (n[0]+1) * (ny[1]+1)* (nz[2])
        
        if self.dim == 3:
            self.nE = np.prod(np.kron(np.ones((3,1)),self.n.T)+np.ones((3,3))-np.eye(3),1)
            print self.nE
        
        # Compute the number of faces  (makes sense only for 2 and 3D)
        # Equivalent to
        # nFaces             =   (n[0]+1)    * n[1]        * n[2]
        #                     +   n[0]       * (ny[1]+1)   * nz[2]
        #                     +   n[0]       * ny[1]       * (nz[2]+1)
        if self.dim >=2:
            self.nF = np.prod(np.kron(np.ones((self.dim,1)),self.n.T)+np.eye(self.dim),1)
            print self.nF

    def xin(self,i):
        """Construct the 1D nodal mesh from the ith-component of h. Return an array."""
        return np.insert(np.cumsum(self.h[i-1]),0,0.0) + self.x0[i-1]
        
    def xic(self,i):
        """Construct the 1D cell-centerd mesh from the ith-component of h. Return an array."""
        return .5*( np.insert(np.cumsum(self.h[i-1][:,0:-1]),0,0.0) + np.cumsum(self.h[i-1]))         

    def getNodalGrid(self):
        """Construct nodal grid for 1, 2 and 3 dimensions"""
        if self.dim==1:
            return [self.xin(1)]
        elif self.dim==2:
            return self.ndgrid([self.xin(1),self.xin(2)])
        elif self.dim==3:
            return self.ndgrid([self.xin(1),self.xin(2),self.xin(3)])
           
    def ndgrid(self, xin):
        """Form tensorial grid for 1, 2 and 3 dimensions. Return X1,X2,X3 arrays depending on the dimension"""
        ei = lambda i : np.ones((np.size(xin[i-1]),1))
        
        if self.dim==1:
            return [xin]
        elif self.dim==2:
            X1 = np.kron(ei(2),xin[0]).reshape(-1,1)
            X2 = np.kron(xin[1],ei(1).T).reshape(-1,1)
            return X1,X2
        elif self.dim==3:
            X1 = np.kron(ei(3),np.kron(ei(2),xin[0])).reshape(-1,1)
            X2 = np.kron(ei(3).T,np.kron(xin[1],ei(1).T)).reshape(-1,1)
            X3 = np.kron(xin[2],np.kron(ei(2),ei(1))).T.reshape(-1,1)
            
            return X1,X2,X3    
    
    def getCellCenteredGrid(self):
        """Construct cell-centered grid for 1, 2 and 3 dimensions."""
        
        if self.dim==1:
            return [self.xic(1)]
        elif self.dim==2:
            return self.ndgrid([self.xic(1),self.xic(2)])
        elif self.dim==3:
            return self.ndgrid([self.xic(1),self.xic(2),self.xic(3)])
       
    def getFaceStgGrid(self,direction):
        """Construct the face staggered grids for 2 and 3 dimensions."""

        if self.dim==1:
            print 'Error: dimension must be larger than 1'
        elif self.dim==2:
            if direction == 1:
                return self.ndgrid([self.xin(1),self.xic(2)])
            elif direction  == 2:
                return self.ndgrid([self.xic(1),self.xin(2)])
            else:
                print 'Error:  direction must be equal to 1 or 2'
        elif self.dim==3:
            if direction == 1:
                return self.ndgrid([self.xin(1),self.xic(2),self.xic(3)])
            elif direction == 2:
                return self.ndgrid([self.xic(1),self.xin(2),self.xic(3)])
            elif direction == 3:
                return self.ndgrid([self.xic(1),self.xic(2),self.xin(3)])
            else:
                print 'Error:  direction must be equal to 1, 2 or 3'
                
    def getEdgeStgGrid(self,direction):
        """Construct the edge staggered grids for 3 dimension case."""
        if self.dim != 3:
            print 'Error: dimension must be equal to 3'
        else:
            if  direction == 1:
                return self.ndgrid([self.xic(1),self.xin(2),self.xin(3)])
            elif direction == 2:
                return self.ndgrid([self.xin(1),self.xic(2),self.xin(3)])
            elif direction == 3:
                return self.ndgrid([self.xin(1),self.xin(2),self.xic(3)])
            else:
                print 'Error:  direction must be equal to 1, 2 or 3'
    
    def plotImage(self,I):
    
        if self.dim==1:
            fig = plt.figure(1)
            fig.clf()
            ax=plt.subplot(111) 
            if np.size(I)==self.n[0]:
                print 'cell-centered image'
                xx = self.getCellCenteredGrid()
                ax.plot(xx[0],I,'ro')
            elif np.size(I)==self.n[0]+1:
                print 'nodal image'
                xx = self.getNodalGrid()
                ax.plot(xx[0],I,'bs')
            
            fig.show()
            
    
    def plotGrid(self):
        """Plot the nodal, cell-centered and staggered grids for 1,2 and 3 dimensions."""
        if self.dim == 1:
            fig = plt.figure(1)
            fig.clf()
            ax = plt.subplot(111) 
            xn = self.getNodalGrid()
            xc = self.getCellCenteredGrid()
            print xn
            ax.hold(True)
            ax.plot(xn,np.ones(np.shape(xn)),'bs')
            ax.plot(xc,np.ones(np.shape(xc)),'ro')
            ax.plot(xn,np.ones(np.shape(xn)),'k--')
            ax.grid(True)
            ax.hold(False)
            ax.set_xlabel('x1')
            fig.show()
        elif self.dim == 2:
            fig = plt.figure(2)
            fig.clf()
            ax = plt.subplot(111) 
            xn = self.getNodalGrid()
            xc = self.getCellCenteredGrid()
            xs1 = self.getFaceStgGrid(1)
            xs2 = self.getFaceStgGrid(2)
            
            ax.hold(True)
            ax.plot(xn[0],xn[1],'bs')
            ax.plot(xc[0],xc[1],'ro')
            ax.plot(xs1[0],xs1[1],'g>')
            ax.plot(xs2[0],xs2[1],'g^')
            ax.grid(True)
            ax.hold(False)
            ax.set_xlabel('x1')
            ax.set_ylabel('x2')
            fig.show()
        elif self.dim == 3:
            fig = plt.figure(3)
            fig.clf()
            ax = fig.add_subplot(111, projection='3d')
            xn = self.getNodalGrid()
            xc = self.getCellCenteredGrid()
            xfs1 = self.getFaceStgGrid(1)
            xfs2 = self.getFaceStgGrid(2)
            xfs3 = self.getFaceStgGrid(3)
            xes1 = self.getEdgeStgGrid(1)
            xes2 = self.getEdgeStgGrid(2)
            xes3 = self.getEdgeStgGrid(3)
            
            ax.hold(True)
            ax.plot(xn[0],xn[1],'bs',zs=xn[2])
            ax.plot(xc[0],xc[1],'ro',zs=xc[2])
            ax.plot(xfs1[0],xfs1[1],'g>',zs=xfs1[2])
            ax.plot(xfs2[0],xfs2[1],'g<',zs=xfs2[2])
            ax.plot(xfs3[0],xfs3[1],'g^',zs=xfs3[2])
            ax.plot(xes1[0],xes1[1],'k>',zs=xes1[2])
            ax.plot(xes2[0],xes2[1],'k<',zs=xes2[2])
            ax.plot(xes3[0],xes3[1],'k^',zs=xes3[2])
            ax.grid(True)
            ax.hold(False)
            ax.set_xlabel('x1')
            ax.set_ylabel('x2')
            ax.set_zlabel('x3')
            fig.show()
                
      
      
if __name__ == '__main__':
    print('Welcome to tensor mesh!')
    
    testDim = 1
    h1 = 0.3*np.ones((1,7))
    h1[:,0]  = 0.5
    h1[:,-1] = 0.6
    h2 = .5* np.ones((1,4))
    h3 = .4* np.ones((1,6))
    x0 = np.zeros((3,1))
    
    if testDim == 1:
        h = [h1]
        x0 = x0[0]  
    elif testDim==2:
        h = [h1,h2]
        x0 = x0[0:2]
    else:
        h = [h1,h2,h3]
    
    I = np.linspace(0,1,8)    
    M = TensorMesh(h,x0)   
    
    xn = M.plotGrid()  
      
   
    