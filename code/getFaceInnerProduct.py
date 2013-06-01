from scipy.sparse import linalg
from scipy import sparse
from sputils import *
from utils import *
from numpy import *
from getEdgeTangent import *
from inv3X3BlockDiagonal import *
from getCellVolume import getCellVolume
from getFaceNormals import getFaceNormals


#-----------------------
def subarray(T,i1,i2,i3):
    return take(take(take(T,i1,0),i2,1),i3,2)

#-----------------------

def getFaceInnerProduct(X,Y,Z,sigma):

    m = array(shape(X))-1
    nc = prod(m)
    mf1 = m+[1, 0, 0]
    mf2 = m+[0, 1, 0]
    mf3 = m+[0, 0, 1]
    
    nf1 = prod(m+[1, 0, 0])
    nf2 = prod(m+[0, 1, 0])
    nf3 = prod(m+[0, 0, 1])
    
    # compute the normals 
    n1x,n1y,n1z,n2x,n2y,n2z,n3x,n3y,n3z,area1,area2,area3 = getFaceNormals(X,Y,Z)
    
    i = int64(linspace(0,m[0]-1,m[0]))
    j = int64(linspace(0,m[1]-1,m[1]))
    k = int64(linspace(0,m[2]-1,m[2]))
        
    ii,jj,kk = ndgrid(i,j,k) 
    ii = mkvc(ii); jj = mkvc(jj); kk = mkvc(kk)
        
    ind1 = sub2ind(mf1,hstack3(ii,jj,kk)) 
    ind2 = sub2ind(mf2,hstack3(ii,jj,kk)) + nf1
    ind3 = sub2ind(mf3,hstack3(ii,jj,kk)) + nf1 + nf2
        
    IND = vstack((vstack((ind1,ind2)),ind3))
    IND = array(IND).flatten()
        
    P1 = sparse.coo_matrix((ones(3*nc),(linspace(0,3*nc-1,3*nc),IND)),shape=(3*nc,nf1+nf2+nf3)).tocsr()
        
    ind1 = sub2ind(mf1,hstack3(ii+1,jj,kk)) 
    ind2 = sub2ind(mf2,hstack3(ii,jj+1,kk)) + nf1
    ind3 = sub2ind(mf3,hstack3(ii,jj,kk+1)) + nf1 + nf2
        
    IND = vstack((vstack((ind1,ind2)),ind3))
    IND = array(IND).flatten()
    
    P2 = sparse.coo_matrix((ones(3*nc),(linspace(0,3*nc-1,3*nc),IND)),shape=(3*nc,nf1+nf2+nf3)).tocsr()
    
    
    invN1 = inv3X3BlockDiagonal(subarray(n1x,i,j,k) , subarray(n1y,i,j,k), subarray(n1z,i,j,k),
                                subarray(n2x,i,j,k) , subarray(n2y,i,j,k), subarray(n2z,i,j,k),
                                subarray(n3x,i,j,k) , subarray(n3y,i,j,k), subarray(n3z,i,j,k) )
    
    
    invN2 = inv3X3BlockDiagonal(subarray(n1x,i+1,j,k) , subarray(n1y,i+1,j,k), subarray(n1z,i+1,j,k),
                                subarray(n2x,i,j+1,k) , subarray(n2y,i,j+1,k), subarray(n2z,i,j+1,k),
                                subarray(n3x,i,j,k+1) , subarray(n3y,i,j,k+1), subarray(n3z,i,j,k+1) )
    
    # Cell volume
    v  = mkvc(getCellVolume(X,Y,Z)) #mkvc(getVolume(X,Y,Z))
    vsig = v*mkvc(sigma)
    v3 = vstack((vstack((vsig,vsig)),vsig))
    v3 = v3.flatten()
        
    V = sdiag(v3)
    
    return (P1.T*invN1.T*V*invN1*P1 + P2.T*invN2.T*V*invN2*P2)/2.0
 

if __name__ == '__main__':

    X,Y,Z = ndgrid(linspace(0,2,3),linspace(0,2,3),linspace(0,2,3))
    Z[2,2,2] = 2.5;   Z[0,0,0] = -0.5
    X[2,2,2] = 2.5; X[0,0,0] = -0.5
    sigma = ones([2,2,2])
    A = getFaceInnerProduct(X,Y,Z,sigma)
    print(A)