from scipy.sparse import linalg
from scipy import sparse
from sputils import *
from utils import *
from sputils import *
from numpy import *
from getEdgeTangent import *
from inv3X3BlockDiagonal import *
from getCellVolume import getCellVolume

# [A] = getEdgeInnerProduct(X,Y,Z,sigma)
# 

#      node(i,j,k+1) ------ edge2(i,j,k+1) ----- node(i,j+1,k+1)
#           /                                    /
#          /                                    / |
#      edge3(i,j,k)     face1(i,j,k)        edge3(i,j+1,k)
#        /                                    /   |
#       /                                    /    |
# node(i,j,k) ------ edge2(i,j,k) ----- node(i,j+1,k)
#      |                                     |    |
#      |                                     |   node(i+1,j+1,k+1)
#      |                                     |    /
# edge1(i,j,k)      face3(i,j,k)        edge1(i,j+1,k)
#      |                                     |  /
#      |                                     | /
#      |                                     |/
# node(i+1,j,k) ------ edge2(i+1,j,k) ----- node(i+1,j+1,k)

# no  | node        | e1          | e2          | e3
# 000 | i  ,j  ,k   | i  ,j  ,k   | i  ,j  ,k   | i  ,j  ,k
# 100 | i+1,j  ,k   | i  ,j  ,k   | i+1,j  ,k   | i+1,j  ,k
# 010 | i  ,j+1,k   | i  ,j+1,k   | i  ,j  ,k   | i  ,j+1,k
# 110 | i+1,j+1,k   | i  ,j+1,k   | i+1,j  ,k   | i+1,j+1,k
# 001 | i  ,j  ,k+1 | i  ,j  ,k+1 | i  ,j  ,k+1 | i  ,j  ,k
# 101 | i+1,j  ,k+1 | i  ,j  ,k+1 | i+1,j  ,k+1 | i+1,j  ,k
# 011 | i  ,j+1,k+1 | i  ,j+1,k+1 | i  ,j  ,k+1 | i  ,j+1,k
# 111 | i+1,j+1,k+1 | i  ,j+1,k+1 | i+1,j  ,k+1 | i+1,j+1,k


def subarray(T,i1,i2,i3):
    return take(take(take(T,i1,0),i2,1),i3,2)
    

def getEdgeInnerProduct(X,Y,Z,sigma):

    m  = array(shape(X))-1
    nc = prod(m)
    
    me1 = m + array([0, 1, 1]); ne1 = prod(me1)
    me2 = m + array([1, 0, 1]); ne2 = prod(me2)
    me3 = m + array([1, 1, 0]); ne3 = prod(me3)
    
    e1x,e1y,e1z,e2x,e2y,e2z,e3x,e3y,e3z,norme1,norme2,norme3 = getEdgeTangent(X,Y,Z)
    
    i = int64(linspace(0,m[0]-1,m[0]))
    j = int64(linspace(0,m[1]-1,m[1]))
    k = int64(linspace(0,m[2]-1,m[2]))
    
    ii,jj,kk = ndgrid(i,j,k) 
    ii = mkvc(ii); jj = mkvc(jj); kk = mkvc(kk)
    
    ## --------
    # no  | node        | e1          | e2          | e3
    # 000 | i  ,j  ,k   | i  ,j  ,k   | i  ,j  ,k   | i  ,j  ,k
    ind1 = sub2ind(me1,hstack3(ii,jj,kk)) 
    ind2 = sub2ind(me2,hstack3(ii,jj,kk)) + ne1
    ind3 = sub2ind(me3,hstack3(ii,jj,kk)) + ne1 + ne2
    
    IND = vstack((vstack((ind1,ind2)),ind3))
    IND = array(IND).flatten()
    
    P000 = sparse.coo_matrix((ones(3*nc),(linspace(0,3*nc-1,3*nc),IND)),shape=(3*nc,ne1+ne2+ne3)).tocsr()
    
    invT000 = inv3X3BlockDiagonal(subarray(e1x,i,j,k) , subarray(e1y,i,j,k), subarray(e1z,i,j,k),
                                  subarray(e2x,i,j,k) , subarray(e2y,i,j,k), subarray(e2z,i,j,k) ,
                                  subarray(e3x,i,j,k) , subarray(e3y,i,j,k), subarray(e3z,i,j,k) )
                                    
    ## --------
    # no  | node        | e1          | e2          | e3
    # 100 | i+1,j  ,k   | i  ,j  ,k   | i+1,j  ,k   | i+1,j  ,k
    ind1 = sub2ind(me1,hstack3(ii,jj,kk)) 
    ind2 = sub2ind(me2,hstack3(ii+1,jj,kk)) + ne1
    ind3 = sub2ind(me3,hstack3(ii+1,jj,kk)) + ne1 + ne2
    
    IND = vstack((vstack((ind1,ind2)),ind3))
    IND = array(IND).flatten()
    
    P100 = sparse.coo_matrix((ones(3*nc),(linspace(0,3*nc-1,3*nc),IND)),shape=(3*nc,ne1+ne2+ne3)).tocsr()
    
    invT100 = inv3X3BlockDiagonal(subarray(e1x,i,j,k), subarray(e1y,i,j,k),    subarray(e1z,i,j,k),
                                  subarray(e2x,i+1,j,k), subarray(e2y,i+1,j,k),  subarray(e2z,i+1,j,k),
                                  subarray(e3x,i+1,j,k) , subarray(e3y,i+1,j,k), subarray(e3z,i+1,j,k))
    
    ## --------
    # no  | node        | e1          | e2          | e3
    # 010 | i  ,j+1,k   | i  ,j+1,k   | i  ,j  ,k   | i  ,j+1,k
    ind1 = sub2ind(me1,hstack3(ii,jj+1,kk)) 
    ind2 = sub2ind(me2,hstack3(ii,jj,kk)) + ne1
    ind3 = sub2ind(me3,hstack3(ii,jj+1,kk)) + ne1 + ne2
    
    IND = vstack((vstack((ind1,ind2)),ind3))
    IND = array(IND).flatten()
    
    P010 = sparse.coo_matrix((ones(3*nc),(linspace(0,3*nc-1,3*nc),IND)),shape=(3*nc,ne1+ne2+ne3)).tocsr()
    
    invT010 = inv3X3BlockDiagonal(subarray(e1x,i,j+1,k) , subarray(e1y,i,j+1,k) , subarray(e1z,i,j+1,k) ,
                                subarray(e2x,i,j,k) , subarray(e2y,i,j,k) , subarray(e2z,i,j,k) ,
                                subarray(e3x,i,j+1,k) , subarray(e3y,i,j+1,k) ,subarray( e3z,i,j+1,k) )
    
    ## --------
    # no  | node        | e1          | e2          | e3
    # 110 | i+1,j+1,k   | i  ,j+1,k   | i+1,j  ,k   | i+1,j+1,k
    ind1 = sub2ind(me1,hstack3(ii,jj+1,kk)) 
    ind2 = sub2ind(me2,hstack3(ii+1,jj,kk)) + ne1
    ind3 = sub2ind(me3,hstack3(ii+1,jj+1,kk)) + ne1 + ne2
    
    IND = vstack((vstack((ind1,ind2)),ind3))
    IND = array(IND).flatten()
    
    P110 = sparse.coo_matrix((ones(3*nc),(linspace(0,3*nc-1,3*nc),IND)),shape=(3*nc,ne1+ne2+ne3)).tocsr()
    
    invT110 = inv3X3BlockDiagonal(subarray(e1x,i,j+1,k) ,subarray(e1y,i,j+1,k) , subarray(e1z,i,j+1,k) ,
                                subarray(e2x,i+1,j,k) ,subarray(e2y,i+1,j,k) , subarray(e2z,i+1,j,k),
                                subarray(e3x,i+1,j+1,k) ,subarray(e3y,i+1,j+1,k) , subarray(e3z,i+1,j+1,k) )
    
    ######
    
    ## --------
    # no  | node        | e1          | e2          | e3
    # 001 | i  ,j  ,k+1   | i  ,j  ,k+1   | i  ,j  ,k+1   | i  ,j  ,k
    ind1 = sub2ind(me1,hstack3(ii,jj,kk+1)) 
    ind2 = sub2ind(me2,hstack3(ii,jj,kk+1)) + ne1
    ind3 = sub2ind(me3,hstack3(ii,jj,kk)) + ne1 + ne2
    
    IND = vstack((vstack((ind1,ind2)),ind3))
    IND = array(IND).flatten()
    
    P001 = sparse.coo_matrix((ones(3*nc),(linspace(0,3*nc-1,3*nc),IND)),shape=(3*nc,ne1+ne2+ne3)).tocsr()
    
    invT001 = inv3X3BlockDiagonal(subarray(e1x,i,j,k+1) ,subarray(e1y,i,j,k+1) , subarray(e1z,i,j,k+1) ,
                                subarray(e2x,i,j,k+1) , subarray(e2y,i,j,k+1) , subarray(e2z,i,j,k+1) ,
                                subarray(e3x,i,j,k) , subarray(e3y,i,j,k) , subarray(e3z,i,j,k) )
                                
    ## --------
    # no  | node          | e1            | e2            | e3
    # 101 | i+1,j  ,k+1   | i  ,j  ,k+1   | i+1,j  ,k+1   | i+1,j  ,k+1
    ind1 = sub2ind(me1,hstack3(ii,jj,kk+1)) 
    ind2 = sub2ind(me2,hstack3(ii+1,jj,kk+1)) + ne1
    ind3 = sub2ind(me3,hstack3(ii+1,jj,kk)) + ne1 + ne2
    
    IND = vstack((vstack((ind1,ind2)),ind3))
    IND = array(IND).flatten()
    
    P101 = sparse.coo_matrix((ones(3*nc),(linspace(0,3*nc-1,3*nc),IND)),shape=(3*nc,ne1+ne2+ne3)).tocsr()
    
    invT101 = inv3X3BlockDiagonal(subarray(e1x,i,j,k+1), subarray(e1y,i,j,k+1), subarray(e1z,i,j,k+1) ,
                                subarray(e2x,i+1,j,k+1), subarray(e2y,i+1,j,k+1) , subarray(e2z,i+1,j,k+1) ,
                                subarray(e3x,i+1,j,k), subarray(e3y,i+1,j,k) , subarray(e3z,i+1,j,k) )
    
    ## --------
    # no  | node          | e1             | e2            | e3
    # 011 | i  ,j+1,k+1   | i  ,j+1,k+1   | i  ,j  ,k+1   | i  ,j+1,k+1
    ind1 = sub2ind(me1,hstack3(ii,jj+1,kk+1)) 
    ind2 = sub2ind(me2,hstack3(ii,jj,kk+1)) + ne1
    ind3 = sub2ind(me3,hstack3(ii,jj+1,kk)) + ne1 + ne2
    
    IND = vstack((vstack((ind1,ind2)),ind3))
    IND = array(IND).flatten()
    
    P011 = sparse.coo_matrix((ones(3*nc),(linspace(0,3*nc-1,3*nc),IND)),shape=(3*nc,ne1+ne2+ne3)).tocsr()
    
    invT011 = inv3X3BlockDiagonal(subarray(e1x,i,j+1,k+1) , subarray(e1y,i,j+1,k+1) , subarray(e1z,i,j+1,k+1) ,
                                subarray(e2x,i,j,k+1) , subarray(e2y,i,j,k+1) , subarray(e2z,i,j,k+1) ,
                                subarray(e3x,i,j+1,k) , subarray(e3y,i,j+1,k) , subarray(e3z,i,j+1,k) )
    
    ## --------
    # no  | node          | e1            | e2            | e3
    # 111 | i+1,j+1,k+1   | i  ,j+1,k+1   | i+1,j  ,k+1   | i+1,j+1,k+1
    ind1 = sub2ind(me1,hstack3(ii,jj+1,kk+1)) 
    ind2 = sub2ind(me2,hstack3(ii+1,jj,kk+1)) + ne1
    ind3 = sub2ind(me3,hstack3(ii+1,jj+1,kk)) + ne1 + ne2
    
    IND = vstack((vstack((ind1,ind2)),ind3))
    IND = array(IND).flatten()
    
    P111 = sparse.coo_matrix((ones(3*nc),(linspace(0,3*nc-1,3*nc),IND)),shape=(3*nc,ne1+ne2+ne3)).tocsr()
    
    invT111 = inv3X3BlockDiagonal(subarray(e1x,i,j+1,k+1) , subarray(e1y,i,j+1,k+1) , subarray(e1z,i,j+1,k+1) ,
                                subarray(e2x,i+1,j,k+1) , subarray(e2y,i+1,j,k+1) , subarray(e2z,i+1,j,k+1) ,
                                subarray(e3x,i+1,j+1,k) , subarray(e3y,i+1,j+1,k) , subarray(e3z,i+1,j+1,k) )
                                
    # Cell volume
    v  = mkvc(getCellVolume(X,Y,Z)) #mkvc(getVolume(X,Y,Z))
    vsig = v*mkvc(sigma)
    v3 = vstack((vstack((vsig,vsig)),vsig))
    v3 = v3.flatten()
    
    V = sdiag(v3)
    
    A = P000.T*invT000.T*V*invT000*P000 + P001.T*invT001.T*V*invT001*P001  + P010.T*invT010.T*V*invT010*P010 + P011.T*invT011.T*V*invT011*P011 +  P100.T*invT100.T*V*invT100*P100 + P101.T*invT101.T*V*invT101*P101 +  P110.T*invT110.T*V*invT110*P110 + P111.T*invT111.T*V*invT111*P111                                                        
    
    A = 0.125*A
    
    return A
    
    
if __name__ == '__main__':

    X,Y,Z = ndgrid(linspace(0,2,3),linspace(0,2,3),linspace(0,2,3))
    Z[2,2,2] = 2.5;   Z[0,0,0] = -0.5
    X[2,2,2] = 2.5; X[0,0,0] = -0.5
    sig = ones([2,2,2])
    A = getEdgeInnerProduct(X,Y,Z,sig)