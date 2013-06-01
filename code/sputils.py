from scipy.sparse import linalg
from scipy import sparse
from numpy import *


#======== Define 1D derivatives =============
def ddx(n):
    return sparse.spdiags(-ones(n),0,n,n+1) + sparse.spdiags(ones(n+1),1,n,n+1)

#======== Define 1D average =============
def av(n):
    return 0.5*(sparse.spdiags(ones(n+1),0,n,n+1) + sparse.spdiags(ones(n+1),1,n,n+1))

#======== Diagonal matrix =============
def sdiag(h):
    return sparse.spdiags(h,0,size(h),size(h))

#======== sparse identity =============
def speye(n):
    return sparse.spdiags(ones(n),0,n,n)

#======== two kron prods =============
def kron3(A,B,C):
    return sparse.kron(sparse.kron(A,B),C)
    
#======== append on bottom =============
def appendBottom(A,B):
    C = sparse.vstack((A,B))
    C = C.tocsr()
    return C    
    
#======== append on bottom =============
def appendBottom3(A,B,C):
    C = appendBottom(appendBottom(A,B),C)
    C = C.tocsr()
    return C    

#======== append on right =============
def appendRight(A,B):
    C = sparse.hstack((A,B))
    C = C.tocsr()
    return C    

#======== append on right =============
def appendRight3(A,B,C):
    C = appendRight(appendRight(A,B),C)
    C = C.tocsr()
    return C    

#======== blockdigonal =============
def blkDiag(A,B):
    O12 = sparse.coo_matrix((shape(A)[0],shape(B)[1]))
    O21 = sparse.coo_matrix((shape(B)[0],shape(A)[1]))
    C = sparse.vstack((sparse.hstack((A,O12)),sparse.hstack((O21,B))))
    C = C.tocsr()
    return C   
    
#======== blockdigonal 3 =============
def blkDiag3(A,B,C):
    ABC = blkDiag(blkDiag(A,B),C)
    ABC = ABC.tocsr()
    return ABC
 
#======== spzeros =============
def spzeros(n1,n2):
    return sparse.coo_matrix((n1,n2))
      	  	      	      	  	  	      	      	     	  	      	      	  	  	      	      	
