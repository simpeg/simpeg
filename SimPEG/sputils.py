import numpy as np
from scipy import sparse

def ddx(n):
    """Define 1D derivatives"""
    return sparse.spdiags((np.ones((n+1,1))*[-1,1]).T, [0,1], n, n+1)
    
def sdiag(h):
    """Sparse diagonal matrix"""
    return sparse.spdiags(h, 0, np.size(h), np.size(h))    

def speye(n):
    """Sparse identity"""
    return sparse.identity(n)

def kron3(A, B, C):
    """Two kron prods"""
    return sparse.kron(sparse.kron(A, B), C)  