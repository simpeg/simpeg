import numpy as np
from scipy import sparse

def ddx(n):
    """Define 1D derivatives"""
    return sparse.spdiags((np.ones((n+1,1))*[-1,1]).T, [0,1], n, n+1, format="csr")
    
def sdiag(h):
    """Sparse diagonal matrix"""
    return sparse.spdiags(h, 0, np.size(h), np.size(h), format="csr")    

def speye(n):
    """Sparse identity"""
    return sparse.identity(n, format="csr")

def kron3(A, B, C):
    """Two kron prods"""
    return sparse.kron(sparse.kron(A, B), C, format="csr")

def spzeros(n1, n2):
    """spzeros"""
    return sparse.coo_matrix((n1, n2)).tocsr()

def av(n):
    """Define 1D averaging operator"""
    return sparse.spdiags((0.5*np.ones((n+1,1))*[1,1]).T, [0,1], n, n+1, format="csr")