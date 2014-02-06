from scipy import sparse as sp
from matutils import mkvc
import numpy as np


def sdiag(h):
    """Sparse diagonal matrix"""
    return sp.spdiags(mkvc(h), 0, h.size, h.size, format="csr")

def sdInv(M):
    "Inverse of a sparse diagonal matrix"
    return sdiag(1/M.diagonal())

def speye(n):
    """Sparse identity"""
    return sp.identity(n, format="csr")


def kron3(A, B, C):
    """Three kron prods"""
    return sp.kron(sp.kron(A, B), C, format="csr")


def spzeros(n1, n2):
    """spzeros"""
    return sp.coo_matrix((n1, n2)).tocsr()


def ddx(n):
    """Define 1D derivatives, inner, this means we go from n+1 to n"""
    return sp.spdiags((np.ones((n+1, 1))*[-1, 1]).T, [0, 1], n, n+1, format="csr")


def av(n):
    """Define 1D averaging operator from nodes to cell-centers."""
    return sp.spdiags((0.5*np.ones((n+1, 1))*[1, 1]).T, [0, 1], n, n+1, format="csr")

def avExtrap(n):
    """Define 1D averaging operator from cell-centers to nodes."""
    Av = sp.spdiags((0.5*np.ones((n, 1))*[1, 1]).T, [-1, 0], n+1, n, format="csr") + sp.csr_matrix(([0.5,0.5],([0,n],[0,n-1])),shape=(n+1,n))
    return Av
