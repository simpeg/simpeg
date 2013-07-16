from scipy import sparse
from numpy import *


def ddx(n):
    """Define 1D derivatives"""
    # ddx = lambda n: sparse.spdiags((np.ones((n+1, 1))*[-1, 1]).T, [0, 1], n, n+1, format='csr')
    return sparse.spdiags(-ones(n), 0, n, n+1) + sparse.spdiags(ones(n+1), 1, n, n+1)


def av(n):
    """Define 1D average"""
    return 0.5*(sparse.spdiags(ones(n+1), 0, n, n+1) + sparse.spdiags(ones(n+1), 1, n, n+1))


def sdiag(h):
    """Diagonal matrix"""
    return sparse.spdiags(h, 0, size(h), size(h))


def speye(n):
    """sparse identity"""
    return sparse.spdiags(ones(n), 0, n, n)


def kron3(A, B, C):
    """two kron prods"""
    return sparse.kron(sparse.kron(A, B), C)


def appendBottom(A, B):
    """append on bottom"""
    C = sparse.vstack((A, B))
    C = C.tocsr()
    return C


def appendBottom3(A, B, C):
    """append on bottom"""
    C = appendBottom(appendBottom(A, B), C)
    C = C.tocsr()
    return C


def appendRight(A, B):
    """append on right"""
    C = sparse.hstack((A, B))
    C = C.tocsr()
    return C


def appendRight3(A, B, C):
    """append on right"""
    C = appendRight(appendRight(A, B), C)
    C = C.tocsr()
    return C


def blkDiag(A, B):
    """blockdigonal"""
    O12 = sparse.coo_matrix((shape(A)[0], shape(B)[1]))
    O21 = sparse.coo_matrix((shape(B)[0], shape(A)[1]))
    C = sparse.vstack((sparse.hstack((A, O12)), sparse.hstack((O21, B))))
    C = C.tocsr()
    return C


def blkDiag3(A, B, C):
    """blockdigonal 3"""
    ABC = blkDiag(blkDiag(A, B), C)
    ABC = ABC.tocsr()
    return ABC


def spzeros(n1, n2):
    """spzeros"""
    return sparse.coo_matrix((n1, n2))
