import numpy as np
from scipy import sparse as sp


def sdiag(h):
    """Sparse diagonal matrix"""
    return sp.spdiags(h, 0, np.size(h), np.size(h), format="csr")


def speye(n):
    """Sparse identity"""
    return sp.identity(n, format="csr")


def kron3(A, B, C):
    """Three kron prods"""
    return sp.kron(sp.kron(A, B), C, format="csr")


def spzeros(n1, n2):
    """spzeros"""
    return sp.coo_matrix((n1, n2)).tocsr()


def appendBottom(A, B):
    """append on bottom"""
    C = sp.vstack((A, B))
    C = C.tocsr()
    return C


def appendBottom3(A, B, C):
    """append on bottom"""
    C = appendBottom(appendBottom(A, B), C)
    C = C.tocsr()
    return C


def appendRight(A, B):
    """append on right"""
    C = sp.hstack((A, B))
    C = C.tocsr()
    return C


def appendRight3(A, B, C):
    """append on right"""
    C = appendRight(appendRight(A, B), C)
    C = C.tocsr()
    return C


def blkDiag(A, B):
    """blockdigonal"""
    O12 = sp.coo_matrix((np.shape(A)[0], np.shape(B)[1]))
    O21 = sp.coo_matrix((np.shape(B)[0], np.shape(A)[1]))
    C = sp.vstack((sp.hstack((A, O12)), sp.hstack((O21, B))))
    C = C.tocsr()
    return C


def blkDiag3(A, B, C):
    """blockdigonal 3"""
    ABC = blkDiag(blkDiag(A, B), C)
    ABC = ABC.tocsr()
    return ABC
