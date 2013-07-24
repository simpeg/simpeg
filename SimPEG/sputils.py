from scipy import sparse as sp


def sdiag(h):
    """Sparse diagonal matrix"""
    return sp.spdiags(h, 0, h.size, h.size, format="csr")


def speye(n):
    """Sparse identity"""
    return sp.identity(n, format="csr")


def kron3(A, B, C):
    """Three kron prods"""
    return sp.kron(sp.kron(A, B), C, format="csr")


def spzeros(n1, n2):
    """spzeros"""
    return sp.coo_matrix((n1, n2)).tocsr()
