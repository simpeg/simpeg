from scipy import sparse as sp
from utils import mkvc


def sdiag(h):
    """Sparse diagonal matrix"""
    return sp.spdiags(mkvc(h), 0, h.size, h.size, format="csr")


def speye(n):
    """Sparse identity"""
    return sp.identity(n, format="csr")


def kron3(A, B, C):
    """Three kron prods"""
    return sp.kron(sp.kron(A, B), C, format="csr")


def spzeros(n1, n2):
    """spzeros"""
    return sp.coo_matrix((n1, n2)).tocsr()


def inv3X3BlockDiagonal(a11, a12, a13, a21, a22, a23, a31, a32, a33):
    """ B = inv3X3BlockDiagonal(a11, a12, a13, a21, a22, a23, a31, a32, a33)

    inverts a stack of 3x3 matrices

    Input:
     A   - a11, a12, a13, a21, a22, a23, a31, a32, a33

    Output:
     B   - inverse
     """

    a11 = mkvc(a11)
    a12 = mkvc(a12)
    a13 = mkvc(a13)
    a21 = mkvc(a21)
    a22 = mkvc(a22)
    a23 = mkvc(a23)
    a31 = mkvc(a31)
    a32 = mkvc(a32)
    a33 = mkvc(a33)

    detA = a31*a12*a23 - a31*a13*a22 - a21*a12*a33 + a21*a13*a32 + a11*a22*a33 - a11*a23*a32

    b11 = +(a22*a33 - a23*a32)/detA
    b12 = -(a12*a33 - a13*a32)/detA
    b13 = +(a12*a23 - a13*a22)/detA

    b21 = +(a31*a23 - a21*a33)/detA
    b22 = -(a31*a13 - a11*a33)/detA
    b23 = +(a21*a13 - a11*a23)/detA

    b31 = -(a31*a22 - a21*a32)/detA
    b32 = +(a31*a12 - a11*a32)/detA
    b33 = -(a21*a12 - a11*a22)/detA

    B = sp.vstack((sp.hstack((sdiag(b11), sdiag(b12),  sdiag(b13))),
                   sp.hstack((sdiag(b21), sdiag(b22),  sdiag(b23))),
                   sp.hstack((sdiag(b31), sdiag(b32),  sdiag(b33)))))

    return B


def inv2X2BlockDiagonal(a11, a12, a21, a22):
    """ B = inv2X2BlockDiagonal(a11, a12, a21, a22)

    Inverts a stack of 2x2 matrices by using the inversion formula

    inv(A) = (1/det(A)) * cof(A)^T

    Input:
    A   - a11, a12, a13, a21, a22, a23, a31, a32, a33

    Output:
    B   - inverse
    """

    a11 = mkvc(a11)
    a12 = mkvc(a12)
    a21 = mkvc(a21)
    a22 = mkvc(a22)

    # compute inverse of the determinant.
    detAinv = 1./(a11*a22 - a21*a12)

    b11 = +detAinv*a22
    b12 = -detAinv*a12
    b21 = -detAinv*a21
    b22 = +detAinv*a11

    B = sp.vstack((sp.hstack((sdiag(b11), sdiag(b12))),
                   sp.hstack((sdiag(b21), sdiag(b22)))))

    return B
